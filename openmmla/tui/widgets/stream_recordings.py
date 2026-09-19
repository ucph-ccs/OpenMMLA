"""Manage on the Streams tab: the recordings the streams of a card left on the
machines that capture them, host by host, with the room left there; the way
to copy them to this machine, to delete them, and how long they are kept.

Download File and Download Day copy whole recordings, tied to no session, into
artifacts/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/ here, with
the staged, resumable transfer of stream_export; a session's part of them is
Sessions -> Export Streams.

How long is `record_keep_days` of each stream (0 keeps them). A choice here
sets it for every stream of the card; the Launcher writes it into the config,
and the Streams tab deletes what is older at a stream's Start and at Refresh
(capture_recordings.prune). Nothing on a capture host does it by itself."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, ProgressBar, Select, Static
from textual.worker import Worker, WorkerState

from openmmla.tui import capture_recordings as cr
from openmmla.tui import download as dl
from openmmla.tui import stream_export
from openmmla.tui.artifacts import merge_tree
from openmmla.tui.recordings import human_size
from openmmla.tui.ssh import get_profile_by_name
from openmmla.tui.widgets.recordings_panel import AGE_CHOICES
from openmmla.utils.artifact_paths import CAPTURE_STREAMS_DIR, STREAMS_DIR, capture_copy_dir

# the Keep choice while the streams keep theirs for different spans
MIXED = -1

HELP = (
    "What the streams of this card recorded on the machines that capture them (Record on/off): one file per "
    "Start, filed under the day it started. A dim name is a stream that is not in this card's Streams (another "
    "card's, or one removed since).\n"
    "Download File copies the selected recording to this machine, Download Day every recording of its day on "
    "that host, whole and tied to no session; a copy already here in full is skipped, and one cut off (Cancel, "
    "a dropped link) resumes at the next press. A session's part of the streams is Sessions → Export Streams. "
    "The file a stream is writing now is neither copied nor deleted: Stop it first. Deleting a recording here "
    "frees its capture host; the copies on this machine stay.\n"
    "Keep recordings for sets record_keep_days of every stream here (Config tab): a recording last written "
    "longer ago is deleted at its stream's next Start and at Refresh on the Streams tab. The Stream Server's "
    "own recordings are on its card."
)

# where the copies go, beside the Download buttons
COPY_NOTE = "into artifacts/streams/capture/<day>/<host label>/ here"

# lines of a download's report kept under its progress row
DOWNLOAD_LINES = 8


class StreamRecordingsScreen(ModalScreen):

    DEFAULT_CSS = """
    StreamRecordingsScreen {
        align: center middle;
    }
    StreamRecordingsScreen > #sr-dialog {
        width: 94%;
        height: 90%;
        border: thick $primary;
        background: $surface;
        padding: 0 2;
    }
    StreamRecordingsScreen #sr-title {
        text-style: bold;
        margin-bottom: 1;
    }
    StreamRecordingsScreen .sr-muted {
        color: $text-muted;
    }
    StreamRecordingsScreen #sr-summary {
        margin-top: 1;
        text-style: bold;
    }
    StreamRecordingsScreen #sr-table {
        height: 1fr;
        margin-top: 1;
    }
    StreamRecordingsScreen .sr-row {
        height: auto;
        margin-top: 1;
    }
    StreamRecordingsScreen .sr-row Label {
        padding-top: 1;
        margin-right: 1;
    }
    StreamRecordingsScreen .sr-row Button {
        margin-right: 1;
    }
    StreamRecordingsScreen #sr-keep {
        width: 36;
        margin-right: 1;
    }
    StreamRecordingsScreen #sr-age {
        width: 16;
        margin-right: 1;
    }
    StreamRecordingsScreen #sr-keep-note, StreamRecordingsScreen #sr-copy-note {
        width: 1fr;
        padding-top: 1;
    }
    StreamRecordingsScreen #sr-log {
        height: auto;
        margin-top: 1;
    }
    /* a download's progress: there only while one runs */
    StreamRecordingsScreen #sr-progress {
        height: auto;
        min-height: 1;
        display: none;
    }
    StreamRecordingsScreen #sr-progress.active {
        display: block;
    }
    StreamRecordingsScreen #sr-progress-label {
        width: 30;
        color: $text-muted;
    }
    StreamRecordingsScreen #sr-progress ProgressBar {
        width: 1fr;
    }
    StreamRecordingsScreen #sr-progress Bar {
        width: 1fr;
    }
    StreamRecordingsScreen #sr-progress-detail {
        width: 34;
        color: $text-muted;
        text-align: right;
    }
    StreamRecordingsScreen #btn-sr-dl-cancel {
        width: 9;
        min-width: 9;
        height: auto;
        min-height: 1;
        margin-left: 1;
    }
    StreamRecordingsScreen #sr-dl-log {
        height: auto;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(
        self,
        streams: Callable[[], list[cr.CaptureStream]],
        live_paths: Callable[[], Iterable[str]] | None = None,
        on_keep_days: Callable[[int], None] | None = None,
        project_root: str | os.PathLike[str] | None = None,
    ) -> None:
        """`streams` gives the card's managed streams as they are now, `live_paths`
        the files the console noted its running streams write, and `on_keep_days`
        is asked to store a new keep time for all of them. `project_root` holds
        the artifacts/ the Download buttons copy into."""
        super().__init__()
        self._stream_source = streams
        self._live_source = live_paths or (lambda: ())
        self._on_keep_days = on_keep_days
        self._project_root = project_root
        self._streams: list[cr.CaptureStream] = list(streams())
        self._listings: dict[cr.CaptureFolder, cr.Listing | None] = {}
        self._rows: list[cr.CaptureFile] = []
        # a deletion waiting for its second press: which button, and the very files it named
        self._pending: tuple | None = None
        self._deleting = False
        self._keep_asked: int | None = None
        # the download running now (one at a time), what it said, and a Close that asked to stop it
        self._download_worker = None
        self._download_lines: list[str] = []
        self._progress_total = 0
        self._close_armed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="sr-dialog"):
            yield Static("Recordings on the capture hosts", id="sr-title")
            yield Static(HELP, classes="sr-muted")
            yield Static("", id="sr-summary", classes="sr-muted")
            yield DataTable(id="sr-table")
            with Horizontal(classes="sr-row"):
                yield Label("Copy to this machine:")
                yield Button("Download File", variant="warning", id="btn-sr-dl-file")
                yield Button("Download Day", variant="warning", id="btn-sr-dl-day")
                yield Static(COPY_NOTE, id="sr-copy-note", classes="sr-muted")
            with Horizontal(id="sr-progress"):
                yield Static("", id="sr-progress-label")
                # show_eta=False: the transfer's own detail carries the rate and the time left
                yield ProgressBar(total=None, show_eta=False, id="sr-progress-bar")
                yield Static("", id="sr-progress-detail")
                yield Button("Cancel", variant="error", compact=True, id="btn-sr-dl-cancel")
            yield Static("", id="sr-dl-log", classes="sr-muted")
            with Horizontal(classes="sr-row"):
                yield Label("Keep recordings for:")
                options, value = self._keep_options()
                yield Select(options, value=value, allow_blank=False, id="sr-keep")
                yield Button("Delete Expired", variant="error", id="btn-sr-expired")
                yield Static("", id="sr-keep-note", classes="sr-muted")
            with Horizontal(classes="sr-row"):
                yield Button("Refresh", variant="primary", id="btn-sr-refresh")
                yield Button("Delete File", variant="error", id="btn-sr-file")
                yield Button("Delete Day", variant="error", id="btn-sr-day")
                yield Button("Delete Older Than", variant="error", id="btn-sr-older")
                yield Select([(label, seconds) for label, seconds in AGE_CHOICES], value=AGE_CHOICES[1][1],
                             allow_blank=False, id="sr-age")
                yield Button("Close", id="btn-sr-close")
            yield Static("", id="sr-log", classes="sr-muted")

    def on_mount(self) -> None:
        table = self.query_one("#sr-table", DataTable)
        table.add_columns("Host", "Day", "Stream", "Started", "Last write", "Length", "Size", "State")
        table.cursor_type = "row"
        table.focus()
        self.reload()

    def action_close(self) -> None:
        if self._downloading and not self._close_armed:
            # closing the screen stops its download, so that takes a second Close (or Escape)
            self._close_armed = True
            self._add_download_line(
                "[yellow]A download is running: closing stops it. What arrived is kept, and the next Download File "
                "or Download Day resumes it. Press Close again to close.[/yellow]")
            return
        self._cancel_download()
        self.dismiss(None)

    # ---- asking the hosts ----

    def reload(self, note: str = "") -> None:
        self._pending = None
        self._streams = list(self._stream_source())
        folders = list(cr.by_folder(self._streams))
        self._set_log(note or f"Asking {', '.join(escape(folder.where) for folder in folders)}...")
        self.run_worker(self._load(folders, note), exclusive=True, group="sr-work")

    async def _load(self, folders: list[cr.CaptureFolder], note: str = "") -> None:
        listings = await asyncio.gather(*(asyncio.to_thread(cr.list_folder, folder) for folder in folders))
        self._listings = dict(zip(folders, listings))
        # a prompt made from the listing before is gone with it
        self._pending = None
        self._show_keep()
        self._show()
        silent = [folder.where for folder, listing in self._listings.items() if listing is None]
        if silent:
            note = (note + "\n" if note else "") + (
                f"[yellow]No answer from {escape(', '.join(silent))}: switched off, off the network, or its SSH "
                "profile has changed. Its recordings stay where they are.[/yellow]")
        self._set_log(note)

    # ---- showing it ----

    def _known(self, item: cr.CaptureFile) -> bool:
        return any((s.folder, s.name, s.kind) == (item.folder, item.stream, item.kind) for s in self._streams)

    def _live(self, item: cr.CaptureFile) -> bool:
        return cr.is_live(item, self._live_source())

    def _writing(self, item: cr.CaptureFile) -> bool:
        """an ffmpeg names it, or the console noted it as a running stream's file."""
        return item.writing or item.path in set(self._live_source())

    def _state(self, item: cr.CaptureFile, now: float) -> Text | str:
        if self._writing(item):
            return Text("recording now", "green")
        if self._live(item):
            # nothing writes it any more, it was written moments ago: a stream just stopped
            return Text(f"written {max(0, int(now - item.written))} s ago", "yellow")
        until = cr.expires(item, self._streams)
        if until is None:
            return ""
        return Text("expired", "yellow") if now >= until else f"until {cr.stamp(until)}"

    def _show(self) -> None:
        table = self.query_one("#sr-table", DataTable)
        selected = self._selected()
        table.clear()
        self._rows = []
        for folder, listing in self._listings.items():
            if listing is None:
                continue
            for item in sorted(listing.files, key=lambda f: (f.day, f.start or f.written), reverse=True):
                self._rows.append(item)
                length = item.written - item.start if item.start else None
                # Text, not str: the table reads a str as markup, and names are anyone's
                table.add_row(
                    Text(folder.where), Text(cr.day_label(item.day)),
                    Text(item.stream) if self._known(item) else Text(item.stream, "dim"),
                    cr.stamp(item.start), cr.stamp(item.written), cr.span(length), human_size(item.size),
                    self._state(item, listing.now),
                )
        if selected is not None:
            for row, item in enumerate(self._rows):
                if item.path == selected.path and item.folder == selected.folder:
                    table.move_cursor(row=row)
                    break
        parts = []
        for folder, listing in self._listings.items():
            if listing is None:
                parts.append(f"{folder.where}: no answer")
                continue
            held = (f"{len(listing.files)} file(s), {human_size(sum(f.size for f in listing.files))}"
                    if listing.files else "nothing recorded")
            free = f", {human_size(listing.free)} free" if listing.free is not None else ""
            parts.append(f"{folder.where}: {held}{free}")
        self.query_one("#sr-summary", Static).update(escape(" · ".join(parts)) if parts else "")
        self._show_keep_note()

    def _keep_options(self) -> tuple[list[tuple[str, int]], int]:
        """the Keep choices and the one that holds now: the streams' common
        keep time, as written when it is none of the offered spans."""
        options = list(cr.KEEP_CHOICES)
        spans = {stream.keep_days for stream in self._streams}
        if len(spans) > 1:
            options.append(("differs per stream", MIXED))
            return options, MIXED
        days = spans.pop() if spans else 0
        if days not in {value for _label, value in options}:
            options.append((f"{days} days (as written)", days))
        return options, days

    def _show_keep(self) -> None:
        select = self.query_one("#sr-keep", Select)
        options, value = self._keep_options()
        with select.prevent(Select.Changed):
            select.set_options(options)
            select.value = value
        self._show_keep_note()

    def _expired(self) -> list[cr.CaptureFile]:
        live = set(self._live_source())
        return [item for listing in self._listings.values() if listing is not None
                for item in cr.due(listing, self._streams, live)]

    def _show_keep_note(self) -> None:
        try:
            note = self.query_one("#sr-keep-note", Static)
        except Exception:
            return
        if not any(stream.keep_days > 0 for stream in self._streams):
            note.update("kept until deleted here")
            return
        expired = self._expired()
        note.update(
            f"{len(expired)} recording(s) past it, {human_size(sum(f.size for f in expired))}: they go at the next "
            "Start or Refresh, or now with Delete Expired" if expired else "nothing is past it"
        )

    def _selected(self) -> cr.CaptureFile | None:
        table = self.query_one("#sr-table", DataTable)
        row = table.cursor_row
        if not self._rows or row is None or not 0 <= row < len(self._rows):
            return None
        return self._rows[row]

    def _set_log(self, text: str) -> None:
        try:
            self.query_one("#sr-log", Static).update(text)
        except Exception:
            pass

    # ---- how long they stay ----

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "sr-keep":
            return
        event.stop()
        days = event.value
        if days is Select.BLANK or days == MIXED:
            return
        days = int(days)
        if {stream.keep_days for stream in self._streams} == {days}:
            return
        if self._on_keep_days is not None:
            self._on_keep_days(days)
        self._pending = None
        self._keep_asked = days
        self._streams = [replace(stream, keep_days=days) for stream in self._streams]
        # one span for all now: `differs per stream` goes from the choices
        self._show_keep()
        self._show()
        self._set_log(
            f"Every stream here keeps its recordings {'until deleted' if not days else f'for {days} day(s)'}"
            + (": older ones go at a stream's next Start and at Refresh on the Streams tab." if days else ".")
        )
        # the Launcher writes the config; a stream it could not find there keeps its old value
        self.set_timer(1.0, self._check_keep_written)

    def _check_keep_written(self) -> None:
        stored = list(self._stream_source())
        if self._keep_asked is not None and {stream.keep_days for stream in stored} != {self._keep_asked}:
            self._streams = stored
            self._show_keep()
            self._show()
            self._set_log("[red]The keep time was not stored for every stream: the Launcher's log says why.[/red]")

    # ---- making room ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id or ""
        event.stop()
        if button == "btn-sr-close":
            self.action_close()
        elif button == "btn-sr-dl-cancel":
            self._cancel_download()
        elif self._deleting:
            # a Refresh would drop its report, and a second one would pile onto the first
            self._set_log("A deletion is running: its report and a fresh listing follow in a moment.")
        elif button == "btn-sr-refresh":
            self.reload()
        elif button in ("btn-sr-dl-file", "btn-sr-dl-day"):
            self._download(whole_day=button == "btn-sr-dl-day")
        elif self._downloading:
            # rsync gives up on a file deleted under it, and so the whole download
            self._pending = None
            self._set_log("A download is running: delete once it is done or cancelled.")
        elif button == "btn-sr-file":
            item = self._selected()
            if item is None:
                self._set_log("Choose a recording in the table first.")
            elif self._writing(item):
                self._pending = None
                self._set_log(
                    f"[yellow]{escape(item.stream)} is writing this file now. Stop it on the Streams tab first: "
                    "deleted while it runs, the file frees no room and the take is lost.[/yellow]")
            elif self._live(item):
                self._pending = None
                self._set_log(
                    "[yellow]This file was written moments ago, though nothing on its host writes it now: its "
                    f"stream has just stopped. Refresh in {cr.LIVE_SECONDS} seconds and delete it then.[/yellow]")
            else:
                self._confirm_or_run(
                    ("file", item.folder, item.path), [item], "Delete File",
                    f"{escape(item.path.rsplit('/', 1)[-1])} on {escape(item.folder.where)} ({human_size(item.size)})")
        elif button == "btn-sr-day":
            item = self._selected()
            if item is None:
                self._set_log("Choose a recording of that day in the table first.")
                return
            listing = self._listings.get(item.folder)
            files = [f for f in (listing.files if listing else []) if f.day == item.day]
            plan = [f for f in files if not self._live(f)]
            kept = [f.stream for f in files if self._live(f)]
            what = (f"the {len(plan)} recording(s) of {cr.day_label(item.day)} on {escape(item.folder.where)} "
                    f"({human_size(sum(f.size for f in plan))})")
            if kept:
                what += f"; the one {escape(', '.join(kept))} is writing now stays"
            self._confirm_or_run(("day", item.folder, item.day), plan, "Delete Day", what)
        elif button == "btn-sr-older":
            age = self.query_one("#sr-age", Select).value
            if age is Select.BLANK:
                return
            listings = [listing for listing in self._listings.values() if listing is not None]
            plan = cr.older_than(listings, float(age), self._live_source())
            hosts = sorted({f.folder.where for f in plan})
            self._confirm_or_run(
                ("older", float(age)), plan, "Delete Older Than",
                f"{len(plan)} recording(s) last written more than {int(float(age) // cr.DAY_SECONDS)} day(s) ago"
                + (f" on {escape(', '.join(hosts))}" if hosts else "")
                + f" ({human_size(sum(f.size for f in plan))}), whichever stream made them")
        elif button == "btn-sr-expired":
            plan = self._expired()
            self._confirm_or_run(
                ("expired",), plan, "Delete Expired",
                f"{len(plan)} recording(s) past their stream's keep time ({human_size(sum(f.size for f in plan))})")

    def _confirm_or_run(self, key: tuple, plan: list[cr.CaptureFile], button: str, what: str) -> None:
        # the second press deletes only what the first one named: any other plan asks again
        key = key + (tuple(sorted((f.folder.ssh_profile, f.folder.record_root, f.path) for f in plan)),)
        if not plan:
            self._pending = None
            self._set_log(f"Nothing to delete: {what}.")
            return
        if self._pending != key:
            self._pending = key
            self._set_log(
                f"[red]This deletes {what} from the capture host's disk, for good: Download File or Download Day "
                f"has to have copied what is still wanted. Press {button} again to confirm.[/red]")
            return
        self._pending = None
        self._deleting = True
        self._set_log(f"Deleting {len(plan)} recording(s)...")
        self.run_worker(self._delete(plan), exclusive=True, group="sr-work")

    async def _delete(self, plan: list[cr.CaptureFile]) -> None:
        try:
            results = await asyncio.to_thread(cr.delete_files, plan)
        finally:
            self._deleting = False
        self.reload(describe(results))

    # ---- copies on this machine ----

    @property
    def _downloading(self) -> bool:
        return self._download_worker is not None

    def _download(self, whole_day: bool) -> None:
        """Download File and Download Day: the selected recording, or every one
        of its day on its host, copied here unless a copy is here in full. The
        file a stream is writing stays out: its copy would stop short."""
        item = self._selected()
        if item is None:
            self._set_download_log(
                "Choose a recording of that day in the table first." if whole_day
                else "Choose a recording in the table first.")
            return
        if self._downloading:
            self._add_download_line("[yellow]A download is running: Cancel it, or wait until it is done.[/yellow]")
            return
        if item.folder.ssh_profile == "local":
            folder = (_day_dir(item) or "") if whole_day else item.path.rsplit("/", 1)[0]
            self._set_download_log(
                f"[cyan]{'The recordings of that day are' if whole_day else 'This recording is'} on this machine "
                f"already, in {escape(folder)}/: there is nothing to copy.[/cyan]")
            return
        if self._project_root is None:
            self._set_download_log("[red]This screen was not told where the console's artifacts/ folder is, so it "
                                   "cannot copy anything. Open Manage from the Streams tab again.[/red]")
            return
        where = escape(item.folder.where)
        if not whole_day:
            if self._writing(item):
                self._set_download_log(
                    f"[yellow]{escape(item.stream)} is writing this file now, and a copy taken now would stop "
                    "short. Stop it on the Streams tab first, then press Download File.[/yellow]")
            elif self._live(item):
                self._set_download_log(
                    "[yellow]This file was written moments ago, though nothing on its host writes it now: its "
                    f"stream has just stopped. Refresh in {cr.LIVE_SECONDS} seconds and download it then.[/yellow]")
            elif is_here(self._project_root, item):
                self._set_download_log(
                    f"[green]{escape(file_name(item))} is here already, in full: "
                    f"{escape(str(copy_dir(self._project_root, item)))}/[/green]")
            else:
                self._start_download(
                    [item], f"[cyan]Copying {escape(file_name(item))} ({human_size(item.size)}) from {where}...[/cyan]")
            return
        listing = self._listings.get(item.folder)
        files = [f for f in (listing.files if listing else []) if f.day == item.day]
        busy = [f for f in files if self._live(f)]
        here = [f for f in files if f not in busy and is_here(self._project_root, f)]
        todo = [f for f in files if f not in busy and f not in here]
        day = escape(cr.day_label(item.day))
        notes = []
        if here:
            notes.append(f"{len(here)} recording(s) of {day} are here already, in full, and are left out.")
        if busy:
            notes.append(f"[yellow]Left out, being written now: {escape(', '.join(f.stream for f in busy))}. "
                         "Download it once its stream has stopped.[/yellow]")
        if not todo:
            folder = escape(str(capture_copy_dir(self._project_root, item.day, item.folder.host_label)))
            self._set_download_log(*notes, f"Nothing else of {day} on {where} to copy (here: {folder}/).")
            return
        self._start_download(
            todo,
            f"[cyan]Copying the {len(todo)} recording(s) of {day} on {where} "
            f"({human_size(sum(f.size for f in todo))})...[/cyan]",
            *notes)

    def _start_download(self, items: list[cr.CaptureFile], *lines: str) -> None:
        self._set_download_log(*lines)
        self._close_armed = False
        # its own group: a Refresh (sr-work) must not cancel it
        self._download_worker = self.run_worker(
            copy_recordings(self._project_root, items, self._download_callbacks()),
            group="sr-download", exclusive=True, exit_on_error=False)

    def _cancel_download(self) -> None:
        if self._download_worker is not None:
            self._download_worker.cancel()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """a download is over, however it ended; one cancelled before it began
        never ran a line of its own, so this is where that is said."""
        worker = event.worker
        if worker is not self._download_worker or not worker.is_finished:
            return
        self._download_worker = None
        self._close_armed = False
        self._progress_end()
        if worker.state == WorkerState.CANCELLED:
            self._add_download_line(
                "[yellow]Download cancelled. What arrived is kept, and the next Download File or Download Day "
                "resumes it.[/yellow]")
        elif worker.state == WorkerState.ERROR:
            self._add_download_line(f"[red]The download stopped on an error: {escape(str(worker.error))}[/red]")

    def _download_callbacks(self) -> stream_export.ExportCallbacks:
        """the transfer's report goes under the progress row; Cancel cancels its worker."""
        return stream_export.ExportCallbacks(
            log=self._add_download_line,
            progress_start=self._progress_start,
            progress_update=self._progress_update,
            progress_end=self._progress_end,
        )

    def _set_download_log(self, *lines: str) -> None:
        self._download_lines = [line for line in lines if line]
        self._show_download_log()

    def _add_download_line(self, line: str) -> None:
        self._download_lines.append(line)
        self._show_download_log()

    def _show_download_log(self) -> None:
        self._download_lines = self._download_lines[-DOWNLOAD_LINES:]
        try:
            self.query_one("#sr-dl-log", Static).update("\n".join(self._download_lines))
        except Exception:
            pass

    def _progress_start(self, label: str, total: int | None) -> None:
        self._progress_total = int(total or 0)
        try:
            bar = self.query_one("#sr-progress-bar", ProgressBar)
            # clearing the total first restarts the bar
            bar.update(total=None)
            bar.update(total=float(total) if total else None, progress=0)
            self.query_one("#sr-progress-label", Static).update(Text(label))
            self.query_one("#sr-progress-detail", Static).update("")
            self.query_one("#sr-progress").add_class("active")
        except Exception:
            pass

    def _progress_update(self, done: int, total: int, detail: str) -> None:
        if total:
            self._progress_total = int(total)
        # a file that grew cannot push the bar past its end
        value = max(0, int(done))
        if self._progress_total:
            value = min(value, self._progress_total)
        try:
            bar = self.query_one("#sr-progress-bar", ProgressBar)
            if self._progress_total and bar.total != float(self._progress_total):
                bar.update(total=float(self._progress_total))
            bar.update(progress=value)
            self.query_one("#sr-progress-detail", Static).update(Text(detail))
        except Exception:
            pass

    def _progress_end(self) -> None:
        self._progress_total = 0
        try:
            self.query_one("#sr-progress").remove_class("active")
            self.query_one("#sr-progress-bar", ProgressBar).update(total=None, progress=0)
            self.query_one("#sr-progress-detail", Static).update("")
        except Exception:
            pass


def describe(results: list[cr.Deletion]) -> str:
    """what the delete scripts did, for the line under the table."""
    removed = [entry for result in results for entry in result.removed]
    lines = [f"Deleted {len(removed)} recording(s), {human_size(sum(size for _path, size in removed))} freed."]
    for result in results:
        where = escape(result.folder.where)
        if not result.answered:
            lines.append(f"[yellow]{where} did not finish: Refresh shows what is left.[/yellow]")
        if result.live:
            lines.append(f"[yellow]Left on {where}, being written now or seconds ago: {len(result.live)}.[/yellow]")
        if result.failed:
            lines.append(f"[red]Could not delete on {where}: "
                         f"{escape(', '.join(p.rsplit('/', 1)[-1] for p in result.failed))} (permissions?).[/red]")
        if result.refused:
            lines.append(f"[red]Refused on {where}, not a stream recording of its folder: {len(result.refused)}.[/red]")
    return "\n".join(lines)


# ---- copies on this machine (no Textual below) ----

def file_name(item: cr.CaptureFile) -> str:
    return item.path.rsplit("/", 1)[-1]


def copy_dir(project_root, item: cr.CaptureFile) -> Path:
    """where Download File and Download Day put a recording here:
    artifacts/streams/capture/<day>/<host label>/<video|audio>/."""
    return capture_copy_dir(project_root, item.day, item.folder.host_label) / item.kind


def is_here(project_root, item: cr.CaptureFile) -> bool:
    """a copy at the size its host listed: a recording is named after its
    start and not written again once it is over, so that is the file."""
    try:
        return (copy_dir(project_root, item) / file_name(item)).stat().st_size == item.size
    except OSError:
        return False


def _day_dir(item: cr.CaptureFile) -> str | None:
    """<record root>/streams/capture/<day>/<host label> on the recording's
    host, the folder above its video/ or audio/."""
    parts = item.path.rsplit("/", 2)
    if len(parts) != 3 or parts[1] != item.kind or not parts[0].startswith("/"):
        return None
    return parts[0]


async def copy_recordings(
    project_root,
    items: list[cr.CaptureFile],
    callbacks: stream_export.ExportCallbacks | None = None,
    *,
    fetch: Callable[..., Awaitable[bool]] | None = None,
) -> bool:
    """copy recordings of one capture host and one day, whole and tied to no
    session, into artifacts/streams/capture/<day>/<host label>/<video|audio>/
    here: the day's folder on the host is transferred with only these files
    in it (fetch_files), staged under artifacts/streams/.staging/ as
    stream_export.copy_capture_days stages a day. True when they are all
    here afterwards."""
    callbacks = callbacks or stream_export.ExportCallbacks()
    fetch = fetch or fetch_files
    if not items:
        return True
    first = items[0]
    folder, day, remote_dir = first.folder, first.day, _day_dir(first)
    if remote_dir is None or any((f.folder, f.day, _day_dir(f)) != (folder, day, remote_dir) for f in items):
        raise ValueError("copy_recordings takes the recordings of one capture host and one day")
    profile = get_profile_by_name(folder.ssh_profile)
    if profile is None:
        callbacks.log(f"[red]SSH profile '{escape(folder.ssh_profile)}' not found: add it again under System "
                      "Settings → SSH Profiles, or pick the stream's new machine on the Streams tab.[/red]")
        return False
    label = folder.host_label
    staging = dl.staging_root(project_root, STREAMS_DIR, CAPTURE_STREAMS_DIR, day, label)
    if len(items) == 1:
        # a file of its own: the staging of a day holds what that day's plan names
        # alone, so a second file's download would throw away this one's partial copy
        staging = staging / f"file-{items[0].kind}-{file_name(items[0])}"
    return await fetch(
        profile, remote_dir, capture_copy_dir(project_root, day, label),
        staging=staging,
        label=label, where=folder.ssh_profile, what=day, callbacks=callbacks,
        only=[f"{item.kind}/{file_name(item)}" for item in items],
    )


def only_files(plan: dl.RemotePlan, wanted) -> dl.RemotePlan:
    """the plan of these files of a remote tree alone: the rest are left out
    as though they were here, so rsync is told to skip them."""
    wanted = set(wanted)
    kept = dl.leave_out(plan, [item.rel for item in plan.files if item.rel not in wanted])
    # leave_out keeps a name rsync would read as a pattern; it is not planned all the same
    kept.files = [item for item in kept.files if item.rel in wanted]
    kept.file_count, kept.total_bytes = len(kept.files), sum(item.size for item in kept.files)
    return kept


def _merge_planned(staged: Path, local_dir: Path, plan: dl.RemotePlan, where: str) -> dict[str, int]:
    """merge the planned files alone: a recursive scp brings the whole tree,
    the file a stream is writing too, and that must not land here cut short."""
    planned = {item.rel for item in plan.files}
    for root, _dirs, names in os.walk(staged):
        for name in names:
            path = Path(root) / name
            if path.relative_to(staged).as_posix() not in planned:
                path.unlink(missing_ok=True)
    return merge_tree(staged, local_dir, conflict_label=where)


async def fetch_files(
    profile,
    remote_dir: str,
    local_dir: Path,
    *,
    staging: Path,
    label: str,
    where: str,
    callbacks: stream_export.ExportCallbacks,
    only,
    what: str | None = None,
) -> bool:
    """stream_export.fetch_tree for some files of a remote tree: `only` are
    their paths below `remote_dir`, and nothing else of it is fetched or
    merged. The same scan, staging, resume, completeness gate, report and
    Cancel; a file already here at the remote size is not fetched again."""
    wanted = set(only)
    log = callbacks.log
    log(f"[cyan]Downloading {escape(where)}:{escape(remote_dir)} -> {escape(str(local_dir))}[/cyan]")
    # the row goes up before the scan, so Cancel is there while the host is asked
    callbacks.progress_start(f"{label} · scanning…", None)
    try:
        plan = await dl.probe_remote(profile, remote_dir)
        if plan.unreachable:
            log(f"[red]Could not reach {escape(where)}: {escape(plan.unreachable)}[/red]")
            return False
        if not plan.exists:
            log(f"[red]Remote folder not found: {escape(remote_dir)} (deleted since the listing? Refresh).[/red]")
            return False
        if not plan.exact:
            log(f"[red]{escape(remote_dir)} holds more files than a download lists one by one, so these cannot "
                "be picked out of it.[/red]")
            return False
        gone = sorted(wanted - {item.rel for item in plan.files})
        if gone:
            log(f"[yellow]Not found on {escape(where)} (deleted since the listing? Refresh shows what is there): "
                f"{escape(', '.join(gone))}.[/yellow]")
        plan = only_files(plan, wanted)
        if not plan.file_count:
            log("[yellow]Nothing left to download.[/yellow]")
            return False
        here = await asyncio.to_thread(stream_export.files_already_here, local_dir, plan)
        stream_export.log_plan(log, plan, here)
        if here:
            plan = only_files(plan, wanted - set(here))
            if not plan.file_count:
                log(f"[green]{escape(what or label)}: all {len(here)} file(s) are already here; nothing to fetch."
                    "[/green]")
                return True
            log(f"  [cyan]{len(here)} file(s) are already here at the same size and are not fetched again; "
                f"fetching {stream_export.describe_files(item.rel for item in plan.files)}.[/cyan]")
        if callbacks.cancelled():
            raise asyncio.CancelledError()
        callbacks.progress_start(
            f"{label} · {stream_export.describe_files(item.rel for item in plan.files)}", plan.total_bytes)
        result = await dl.download_tree(
            profile, remote_dir, staging=staging, plan=plan, on_progress=callbacks.progress_update, log=log)
    finally:
        callbacks.progress_end()

    if result.status != "complete":
        stream_export.report_incomplete(log, result)
        return False
    stats = await asyncio.to_thread(_merge_planned, result.staged_root, local_dir, plan, where)
    await asyncio.to_thread(dl.finalize, staging)
    log(
        f"[green]Downloaded {stream_export.describe_files(item.rel for item in plan.files)} to "
        f"{escape(str(local_dir))} via {result.tool} (copied {stats['copied']}, unchanged {stats['skipped']}, "
        f"conflicts {stats['conflicted']})" + (f"; {len(here)} file(s) were already here" if here else "") + ".[/green]"
    )
    return True
