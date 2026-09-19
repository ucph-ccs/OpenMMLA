"""Manage on the Streams tab: the recordings the streams of a card left on the
machines that capture them, host by host, with the room left there; the way
to delete them, and how long they are kept. It copies nothing to this
machine: a session's part of them is Sessions -> Export Streams.

How long is `record_keep_days` of each stream (0 keeps them). A choice here
sets it for every stream of the card; the Launcher writes it into the config,
and the Streams tab deletes what is older at a stream's Start and at Refresh
(capture_recordings.prune). Nothing on a capture host does it by itself."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Callable, Iterable

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Select, Static

from openmmla.tui import capture_recordings as cr
from openmmla.tui.recordings import human_size
from openmmla.tui.widgets.recordings_panel import AGE_CHOICES

# the Keep choice while the streams keep theirs for different spans
MIXED = -1

HELP = (
    "What the streams of this card recorded on the machines that capture them (Record on/off): one file per "
    "Start, filed under the day it started. A dim name is a stream that is not in this card's Streams (another "
    "card's, or one removed since).\n"
    "The file a stream is writing now is never deleted: Stop it first. A session's part of the streams is "
    "copied to this machine with Sessions → Export Streams.\n"
    "Keep recordings for sets record_keep_days of every stream here (Config tab): a recording last written "
    "longer ago is deleted at its stream's next Start and at Refresh on the Streams tab. The Stream Server's "
    "own recordings are on its card."
)


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
    StreamRecordingsScreen #sr-keep-note {
        width: 1fr;
        padding-top: 1;
    }
    StreamRecordingsScreen #sr-log {
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(
        self,
        streams: Callable[[], list[cr.CaptureStream]],
        live_paths: Callable[[], Iterable[str]] | None = None,
        on_keep_days: Callable[[int], None] | None = None,
    ) -> None:
        """`streams` gives the card's managed streams as they are now, `live_paths`
        the files the console noted its running streams write, and `on_keep_days`
        is asked to store a new keep time for all of them."""
        super().__init__()
        self._stream_source = streams
        self._live_source = live_paths or (lambda: ())
        self._on_keep_days = on_keep_days
        self._streams: list[cr.CaptureStream] = list(streams())
        self._listings: dict[cr.CaptureFolder, cr.Listing | None] = {}
        self._rows: list[cr.CaptureFile] = []
        # a deletion waiting for its second press: which button, and the very files it named
        self._pending: tuple | None = None
        self._deleting = False
        self._keep_asked: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="sr-dialog"):
            yield Static("Recordings on the capture hosts", id="sr-title")
            yield Static(HELP, classes="sr-muted")
            yield Static("", id="sr-summary", classes="sr-muted")
            yield DataTable(id="sr-table")
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
        elif self._deleting:
            # a Refresh would drop its report, and a second one would pile onto the first
            self._set_log("A deletion is running: its report and a fresh listing follow in a moment.")
        elif button == "btn-sr-refresh":
            self.reload()
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
                f"[red]This deletes {what} from the capture host's disk, for good: a session that still needs its "
                f"part has to be exported first (Sessions → Export Streams). Press {button} again to confirm.[/red]")
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

