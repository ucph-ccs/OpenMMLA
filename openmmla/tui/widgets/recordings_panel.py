"""the Recordings tab of the Stream Server card: what MediaMTX holds on its
disk, path by path, and the way to make room.

The Sessions tab exports footage by session; this tab is the server's own
inventory. The segments come from the control API, which works for a docker
and a native run alike on any host, and so does a deletion. The sizes and
the free space come from a shell on the server, when the card has one."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Select, Static

from openmmla.tui import recordings
from openmmla.tui.stream_cuts import bash

# a shell on the server: a command's stdout, or None when it could not run
RunShell = Callable[[str], "str | None"]

# what Delete Older Than offers
AGE_CHOICES: list[tuple[str, float]] = [
    ("1 day", 86400.0), ("3 days", 3 * 86400.0), ("7 days", 7 * 86400.0), ("30 days", 30 * 86400.0),
]


def _stamp(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC") if moment is not None else "-"


class StreamServerRecordingsPanel(Widget):

    DEFAULT_CSS = """
    StreamServerRecordingsPanel {
        height: auto;
        padding: 1 2;
    }
    StreamServerRecordingsPanel .rp-title {
        text-style: bold;
        margin-bottom: 1;
    }
    StreamServerRecordingsPanel .rp-muted {
        color: $text-muted;
    }
    StreamServerRecordingsPanel #rp-summary {
        margin-top: 1;
        text-style: bold;
    }
    StreamServerRecordingsPanel #rp-table {
        height: 14;
        margin-top: 1;
    }
    StreamServerRecordingsPanel .rp-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    StreamServerRecordingsPanel .rp-actions Button {
        min-width: 16;
        margin-right: 1;
    }
    StreamServerRecordingsPanel .rp-actions Select {
        width: 16;
    }
    StreamServerRecordingsPanel #rp-log {
        margin-top: 1;
    }
    """

    def __init__(self, *, host: str, api_port: int, quoted_root: str | None = None,
                 run_shell: RunShell | None = None) -> None:
        """`host` and `api_port` are where the control API answers; `run_shell`
        runs a command on the machine that holds `quoted_root`, the record
        folder quoted for its shell, and is None when there is no such shell."""
        super().__init__()
        self._host = host
        self._api_port = api_port
        self._quoted_root = quoted_root
        self._run_shell = run_shell
        self._recorded: list[recordings.Recorded] = []
        self._sizes: dict[str, int] = {}
        self._free: int | None = None
        self._retention: float | None = None
        self._pending: tuple | None = None  # a deletion waiting for its second press

    def compose(self) -> ComposeResult:
        yield Static("[b]Recordings on the Stream Server[/b]", classes="rp-title")
        yield Static(
            "What MediaMTX holds on its disk, path by path: every stream published to it while server-side "
            "recording is on, in ten-minute segments, whether or not a session ran. A session's footage is "
            "exported under Sessions → Export Streams; this is what there is to export from, and the way "
            "to make room before the retention of the Config tab does. A deletion goes through the server's "
            "API, so it works for a docker and a native run alike.",
            classes="rp-muted",
        )
        yield Static("", id="rp-summary", classes="rp-muted")
        yield DataTable(id="rp-table")
        with Horizontal(classes="rp-actions"):
            yield Button("Refresh", variant="primary", id="btn-rp-refresh")
            yield Button("Delete Path", variant="error", id="btn-rp-delete-path")
            yield Button("Delete Older Than", variant="error", id="btn-rp-delete-older")
            yield Select([(label, seconds) for label, seconds in AGE_CHOICES], value=AGE_CHOICES[1][1],
                         allow_blank=False, id="rp-age")
        yield Static("", id="rp-log", classes="rp-muted")

    def on_mount(self) -> None:
        table = self.query_one("#rp-table", DataTable)
        table.add_columns("Path", "Segments", "First", "Last", "Size")
        table.cursor_type = "row"
        self.reload()

    # ---- asking the server ----

    def reload(self) -> None:
        self._pending = None
        self._set_log(f"Asking {self._host}:{self._api_port}...")
        self.run_worker(self._load(), exclusive=True, group="rp-work")

    async def _load(self) -> None:
        try:
            recorded, kept = await asyncio.to_thread(self._ask_server)
        except recordings.RecordingsError as error:
            self._recorded, self._retention = [], None
            self._sizes, self._free = {}, None
            self._show_inventory()
            self._set_log(
                f"[red]The Stream Server does not answer: {escape(str(error))}[/red]\n"
                "Its host and API port are under System Settings → Stream Server; the API is switched on "
                "in mediamtx.yml (Config tab)."
            )
            return
        self._recorded, self._retention = recorded, kept
        self._sizes, self._free = {}, None
        self._show_inventory()
        self._set_log("" if recorded else "The server holds no recording.")
        if self._run_shell is None or not self._quoted_root or not recorded:
            return
        usage = await asyncio.to_thread(self._ask_disk, [item.path for item in recorded])
        if usage is None:
            self._set_log("The sizes could not be read: the record folder is not where the card expects it, "
                          "or the host could not be asked.")
            return
        self._sizes, self._free = usage
        self._show_inventory()

    def _ask_server(self) -> tuple[list[recordings.Recorded], float]:
        return (recordings.inventory(self._host, self._api_port),
                recordings.retention(self._host, self._api_port))

    def _ask_disk(self, paths: list[str]) -> tuple[dict[str, int], int | None] | None:
        output = self._run_shell(bash(recordings.usage_script(self._quoted_root, paths)))
        return recordings.parse_usage(output) if output is not None else None

    # ---- showing it ----

    def _show_inventory(self) -> None:
        table = self.query_one("#rp-table", DataTable)
        selected = self._selected_path()
        table.clear()
        for item in self._recorded:
            first = item.segments[0] if item.segments else None
            last = item.segments[-1] if item.segments else None
            size = recordings.human_size(self._sizes.get(item.path)) if self._sizes else ""
            table.add_row(item.path, str(len(item.segments)), _stamp(first), _stamp(last), size)
        segments = sum(len(item.segments) for item in self._recorded)
        parts = [f"{self._host}:{self._api_port}", f"kept {recordings.describe_retention(self._retention)}",
                 f"{len(self._recorded)} path(s), {segments} segment(s)"]
        if self._sizes:
            parts[-1] += f", {recordings.human_size(sum(self._sizes.values()))}"
        if self._free is not None:
            parts.append(f"{recordings.human_size(self._free)} free on the server's disk")
        self.query_one("#rp-summary", Static).update(" · ".join(parts))
        paths = [item.path for item in self._recorded]
        if selected in paths:
            table.move_cursor(row=paths.index(selected))

    def _selected_path(self) -> str | None:
        table = self.query_one("#rp-table", DataTable)
        row = table.cursor_row
        if table.row_count == 0 or row is None or not 0 <= row < len(self._recorded):
            return None
        return self._recorded[row].path

    def _set_log(self, text: str) -> None:
        try:
            self.query_one("#rp-log", Static).update(text)
        except Exception:
            pass

    # ---- making room ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-rp-refresh":
            self.reload()
        elif event.button.id == "btn-rp-delete-path":
            path = self._selected_path()
            if path is None:
                self._set_log("Choose a path in the table first.")
                return
            plan = [(path, start) for item in self._recorded if item.path == path for start in item.segments]
            self._confirm_or_run(("path", path), plan, f"every segment of {escape(path)} ({len(plan)})")
        elif event.button.id == "btn-rp-delete-older":
            age = self.query_one("#rp-age", Select).value
            if age is Select.BLANK:
                return
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=float(age))
            plan = recordings.segments_before(self._recorded, cutoff)
            self._confirm_or_run(("older", float(age)), plan,
                                 f"{len(plan)} segment(s) that began before {cutoff:%Y-%m-%d %H:%M} UTC")

    def _confirm_or_run(self, key: tuple, plan: list[tuple[str, datetime]], what: str) -> None:
        if not plan:
            self._pending = None
            self._set_log(f"Nothing to delete: {what}.")
            return
        if self._pending != key:
            self._pending = key
            self._set_log(
                f"[red]This removes {what} from the server's disk; a session whose footage is in there "
                "cannot export it afterwards. Press the button again to confirm.[/red]"
            )
            return
        self._pending = None
        self.run_worker(self._delete(plan), exclusive=True, group="rp-work")

    async def _delete(self, plan: list[tuple[str, datetime]]) -> None:
        failures: list[str] = []

        def work() -> int:
            done = 0
            for path, start in plan:
                try:
                    recordings.delete_segment(self._host, path, start, self._api_port)
                    done += 1
                except recordings.RecordingsError as error:
                    failures.append(str(error))
                    if len(failures) >= 5:
                        break  # the server is refusing, not the odd segment gone meanwhile
                if done % 50 == 0 and done:
                    self.app.call_from_thread(self._set_log, f"Deleted {done} of {len(plan)}...")
            return done

        done = await asyncio.to_thread(work)
        note = f"Deleted {done} of {len(plan)} segment(s)."
        if failures:
            note += " [red]Not deleted: " + "; ".join(escape(text) for text in failures[:3])
            note += ("; ..." if len(failures) > 3 else "") + "[/red]"
        self._set_log(note)
        try:
            self._recorded, self._retention = await asyncio.to_thread(self._ask_server)
        except recordings.RecordingsError:
            return
        self._sizes, self._free = {}, None
        self._show_inventory()
        self._set_log(note)
        if self._run_shell is not None and self._quoted_root and self._recorded:
            usage = await asyncio.to_thread(self._ask_disk, [item.path for item in self._recorded])
            if usage is not None:
                self._sizes, self._free = usage
                self._show_inventory()
