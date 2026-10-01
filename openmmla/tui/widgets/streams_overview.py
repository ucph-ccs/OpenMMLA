"""the Streams tab of the Stream Server card: every stream of every pipeline in
one table, whichever card started it, and the way to stop them.

A stream is started on its pipeline card (IPS, VFA or ASR Base, Streams tab)
and publishes until someone stops it there, so a camera left running is easy
to forget. This tab lists them side by side: the Streams entries of the three
cards, each read from the config of the host its card is on; the captures the
stream registry says this console started that no entry names any more; any
mmla-stream-* session a machine still has that neither knows of (started from
another console); and whatever the server receives that none of those
accounts for, with the address it comes from. Each machine is asked once (all
of them at the same time), the server through its API."""

from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, DataTable, Static

from openmmla.tui import recordings
from openmmla.tui.schema.loader import StreamDef
from openmmla.tui.ssh import TARGET_PLATFORMS, TARGET_STATES, get_profile_by_name, load_ssh_profiles
from openmmla.tui.stream_cuts import bash
from openmmla.tui.system_services import stream_server_path
from openmmla.tui.widgets.stream_panel import (
    STREAM_EXITED, STREAM_RUNNING, STREAM_STARTING, STREAM_STOP_GRACE_SECONDS, STREAM_STOPPED,
    STREAM_UNKNOWN, StreamPanel, _build_stop_stream_cmd, _parse_stream_state, _run_on_host, _stream_app,
    _stream_state_cmd, _tmux_pane_target, _tmux_session_name, _with_stream_path,
)
from openmmla.utils.stream_registry import load_stream_registry, mark_stream_stopped

# the Streams entries of the pipeline cards, (card, stream), and what could not be read
Configured = Callable[[], "tuple[list[tuple[str, StreamDef]], list[str]]"]

# what a row's capture is besides the states its machine reports (stream_panel.STREAM_*)
EXTERNAL = "external"      # no machine: someone else publishes it
OFFLINE = "offline"        # the console's host check says its machine is offline: not asked
NO_PROFILE = "no profile"  # its SSH profile is gone
ASKING = "asking"          # its machine is being asked

# a capture that still holds something on its machine: Stop All stops these
_STOPPABLE = (STREAM_RUNNING, STREAM_STARTING, STREAM_EXITED)

SESSION_PREFIX = _tmux_session_name("")

# seconds a machine has to answer; ssh itself gives up connecting after 5
SCAN_TIMEOUT = 15.0

# the order the cards stand in
_CARD_ORDER = ("IPS Base", "VFA Base", "ASR Base")


@dataclass
class StreamRow:
    """one line of the table: a capture on a machine, a stream someone else
    publishes, or a path the server receives that nothing here accounts for."""
    name: str
    machine: str = ""                 # "local" or an SSH profile; "" when the console does not run it
    cards: tuple[str, ...] = ()       # the pipeline cards whose Streams name it
    app: str = ""                     # ips, vfa, asr: the first part of its path
    target: str = ""
    path: str | None = None           # its path on the Stream Server
    state: str = ASKING
    noted: bool = False               # the stream registry says this console started it
    live: recordings.Published | None = None
    publisher: str = ""               # where an unclaimed path comes from

    @property
    def key(self) -> tuple[str, str, str]:
        return self.machine, self.name, self.path or ""

    @property
    def in_config(self) -> bool:
        return bool(self.cards)


@dataclass
class HostScan:
    """what a machine said: whether it answered, the stream sessions it has
    (by stream name), and the state of each stream it was asked about."""
    answered: bool
    sessions: set[str]
    states: dict[str, str]


def _natural(name: str):
    """cam-2 before cam-10."""
    parts = re.split(r"(\d+)", name)
    return [int(part) if index % 2 else part.lower() for index, part in enumerate(parts)], name


def _card_label(card: str) -> str:
    """IPS Base -> IPS."""
    return card.split()[0] if card.split() else card


def capture_rows(configured: list[tuple[str, StreamDef]], registry: dict, server: dict) -> list[StreamRow]:
    """the rows the console knows of before it asks anyone: every Streams entry
    of the cards, then each capture the registry notes as running that no entry
    names on that machine. A stream two cards name on the same machine is one
    capture, as its tmux session is named after the stream alone."""
    rows: dict[tuple[str, str], StreamRow] = {}
    for card, stream in configured:
        machine = (stream.ssh_profile or "").strip()
        path = stream_server_path(stream.target, server) or stream_server_path(stream.read_url, server)
        key = (machine, stream.name) if machine else ("", f"{card}\0{stream.name}")
        if key in rows:
            if card not in rows[key].cards:
                rows[key].cards += (card,)
            continue
        rows[key] = StreamRow(
            stream.name, machine, (card,), _stream_app(stream.read_url, server), stream.target, path,
            ASKING if machine else EXTERNAL)
    for name, entry in (registry or {}).items():
        if not isinstance(entry, dict) or entry.get("status") != "running":
            continue
        machine = str(entry.get("ssh_profile") or "").strip()
        if not machine:
            continue
        key = (machine, str(name))
        if key in rows:
            rows[key].noted = True
            continue
        target = str(entry.get("target") or "")
        read = str(entry.get("read_target") or "") or target
        rows[key] = StreamRow(
            str(name), machine, (), _stream_app(read, server), target,
            stream_server_path(target, server) or stream_server_path(read, server), ASKING, noted=True)
    return list(rows.values())


def scan_command(names, discover: bool = True) -> str:
    """what a machine is asked: the stream sessions it has (its tmux sessions,
    and the noted ffmpeg of a Mac's Terminal window, which runs on without
    one), and the state of each of `names`. Ends with SCANNED, so an answer
    tells itself from a shell that never ran it."""
    parts = []
    if discover:
        parts += [
            f"tmux ls -F '#{{session_name}}' 2>/dev/null | sed -n 's/^{SESSION_PREFIX}/SESSION /p'",
            f'for f in "$HOME"/.openmmla/streams/{SESSION_PREFIX}*.pid; do [ -e "$f" ] || continue; '
            f'n="${{f##*/}}"; n="${{n%.pid}}"; echo "SESSION ${{n#{SESSION_PREFIX}}}"; done',
        ]
    for name in sorted(set(names)):
        parts.append(f"printf 'STATE %s ' {shlex.quote(name)}; {_stream_state_cmd(_tmux_session_name(name))}")
    parts.append("echo SCANNED")
    return bash(_with_stream_path("; ".join(parts)))


def parse_scan(output: str | None) -> HostScan:
    sessions: set[str] = set()
    states: dict[str, str] = {}
    lines = (output or "").splitlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith("SESSION "):
            name = line[len("SESSION "):].strip()
            if name:
                sessions.add(name)
        elif line.startswith("STATE "):
            name, _, word = line[len("STATE "):].rpartition(" ")
            state = _parse_stream_state(word)
            if name and state:
                states[name] = state
    return HostScan("SCANNED" in (line.strip() for line in lines), sessions, states)


def _ask(machine: str, names, discover: bool) -> HostScan:
    """one round trip to a machine ("local": this one)."""
    profile = None
    if machine != "local":
        profile = get_profile_by_name(machine)
        if profile is None:
            return HostScan(False, set(), {})
    try:
        result = _run_on_host(profile, scan_command(names, discover), SCAN_TIMEOUT)
    except Exception:
        # never shown: an ssh command line carries the profile's password
        return HostScan(False, set(), {})
    return parse_scan(result.stdout)


def scan_machine(machine: str, names) -> HostScan | str:
    """the stream sessions of a machine and the state of each, the ones of
    `names` included: a HostScan, or OFFLINE / NO_PROFILE when it was not asked.
    A session found there that `names` does not hold is asked about too."""
    if machine != "local":
        if get_profile_by_name(machine) is None:
            return NO_PROFILE
        if TARGET_STATES.get(machine) == "offline":
            return OFFLINE
    scan = _ask(machine, names, discover=True)
    unknown = scan.sessions - set(names)
    if scan.answered and unknown:
        more = _ask(machine, unknown, discover=False)
        scan.states.update(more.states)
    return scan


def machines_to_scan(rows: list[StreamRow]) -> dict[str, set[str]]:
    """machine -> the streams to ask it about: those of the rows, and every
    other machine the console can reach (no names: it is asked what it has),
    which is where a capture started from another console would be."""
    wanted: dict[str, set[str]] = {"local": set()}
    for profile in load_ssh_profiles():
        if TARGET_PLATFORMS.get(profile.name) != "windows":
            wanted.setdefault(profile.name, set())
    for row in rows:
        if row.machine:
            wanted.setdefault(row.machine, set()).add(row.name)
    return wanted


def apply_scans(rows: list[StreamRow], scans: dict[str, HostScan | str], server: dict) -> list[StreamRow]:
    """the rows with what their machines said, and a row for each stream
    session a machine has that no row named."""
    named = {(row.machine, row.name) for row in rows if row.machine}
    for row in rows:
        if not row.machine:
            continue
        scan = scans.get(row.machine)
        if isinstance(scan, str):
            row.state = scan
        elif scan is None or not scan.answered:
            row.state = STREAM_UNKNOWN
        else:
            row.state = scan.states.get(row.name, STREAM_UNKNOWN)
    found = []
    for machine, scan in scans.items():
        if isinstance(scan, str) or not scan.answered:
            continue
        for name in sorted(scan.sessions):
            if (machine, name) in named:
                continue
            state = scan.states.get(name, STREAM_UNKNOWN)
            if state == STREAM_STOPPED:
                continue  # a noted pid of a run long gone
            found.append(StreamRow(name, machine, state=state))
    return rows + found


def attach_live(rows: list[StreamRow], live: list[recordings.Published]) -> list[StreamRow]:
    """what the server receives, on the rows of the streams it is: by path, and
    for a capture found on a machine that nothing names (so no path is known
    for it) the one path that ends in its name, as + Add Stream names them.
    Every other path is a row of its own: something else publishes it."""
    by_path = {item.path: item for item in live}
    claimed: set[str] = set()
    for row in rows:
        if row.path and row.path in by_path:
            row.live = by_path[row.path]
            claimed.add(row.path)
    for row in rows:
        if row.path is None and row.machine and not row.target:
            matches = [path for path in by_path if path not in claimed and path.rsplit("/", 1)[-1] == row.name]
            if len(matches) == 1:
                row.path, row.live = matches[0], by_path[matches[0]]
                row.app = matches[0].split("/", 1)[0] if "/" in matches[0] else ""
                claimed.add(matches[0])
    extra = [
        StreamRow(path.rsplit("/", 1)[-1], app=path.split("/", 1)[0] if "/" in path else "", path=path,
                  state=EXTERNAL, live=item)
        for path, item in sorted(by_path.items()) if path not in claimed
    ]
    return rows + extra


def is_active(row: StreamRow) -> bool:
    """something of it runs: its capture, or the server receives it."""
    return row.state in _STOPPABLE or row.live is not None


def sort_rows(rows: list[StreamRow]) -> list[StreamRow]:
    """what runs first; then by card, captures nothing names last, and by name."""
    def key(row: StreamRow):
        cards = [_CARD_ORDER.index(card) for card in row.cards if card in _CARD_ORDER]
        return (not is_active(row), min(cards) if cards else len(_CARD_ORDER), _natural(row.name), row.machine)
    return sorted(rows, key=key)


def live_for(ready: datetime | None, now: datetime | None = None) -> str:
    """how long a path has been live: 12m, 3h 05m, 2d 4h."""
    if ready is None:
        return ""
    seconds = max(((now or datetime.now(timezone.utc)) - ready).total_seconds(), 0)
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


_STATE_CELLS = {
    STREAM_RUNNING: ("Running", "green"),
    STREAM_STARTING: ("Starting", "green"),
    STREAM_EXITED: ("Exited", "yellow"),
    STREAM_STOPPED: ("Stopped", "dim"),
    STREAM_UNKNOWN: ("No answer", "yellow"),
    OFFLINE: ("Offline", "dim"),
    NO_PROFILE: ("No profile", "red"),
    EXTERNAL: ("External", "dim"),
    ASKING: ("…", "dim"),
}


class StreamServerStreamsPanel(Widget):

    DEFAULT_CSS = """
    StreamServerStreamsPanel {
        height: auto;
        padding: 1 2;
    }
    StreamServerStreamsPanel .sp-title {
        text-style: bold;
        margin-bottom: 1;
    }
    StreamServerStreamsPanel .sp-muted {
        color: $text-muted;
    }
    StreamServerStreamsPanel #sp-summary {
        text-style: bold;
    }
    StreamServerStreamsPanel #sp-table {
        height: 16;
        margin-top: 1;
    }
    StreamServerStreamsPanel .sp-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }
    StreamServerStreamsPanel .sp-actions Button {
        min-width: 14;
        margin-right: 1;
    }
    StreamServerStreamsPanel #sp-log {
        margin-top: 1;
    }
    """

    def __init__(self, *, host: str, api_port: int, configured: Configured,
                 server: Callable[[], dict], project_dir: str) -> None:
        """`host` and `api_port` are where the control API answers; `configured`
        gives the Streams entries of the pipeline cards (it may ask other hosts,
        so it runs off the UI thread); `server` is System Settings → Stream
        Server, which tells a stream URL's path on it."""
        super().__init__()
        self._host = host
        self._api_port = api_port
        self._configured = configured
        self._server = server
        self._project_dir = project_dir
        self._rows: list[StreamRow] = []
        self._server_answered = True
        self._loaded = False
        self._busy = False  # a Stop is under way: it is not cut short by a Refresh
        self._pending: tuple | None = None  # Stop All waiting for its second press

    def compose(self) -> ComposeResult:
        yield Static("[b]Streams of every pipeline[/b]", classes="sp-title")
        yield Static("", id="sp-summary", classes="sp-muted")
        yield DataTable(id="sp-table")
        with Horizontal(classes="sp-actions"):
            yield Button("Refresh", variant="primary", id="btn-sp-refresh")
            yield Button("Stop", variant="error", id="btn-sp-stop")
            yield Button("Stop All", variant="error", id="btn-sp-stop-all")
            yield Button("Logs", variant="primary", id="btn-sp-logs")
        yield Static("", id="sp-log", classes="sp-muted")

    def on_mount(self) -> None:
        table = self.query_one("#sp-table", DataTable)
        table.add_columns("Stream", "Pipeline", "Machine", "Capture", "Stream Server", "Readers", "Path")
        table.cursor_type = "row"

    def on_show(self) -> None:
        # the card mounts every tab at once: its machines are asked when this one is first opened
        if not self._loaded:
            self._loaded = True
            self.reload()

    # ---- asking ----

    def reload(self, note: str = "") -> None:
        """ask everyone again; `note` (what a Stop did) stays above what the asking found."""
        if self._busy:
            self._set_log("Still stopping; the table is asked again when that is done.")
            return
        self._pending = None
        self.run_worker(self._load(note), exclusive=True, group="sp-work", exit_on_error=False)

    def _server_address(self) -> dict:
        try:
            return dict(self._server() or {})
        except Exception:
            return {}

    async def _load(self, note: str = "") -> None:
        try:
            await self._ask_everyone(note)
        except Exception as error:
            # its type only: an ssh error's text may quote the command line, password and all
            self._set_log(f"[red]The table could not be filled ({type(error).__name__}). Refresh asks again.[/red]")

    async def _ask_everyone(self, note: str) -> None:
        self._set_log("\n".join(filter(None, [note, "Reading the pipeline configs..."])))
        try:
            configured, notes = await asyncio.to_thread(self._configured)
        except Exception as error:
            configured, notes = [], [f"The pipeline configs could not be read ({type(error).__name__})."]
        server = self._server_address()
        try:
            registry = load_stream_registry(self._project_dir).get("streams", {})
        except Exception:
            registry = {}
        rows = capture_rows(configured, registry, server)
        self._rows = rows
        self._show()
        wanted = await asyncio.to_thread(machines_to_scan, rows)
        self._set_log("\n".join(filter(None, [
            note, f"Asking {len(wanted)} machine(s) and {self._host}:{self._api_port}..."])))
        scans, live = await asyncio.gather(
            asyncio.to_thread(self._scan_all, wanted), asyncio.to_thread(self._ask_server))
        rows = apply_scans(rows, scans, server)
        self._forget_stopped(rows)
        # a capture nothing names any more is listed while its machine still holds it
        rows = [row for row in rows if row.in_config or row.state != STREAM_STOPPED]
        self._server_answered = live is not None
        rows = attach_live(rows, live or [])
        unclaimed = {row.live.source_type for row in rows if row.state == EXTERNAL and row.live is not None}
        if unclaimed:
            try:
                addresses = await asyncio.to_thread(
                    recordings.publisher_addresses, self._host, self._api_port, 3.0, unclaimed)
            except Exception:
                addresses = {}
            for row in rows:
                if row.state == EXTERNAL and row.live is not None:
                    row.publisher = addresses.get(row.live.source_id, "")
        self._rows = sort_rows(rows)
        self._show()
        self._set_log("\n".join(filter(None, [note, self._diagnosis(notes)])))

    @staticmethod
    def _scan_all(wanted: dict[str, set[str]]) -> dict[str, HostScan | str]:
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(wanted)))) as pool:
            futures = {machine: pool.submit(scan_machine, machine, names) for machine, names in wanted.items()}
            return {machine: future.result() for machine, future in futures.items()}

    def _ask_server(self) -> list[recordings.Published] | None:
        try:
            return recordings.published(self._host, self._api_port)
        except recordings.RecordingsError:
            return None

    def _forget_stopped(self, rows: list[StreamRow]) -> None:
        """the registry is told of a capture it notes as running that its
        machine has nothing left of, as the pipeline card's Refresh does; the
        entry is the one started on that machine (the same name may run on
        another)."""
        try:
            entries = load_stream_registry(self._project_dir).get("streams", {})
        except Exception:
            return
        for row in rows:
            if row.noted and row.state == STREAM_STOPPED:
                self._mark_stopped(row, entries)

    def _mark_stopped(self, row: StreamRow, entries: dict | None = None) -> str:
        """note in the registry that a capture stopped; the recording its Start noted, if any."""
        if entries is None:
            try:
                entries = load_stream_registry(self._project_dir).get("streams", {})
            except Exception:
                return ""
        entry = entries.get(row.name)
        if not isinstance(entry, dict) or str(entry.get("ssh_profile") or "").strip() != row.machine:
            return ""
        recorded = str(entry.get("record_path") or "").strip() if entry.get("status") == "running" else ""
        try:
            mark_stream_stopped(row.name, project_dir=self._project_dir)
        except Exception:
            pass
        return recorded

    # ---- showing ----

    def _show(self) -> None:
        try:
            table = self.query_one("#sp-table", DataTable)
        except Exception:
            return
        selected = self._selected()
        selected_key = selected.key if selected is not None else None
        table.clear()
        for row in self._rows:
            table.add_row(*self._cells(row))
        keys = [row.key for row in self._rows]
        if selected_key in keys:
            table.move_cursor(row=keys.index(selected_key))
        self.query_one("#sp-summary", Static).update(self._summary())

    def _cells(self, row: StreamRow) -> list:
        name: Text | str = row.name
        if row.machine and not row.in_config:
            name = Text.assemble(row.name, ("  (in no Streams)", "yellow"))
        elif row.state == EXTERNAL and row.live is not None and not row.in_config:
            name = Text.assemble(row.name, ("  (not from here)", "yellow"))
        cards = ", ".join(_card_label(card) for card in row.cards) or (row.app.upper() if row.app else "-")
        machine = row.machine or row.publisher or "-"
        state, style = _STATE_CELLS.get(row.state, (row.state, ""))
        if row.path is None:
            server = Text("-", "dim")
        elif row.live is not None:
            since = live_for(row.live.ready)
            server = Text(f"● live {since}".rstrip(), "green")
        elif not self._server_answered:
            server = Text("no answer", "dim")
        else:
            server = Text("○ not live", "dim")
        readers = str(row.live.readers) if row.live is not None else ""
        return [name, cards, machine, Text(state, style), server, readers, row.path or row.target or "-"]

    def _summary(self) -> str:
        running = [row for row in self._rows if row.state in (STREAM_RUNNING, STREAM_STARTING)]
        exited = [row for row in self._rows if row.state == STREAM_EXITED]
        live = [row for row in self._rows if row.live is not None]
        foreign = [row for row in self._rows if row.state == EXTERNAL and row.live is not None and not row.in_config]
        machines = {row.machine for row in running}
        parts = [f"{self._host}:{self._api_port}",
                 f"{len(running)} capture(s) running on {len(machines)} machine(s)"]
        if exited:
            parts.append(f"{len(exited)} exited")
        offline = [row for row in self._rows if row.state == OFFLINE]
        if offline:
            parts.append(f"{len(offline)} on offline machine(s)")
        parts.append(f"{len(live)} path(s) live on the server" if self._server_answered
                     else "the server does not answer")
        if foreign:
            parts.append(f"{len(foreign)} from elsewhere")
        return " · ".join(parts)

    def _diagnosis(self, notes: list[str]) -> str:
        lines = [f"[yellow]{escape(note)}[/yellow]" for note in notes]
        if not self._server_answered:
            lines.append(
                f"[red]The Stream Server does not answer at {escape(self._host)}:{self._api_port}, so what it "
                "receives is not known.[/red] Its host and API port are under System Settings → Stream Server.")
        unanswered = sorted({row.machine for row in self._rows if row.state == STREAM_UNKNOWN})
        if unanswered:
            lines.append(f"No answer from {escape(', '.join(unanswered))}: what runs there is not known.")
        return "\n".join(lines)

    def _selected(self) -> StreamRow | None:
        try:
            table = self.query_one("#sp-table", DataTable)
        except Exception:
            return None
        row = table.cursor_row
        if table.row_count == 0 or row is None or not 0 <= row < len(self._rows):
            return None
        return self._rows[row]

    def _set_log(self, text: str) -> None:
        try:
            self.query_one("#sp-log", Static).update(text)
        except Exception:
            pass

    def _log(self, text: str) -> None:
        """into the console's log below the card, where the pipeline cards' Streams tabs write too."""
        self.post_message(StreamPanel.StreamLog(text))

    # ---- stopping ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id or ""
        if button == "btn-sp-refresh":
            self.reload()
        elif button == "btn-sp-stop":
            self._pending = None
            row = self._selected()
            if row is None:
                self._set_log("Choose a stream in the table first.")
            else:
                self._stop(row)
        elif button == "btn-sp-stop-all":
            self._stop_all()
        elif button == "btn-sp-logs":
            self._pending = None
            row = self._selected()
            if row is None:
                self._set_log("Choose a stream in the table first.")
            elif not row.machine:
                self._set_log(f"{escape(row.name)} is not run from here: it has no capture to read the logs of.")
            else:
                self.run_worker(self._show_logs(row), group="sp-logs", exit_on_error=False)

    def _stop(self, row: StreamRow) -> None:
        if self._busy:
            self._set_log("Still stopping.")
            return
        if row.machine and row.state not in (STREAM_STOPPED, OFFLINE, NO_PROFILE):
            self._busy = True
            self.run_worker(self._stop_captures([row]), group="sp-stop", exit_on_error=False)
        elif row.live is not None:
            self._busy = True
            self.run_worker(self._disconnect(row), group="sp-stop", exit_on_error=False)
        elif row.state == OFFLINE:
            self._set_log(f"{escape(row.machine)} is offline: nothing can be stopped there now.")
        elif row.state == NO_PROFILE:
            self._set_log(f"There is no SSH profile {escape(row.machine)} any more to reach {escape(row.name)} with.")
        else:
            self._set_log(f"{escape(row.name)} is not running: there is nothing to stop.")

    def _stop_all(self) -> None:
        if self._busy:
            self._set_log("Still stopping.")
            return
        rows = [row for row in self._rows if row.machine and row.state in _STOPPABLE]
        if not rows:
            self._pending = None
            self._set_log("No capture is running.")
            return
        key = ("all", frozenset(row.key for row in rows))
        if self._pending != key:
            self._pending = key
            machines = len({row.machine for row in rows})
            readers = sum(row.live.readers for row in rows if row.live is not None)
            others = sum(1 for row in self._rows if row.state == EXTERNAL and row.live is not None)
            text = (f"[red]This stops {len(rows)} capture(s) on {machines} machine(s)"
                    + (f"; {readers} reader(s) pull them now and lose their input" if readers else "")
                    + ". Press Stop All again to confirm.[/red]")
            if others:
                text += f"\n{others} stream(s) published from elsewhere stay: Stop disconnects one at a time."
            self._set_log(text)
            return
        self._pending = None
        self._busy = True
        self.run_worker(self._stop_captures(rows), group="sp-stop", exit_on_error=False)

    async def _stop_captures(self, rows: list[StreamRow]) -> None:
        try:
            names = ", ".join(row.name for row in rows[:6]) + (" ..." if len(rows) > 6 else "")
            self._set_log(f"Stopping {escape(names)}...")
            outcomes = await asyncio.gather(*(asyncio.to_thread(self._stop_capture, row) for row in rows))
            stopped = 0
            for row, outcome in zip(rows, outcomes):
                where = "this machine" if row.machine == "local" else row.machine
                if outcome == "stopped":
                    stopped += 1
                    recorded = self._mark_stopped(row)
                    row.state = STREAM_STOPPED
                    self._log(f"[red]{escape(row.name)} stopped on {escape(where)}.[/red]")
                    if recorded:
                        self._log(f"  Recording kept at {escape(recorded)} on {escape(where)}.")
                elif outcome == "no profile":
                    self._log(f"[red]{escape(row.name)}: SSH profile '{escape(row.machine)}' not found.[/red]")
                elif outcome == "timeout":
                    self._log(f"[yellow]{escape(row.name)}: {escape(where)} did not answer in time; "
                              f"it may still be running.[/yellow]")
                else:
                    self._log(f"[yellow]{escape(row.name)} may still be running on {escape(where)}.[/yellow]")
            self._show()
            note = f"Stopped {stopped} of {len(rows)}" + ("." if stopped == len(rows) else ": the log below says why.")
            self._set_log(note)
        finally:
            self._busy = False
        # the server lets a path go a moment after its publisher left
        await asyncio.sleep(1.0)
        self.reload(note)

    @staticmethod
    def _stop_capture(row: StreamRow) -> str:
        """Ctrl-C a capture's ffmpeg on its machine and close its session, as the
        card's Stop does (it waits for ffmpeg to finish its recording)."""
        profile = None
        if row.machine != "local":
            profile = get_profile_by_name(row.machine)
            if profile is None:
                return "no profile"
        command = _with_stream_path(_build_stop_stream_cmd(_tmux_session_name(row.name)))
        try:
            result = _run_on_host(profile, command, STREAM_STOP_GRACE_SECONDS + 7.0)
        except subprocess.TimeoutExpired:
            return "timeout"  # its text is the ssh command line, which carries the password
        except Exception:
            return "error"
        return "stopped" if "DONE" in (result.stdout or "") else "unsure"

    async def _disconnect(self, row: StreamRow) -> None:
        """a path the server receives from something the console does not run:
        the server closes that connection."""
        try:
            source = row.live
            sender = row.publisher or (f"{row.machine}?" if row.machine else "its publisher")
            try:
                await asyncio.to_thread(
                    recordings.kick_publisher, self._host, source.source_type, source.source_id, self._api_port)
            except recordings.RecordingsError as error:
                self._set_log(f"[red]The server did not disconnect {escape(row.path or row.name)}: "
                              f"{escape(str(error))}[/red]")
                return
            self._log(f"[red]{escape(row.path or row.name)}: the server disconnected {escape(sender)}.[/red]")
            note = (f"Disconnected {escape(sender)} from {escape(row.path or row.name)}. Whatever publishes it is "
                    "not run from here, so it may connect again; stop it where it runs.")
            self._set_log(note)
        finally:
            self._busy = False
        await asyncio.sleep(1.0)
        self.reload(note)

    async def _show_logs(self, row: StreamRow) -> None:
        """the last lines of a capture's pane, into the console's log."""
        profile = None
        if row.machine != "local":
            profile = get_profile_by_name(row.machine)
            if profile is None:
                self._set_log(f"There is no SSH profile {escape(row.machine)} any more.")
                return
        session = _tmux_session_name(row.name)
        command = _with_stream_path(f"tmux capture-pane -t {_tmux_pane_target(session)} -p -S -120 2>/dev/null")
        try:
            result = await asyncio.to_thread(_run_on_host, profile, command, 10.0)
        except Exception:
            self._set_log(f"{escape(row.machine)} did not answer.")
            return
        output = (result.stdout or "").strip()
        if result.returncode != 0:
            self._set_log(f"{escape(row.name)} has no session on {escape(row.machine)} to read.")
            return
        if not output:
            self._set_log(f"Nothing in the pane of {escape(row.name)} on {escape(row.machine)} yet.")
            return
        self._log(f"[cyan]── Logs for {escape(row.name)} on {escape(row.machine)} ──[/cyan]")
        for line in output.splitlines()[-80:]:
            self._log(escape(line))
        self._set_log(f"The last lines of {escape(row.name)} are in the log below.")
