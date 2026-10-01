from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Label, Select, Static

from openmmla.tui.ssh import is_select_sentinel


def _store_sections(config_path: str) -> tuple[dict, dict]:
    """the Redis and MongoDB sections of the System Settings store."""
    from openmmla.tui.schema.loader import load_existing_config

    try:
        config = load_existing_config(config_path) or {}
    except Exception:
        config = {}
    redis = config.get("Redis") if isinstance(config.get("Redis"), dict) else {}
    mongo = config.get("MongoDB") if isinstance(config.get("MongoDB"), dict) else {}
    return redis, mongo


def unset_control_endpoints(config_path: str) -> list[str]:
    """the System Settings sections Session Control needs and the store does
    not fill in (no store, no section, an empty or unfilled host or url):
    Redis carries the signals, MongoDB marks a stopped session ended."""
    from openmmla.tui.system_services import section_address_set

    redis, mongo = _store_sections(config_path)
    return [name for name, section in (("Redis", redis), ("MongoDB", mongo))
            if not section_address_set(section, name)]


def _unset_note(config_path: str, name: str) -> str:
    """why Session Control has no address for `name`. A section the store
    lacks may still show a host on its form, taken from the pipeline configs
    (a machine whose System Settings were never saved): the form then looks
    filled in, so say that it is only shown, and what saves it."""
    import os

    from openmmla.tui.system_services import (
        SHARED_SECTIONS, SYSTEM_SERVICE_ADDRESS_FIELDS, load_system_service_values, unset_address_note,
        usable_system_service_value,
    )

    field = SYSTEM_SERVICE_ADDRESS_FIELDS.get(name, "host")
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
        shown = load_system_service_values(root).get(f"{name}.{field}")
    except Exception:
        shown = None
    if usable_system_service_value(shown):
        label = str((SHARED_SECTIONS.get(name) or {}).get("label") or name)
        return (f"System Settings → Connections → {label} shows {shown} from the pipeline configs, "
                f"but it is not saved yet: press Save on that form")
    return unset_address_note(name)


def control_endpoints(config_path: str) -> tuple[str, str]:
    """(where the signals go, a warning or "") from the System Settings store.

    The panel has no host: it publishes to the configured Redis from this
    machine, and every base subscribed to that same Redis hears it."""
    from urllib.parse import urlsplit

    from openmmla.tui.system_services import is_loopback_host

    redis, mongo = _store_sections(config_path)
    unset = unset_control_endpoints(config_path)
    redis_host = str(redis.get("host") or "").strip()
    redis_where = ("(no host yet)" if "Redis" in unset
                   else f"{redis_host}:{redis.get('port') or 6379} (db {redis.get('db') or 0})")
    try:
        parts = urlsplit(str(mongo.get("url") or ""))
        mongo_where = "(no url yet)" if "MongoDB" in unset else f"{parts.hostname or 'localhost'}:{parts.port or 27017}"
    except ValueError:
        mongo_where = str(mongo.get("url") or "unset")
    summary = f"Redis {redis_where}  ·  MongoDB {mongo_where}  (from System Settings)"
    warning = ""
    if unset:
        warning = f"{'; '.join(_unset_note(config_path, name) for name in unset)}."
    elif is_loopback_host(redis_host):
        warning = (
            f"Redis.host is {redis_host}: only bases on this machine hear the signal, because a "
            f"base on another machine reads {redis_host} as itself. For a session that spans "
            f"machines, put this machine's host name into System Settings → Redis."
        )
    return summary, warning


class SessionControlPanel(Widget):
    """send START/STOP control signals to launched bases and synchronizers.

    Bases launched from the Launcher initialize and then block until a START
    signal arrives on the redis channel `<session_id>/<service>/control`
    (the role of the old control.sh / `mmla ses-ctl`). There is no host to
    pick: whichever machines the bases run on, they hear the signal as long
    as they use the Redis this panel publishes to."""

    class RefreshRequested(Message):
        """user clicked the ↻ next to the session Select: the launcher
        re-queries the active session list and calls update_session_choices."""

    class SessionStopped(Message):
        """STOP was sent for a session: the launcher's base cards stop opening
        on it, so the next Start is a new take."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    DEFAULT_CSS = """
    SessionControlPanel {
        height: auto;
        padding: 1 2;
    }
    SessionControlPanel .sc-title {
        text-style: bold;
        margin-bottom: 1;
    }
    SessionControlPanel .sc-help {
        color: $text-muted;
        margin-bottom: 1;
    }
    SessionControlPanel .sc-warning {
        color: $warning;
        margin-bottom: 1;
    }
    SessionControlPanel .sc-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    SessionControlPanel .sc-row Label {
        width: 12;
        padding-top: 1;
    }
    SessionControlPanel .sc-row Select {
        width: 1fr;
    }
    SessionControlPanel .sc-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    SessionControlPanel .sc-services {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    SessionControlPanel .sc-services Checkbox {
        margin-right: 2;
    }
    SessionControlPanel .sc-actions {
        layout: horizontal;
        height: auto;
    }
    SessionControlPanel .sc-actions Button {
        margin-right: 1;
        min-width: 16;
    }
    SessionControlPanel .sc-status {
        margin-top: 1;
        height: auto;
    }
    """

    SERVICES = ("asr", "ips", "vfa")

    def __init__(self, session_choices: list[str], config_path: str) -> None:
        super().__init__()
        self._choices = [choice for choice in session_choices if choice]
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        yield Static("[b]Session Control[/b]", classes="sc-title")
        yield Static(
            "Bases and synchronizers wait for a START signal after launch. "
            "Pick the session they were started with, tick the pipelines, then send the signal. "
            "STOP also marks the session as ended in MongoDB.",
            classes="sc-help",
        )
        summary, warning = control_endpoints(self._config_path)
        yield Static(summary, classes="sc-help")
        if warning:
            yield Static(warning, classes="sc-warning")
        with Horizontal(classes="sc-row"):
            yield Label("Session:")
            if self._choices:
                yield Select(
                    [(choice, choice) for choice in self._choices],
                    value=self._choices[0],
                    id="sc-session",
                )
            else:
                yield Select([], prompt="No sessions found", id="sc-session")
            yield Button("↻", variant="primary", compact=True, id="sc-session-refresh", classes="sc-refresh")
        with Horizontal(classes="sc-services"):
            yield Checkbox("ASR", value=True, id="sc-svc-asr")
            yield Checkbox("IPS", value=True, id="sc-svc-ips")
            yield Checkbox("VFA", value=True, id="sc-svc-vfa")
        with Horizontal(classes="sc-actions"):
            yield Button("Send START", variant="success", id="sc-start")
            yield Button("Send STOP", variant="error", id="sc-stop")
        yield Static("", id="sc-status", classes="sc-status")

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#sc-status", Static).update(text)
        except Exception:
            pass

    def _selected_session(self) -> str:
        try:
            value = self.query_one("#sc-session", Select).value
        except Exception:
            return ""
        if is_select_sentinel(value):
            return ""
        return str(value)

    def _selected_services(self) -> list[str]:
        services = []
        for service in self.SERVICES:
            try:
                if self.query_one(f"#sc-svc-{service}", Checkbox).value:
                    services.append(service)
            except Exception:
                pass
        return services

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sc-start":
            self._dispatch("START")
        elif event.button.id == "sc-stop":
            self._dispatch("STOP")
        elif event.button.id == "sc-session-refresh":
            self._set_status("[yellow]Refreshing session list...[/yellow]")
            self.post_message(self.RefreshRequested())

    def update_session_choices(self, choices: list[str]) -> None:
        """replace the session Select options in place, keeping the current
        selection when it is still in the fresh list."""
        self._choices = [str(c) for c in choices if c]
        try:
            sel = self.query_one("#sc-session", Select)
        except Exception:
            return
        current = sel.value
        sel.set_options((c, c) for c in self._choices)
        if current is not Select.NULL and str(current) in self._choices:
            sel.value = str(current)
        elif self._choices:
            sel.value = self._choices[0]
        self._set_status(f"[green]Session list updated ({len(self._choices)} session(s)).[/green]")

    def _dispatch(self, command: str) -> None:
        session_id = self._selected_session()
        if not session_id:
            self._set_status("[red]Select a session first.[/red]")
            return
        services = self._selected_services()
        if not services:
            self._set_status("[red]Tick at least one pipeline.[/red]")
            return
        self._set_status(f"[yellow]Sending {command} to {', '.join(services)}...[/yellow]")
        self.run_worker(
            self._async_send(command, session_id, services),
            group="session-control",
            exclusive=True,
        )

    async def _async_send(self, command: str, session_id: str, services: list[str]) -> None:
        result = await asyncio.to_thread(self._send_sync, command, session_id, services)
        self._set_status(result)
        if command == "STOP":
            self.post_message(self.SessionStopped(session_id))

    def _send_sync(self, command: str, session_id: str, services: list[str]) -> str:
        unset = unset_control_endpoints(self._config_path)
        if "Redis" in unset:
            # not a connection to <uber-server>, nor to a localhost nobody chose
            return f"[red]{command} not sent: {_unset_note(self._config_path, 'Redis')}.[/red]"
        try:
            from openmmla.utils.client import RedisClientWrapper
        except ImportError as exc:
            return f"[red]redis client unavailable: {exc} — pip install redis[/red]"
        try:
            redis_client = RedisClientWrapper(self._config_path)
        except Exception as exc:
            return f"[red]Redis connection failed: {exc}[/red]"

        notes = []
        if command == "STOP" and "MongoDB" in unset:
            notes.append(f"not marked ended in MongoDB: {_unset_note(self._config_path, 'MongoDB')}")
        elif command == "STOP":
            try:
                from openmmla.utils.client import MongoDBClientWrapper
                MongoDBClientWrapper(self._config_path).end_session(session_id)
                notes.append("session marked ended in MongoDB")
            except Exception as exc:
                notes.append(f"MongoDB end_session failed: {exc}")

        receivers_by_service = {}
        for service in services:
            channel = f"{session_id}/{service}/control"
            try:
                receivers_by_service[service] = int(redis_client.publish(channel, command))
            except Exception as exc:
                return f"[red]publish failed on {channel}: {exc}[/red]"

        # the Stream Server records the session's streams only while it runs
        from rich.markup import escape

        from openmmla.utils.stream_recording import set_session_recording
        recording = set_session_recording(session_id, command == "START", start_path=self._config_path)
        notes.append(f"[yellow]{escape(recording['text'])}[/yellow]" if recording["warnings"]
                     else escape(recording["text"]))

        total = sum(receivers_by_service.values())
        detail = ", ".join(f"{svc}:{n}" for svc, n in receivers_by_service.items())
        extra = f" — {'; '.join(notes)}" if notes else ""
        if total == 0:
            return (
                f"[yellow]{command} published ({detail}) but no listeners received it — "
                f"are the bases running and using session '{session_id}'?[/yellow]{extra}"
            )
        icon = "✅" if command == "START" else "🛑"
        return f"[green]{icon} {command} sent for '{session_id}' (listeners {detail}){extra}[/green]"
