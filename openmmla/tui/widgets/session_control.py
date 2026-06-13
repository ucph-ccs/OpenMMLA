from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Label, Select, Static

from openmmla.tui.ssh import is_select_sentinel


class SessionControlPanel(Widget):
    """send START/STOP control signals to launched bases and synchronizers.

    Bases launched from the Launcher initialize and then block until a START
    signal arrives on the redis channel `<session_id>/<service>/control`
    (the role of the old control.sh / `mmla ses-ctl`)."""

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

    def _send_sync(self, command: str, session_id: str, services: list[str]) -> str:
        try:
            from openmmla.utils.client import RedisClientWrapper
        except ImportError as exc:
            return f"[red]redis client unavailable: {exc} — pip install redis[/red]"
        try:
            redis_client = RedisClientWrapper(self._config_path)
        except Exception as exc:
            return f"[red]Redis connection failed: {exc}[/red]"

        notes = []
        if command == "STOP":
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
