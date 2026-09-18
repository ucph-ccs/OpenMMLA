from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from openmmla.tui.screens.environment import EnvironmentPanel
from openmmla.tui.screens.launcher import ServicePanel
from openmmla.tui.screens.sessions import SessionsPanel
from openmmla.tui.screens.status import StatusPanel


CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"


class OpenMMLAApp(App):
    TITLE = "OpenMMLA Management Console"
    SUB_TITLE = "Environment / Launcher / Sessions / Status"

    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Environment", "Launcher", "Sessions", "Status"):
            with TabPane("Environment", id="tab-env"):
                yield EnvironmentPanel()
            with TabPane("Launcher", id="tab-launcher"):
                yield ServicePanel()
            with TabPane("Sessions", id="tab-sessions"):
                yield SessionsPanel()
            with TabPane("Status", id="tab-status"):
                yield StatusPanel()
        yield Footer()

    @on(SessionsPanel.SessionDeleted)
    def _forget_deleted_session(self, event: SessionsPanel.SessionDeleted) -> None:
        # the two tabs are siblings: the news has to be carried across
        self.query_one(ServicePanel).forget_session(event.session_id)

    @on(EnvironmentPanel.EnvsChanged)
    def _envs_changed(self, event: EnvironmentPanel.EnvsChanged) -> None:
        # the Launcher's [E] markers come from a cache of each host's envs
        self.query_one(ServicePanel).refresh_env_markers(event.target)
