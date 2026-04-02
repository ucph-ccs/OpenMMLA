from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from openmmla.tui.screens.config import ConfigPanel
from openmmla.tui.screens.environment import EnvironmentPanel
from openmmla.tui.screens.launcher import LauncherPanel
from openmmla.tui.screens.status import StatusPanel


CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"


class OpenMMLAApp(App):
    TITLE = "OpenMMLA Management Console"
    SUB_TITLE = "Config / Env / Launch / Monitor"

    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Config", "Environment", "Launcher", "Status"):
            with TabPane("Config", id="tab-config"):
                yield ConfigPanel()
            with TabPane("Environment", id="tab-env"):
                yield EnvironmentPanel()
            with TabPane("Launcher", id="tab-launcher"):
                yield LauncherPanel()
            with TabPane("Status", id="tab-status"):
                yield StatusPanel()
        yield Footer()
