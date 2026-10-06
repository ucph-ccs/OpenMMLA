from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from textual import events, on
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from openmmla.tui.screens.environment import EnvironmentPanel
from openmmla.tui.screens.launcher import ServicePanel
from openmmla.tui.screens.sessions import SessionsPanel
from openmmla.tui.screens.status import StatusPanel


CSS_PATH = Path(__file__).parent / "styles" / "app.tcss"

# copies leave Textual as OSC 52, which macOS Terminal ignores, and its ctrl+v pastes
# only what was copied inside the console; a console run on the Mac itself goes
# through pbcopy/pbpaste instead (over SSH the remote end has no Mac clipboard)
_MAC_CLIPBOARD = sys.platform == "darwin" and not os.environ.get("SSH_CONNECTION")
# pbcopy/pbpaste encode in the locale's charset, which can be ASCII
_PB_ENV = {**os.environ, "LC_ALL": "en_US.UTF-8"}


class OpenMMLAApp(App):
    TITLE = "OpenMMLA Management Console"
    SUB_TITLE = "Environment / Launcher / Sessions / Status"

    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("q", "quit", "Quit"),
        # priority: ahead of the Input/TextArea ctrl+v
        Binding("ctrl+v", "paste_system", show=False, priority=True),
    ]

    def copy_to_clipboard(self, text: str) -> None:
        super().copy_to_clipboard(text)
        if _MAC_CLIPBOARD:
            try:
                subprocess.run(["pbcopy"], input=text.encode("utf-8"), env=_PB_ENV,
                               timeout=2, check=False)
            except (OSError, subprocess.SubprocessError):
                pass

    def action_paste_system(self) -> None:
        if not _MAC_CLIPBOARD or self.focused is None:
            raise SkipAction()
        try:
            result = subprocess.run(["pbpaste"], capture_output=True, env=_PB_ENV,
                                    timeout=2, check=False)
        except (OSError, subprocess.SubprocessError):
            raise SkipAction()
        text = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0 or not text:
            raise SkipAction()
        # the event cmd+v delivers, sent the same way (the app forwards it to the focused
        # widget once): an Input keeps the first line, a read-only TextArea ignores it
        self.post_message(events.Paste(text))

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

    @on(SessionsPanel.SessionEnded)
    def _session_ended(self, event: SessionsPanel.SessionEnded) -> None:
        self.query_one(ServicePanel).session_ended(event.session_id)

    @on(EnvironmentPanel.EnvsChanged)
    def _envs_changed(self, event: EnvironmentPanel.EnvsChanged) -> None:
        # the Launcher's [E] markers come from a cache of each host's envs
        self.query_one(ServicePanel).refresh_env_markers(event.target)
