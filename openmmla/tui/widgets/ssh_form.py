from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label

from openmmla.tui.ssh import (
    SSHProfile,
    load_ssh_profiles,
    save_ssh_profiles,
    ssh_test_connection,
)


def _profile_row_text(profile: SSHProfile) -> str:
    auth = "password" if profile.password else ("key" if profile.key_path else "default")
    # the bracket is escaped: "[password]" on its own reads as a markup tag
    # and was never displayed
    return f"{profile.name}  ({profile.user}@{profile.host}:{profile.port})  \\[{auth}]"


def _profile_row_buttons(profile_name: str) -> tuple[Button, Button, Button]:
    """the Test / Edit / Delete buttons of one profile row. The profile rides in
    `name`: a widget id cannot hold the dots of a name like "pi.local", and one
    such profile used to take the whole form down."""
    return (
        Button("Test", variant="warning", name=profile_name, classes="ssh-row-test"),
        Button("Edit", name=profile_name, classes="ssh-row-edit"),
        Button("Delete", variant="error", name=profile_name, classes="ssh-row-del"),
    )

class SSHForm(Widget):
    """form widget for creating / editing / deleting SSH profiles."""

    class ProfilesChanged(Message):
        """posted when the profile list is modified; `renamed` is (old, new)
        when a profile changed its name, so what refers to it can follow."""

        def __init__(self, renamed: tuple[str, str] | None = None) -> None:
            super().__init__()
            self.renamed = renamed

    class ConnectionTested(Message):
        """posted after a successful Test Connection (lets the launcher re-probe targets)."""

    DEFAULT_CSS = """
    SSHForm {
        height: 1fr;
        padding: 1 2;
    }
    SSHForm .ssh-root {
        height: 1fr;
    }
    SSHForm .ssh-title {
        text-style: bold;
        margin-bottom: 1;
    }
    SSHForm .ssh-field {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    SSHForm .ssh-field-label {
        width: 22;
        padding-top: 1;
    }
    SSHForm .ssh-field-input {
        width: 1fr;
    }
    SSHForm .ssh-actions {
        height: auto;
        margin-top: 1;
    }
    SSHForm .ssh-actions Button {
        margin: 0 1;
        min-width: 18;
    }
    SSHForm .ssh-status {
        margin-top: 1;
        height: auto;
    }
    SSHForm .ssh-profile-list {
        margin-top: 1;
        height: 1fr;
        min-height: 5;
        border-bottom: solid $surface-lighten-1;
        padding-bottom: 1;
        scrollbar-size: 1 1;
    }
    SSHForm .ssh-form-section {
        height: auto;
        padding-top: 1;
    }
    SSHForm .profile-entry {
        layout: horizontal;
        height: auto;
    }
    SSHForm .profile-entry-name {
        width: 1fr;
        padding-top: 1;
    }
    SSHForm .profile-entry Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._profiles = load_ssh_profiles()
        self._editing: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="ssh-root"):
            yield Static("[b]SSH Profiles[/b]", classes="ssh-title")

            with VerticalScroll(classes="ssh-profile-list", id="ssh-profile-list"):
                for p in self._profiles:
                    with Horizontal(classes="profile-entry"):
                        yield Static(_profile_row_text(p), classes="profile-entry-name")
                        yield from _profile_row_buttons(p.name)

            with Vertical(classes="ssh-form-section"):
                yield Static("[b]Add / Edit Profile[/b]", classes="ssh-title")

                with Horizontal(classes="ssh-actions"):
                    yield Button("Add Profile", variant="success", id="ssh-save")
                    yield Button("Test Connection", variant="warning", id="ssh-test")
                    yield Button("Clear Form", variant="default", id="ssh-clear")

                yield Static("", id="ssh-status", classes="ssh-status")

                with Horizontal(classes="ssh-field"):
                    yield Label("Profile Name:", classes="ssh-field-label")
                    yield Input(placeholder="e.g. lab-server", id="ssh-name", classes="ssh-field-input")
                with Horizontal(classes="ssh-field"):
                    yield Label("Host:", classes="ssh-field-label")
                    yield Input(placeholder="e.g. 192.168.1.100", id="ssh-host", classes="ssh-field-input")
                with Horizontal(classes="ssh-field"):
                    yield Label("User:", classes="ssh-field-label")
                    yield Input(placeholder="e.g. ubuntu", id="ssh-user", classes="ssh-field-input")
                with Horizontal(classes="ssh-field"):
                    yield Label("Port:", classes="ssh-field-label")
                    yield Input(value="22", id="ssh-port", classes="ssh-field-input")
                with Horizontal(classes="ssh-field"):
                    yield Label("Password:", classes="ssh-field-label")
                    yield Input(
                        placeholder="leave empty for key-based auth",
                        id="ssh-password",
                        password=True,
                        classes="ssh-field-input",
                    )
                with Horizontal(classes="ssh-field"):
                    yield Label("Key Path:", classes="ssh-field-label")
                    yield Input(placeholder="e.g. ~/.ssh/id_rsa (leave empty for default)", id="ssh-key", classes="ssh-field-input")
                with Horizontal(classes="ssh-field"):
                    yield Label("Remote Project Path:", classes="ssh-field-label")
                    yield Input(value="~/OpenMMLA", id="ssh-remote-path", classes="ssh-field-input")

    def _get_form_values(self) -> dict:
        return {
            "name": self.query_one("#ssh-name", Input).value.strip(),
            "host": self.query_one("#ssh-host", Input).value.strip(),
            "user": self.query_one("#ssh-user", Input).value.strip(),
            "port": self.query_one("#ssh-port", Input).value.strip(),
            "password": self.query_one("#ssh-password", Input).value,
            "key_path": self.query_one("#ssh-key", Input).value.strip(),
            "remote_project_path": self.query_one("#ssh-remote-path", Input).value.strip(),
        }

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#ssh-status", Static).update(text)
        except Exception:
            pass

    def _fill_form(self, profile: SSHProfile) -> None:
        self.query_one("#ssh-name", Input).value = profile.name
        self.query_one("#ssh-host", Input).value = profile.host
        self.query_one("#ssh-user", Input).value = profile.user
        self.query_one("#ssh-port", Input).value = str(profile.port)
        self.query_one("#ssh-password", Input).value = profile.password
        self.query_one("#ssh-key", Input).value = profile.key_path
        self.query_one("#ssh-remote-path", Input).value = profile.remote_project_path
        self._editing = profile.name
        self._set_save_mode(editing=True)

    def _clear_form(self) -> None:
        self.query_one("#ssh-name", Input).value = ""
        self.query_one("#ssh-host", Input).value = ""
        self.query_one("#ssh-user", Input).value = ""
        self.query_one("#ssh-port", Input).value = "22"
        self.query_one("#ssh-password", Input).value = ""
        self.query_one("#ssh-key", Input).value = ""
        self.query_one("#ssh-remote-path", Input).value = "~/OpenMMLA"
        self._editing = None
        self._set_save_mode(editing=False)
        self._set_status("")

    def _set_save_mode(self, editing: bool) -> None:
        """update the primary action label for add vs edit mode."""
        try:
            button = self.query_one("#ssh-save", Button)
            button.label = "Save Profile" if editing else "Add Profile"
            button.variant = "primary" if editing else "success"
        except Exception:
            pass

    def _build_profile_from_form(self) -> SSHProfile | None:
        vals = self._get_form_values()
        if not vals["name"]:
            self._set_status("[red]Profile name is required.[/red]")
            return None
        if not vals["host"]:
            self._set_status("[red]Host is required.[/red]")
            return None
        if not vals["user"]:
            self._set_status("[red]User is required.[/red]")
            return None
        try:
            port = int(vals["port"]) if vals["port"] else 22
        except ValueError:
            self._set_status("[red]Port must be a number.[/red]")
            return None
        return SSHProfile(
            name=vals["name"],
            host=vals["host"],
            user=vals["user"],
            port=port,
            password=vals["password"],
            key_path=vals["key_path"],
            remote_project_path=vals["remote_project_path"] or "~/OpenMMLA",
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "ssh-save":
            await self._save_profile()
        elif btn_id == "ssh-test":
            await self._test_connection(event.button)
        elif btn_id == "ssh-clear":
            self._clear_form()
        elif event.button.has_class("ssh-row-test"):
            await self._test_profile_row(event.button.name or "", event.button)
        elif event.button.has_class("ssh-row-edit"):
            name = event.button.name or ""
            profile = next((p for p in self._profiles if p.name == name), None)
            if profile:
                self._fill_form(profile)
                self._set_status(f"Editing profile '{name}'.")
        elif event.button.has_class("ssh-row-del"):
            await self._delete_profile(event.button.name or "")

    async def _save_profile(self) -> None:
        profile = self._build_profile_from_form()
        if profile is None:
            return
        names = [p.name for p in self._profiles]
        editing = self._editing if self._editing in names else None
        if editing and profile.name != editing and profile.name in names:
            self._set_status(f"[red]A profile named '{profile.name}' already exists.[/red]")
            return
        # an edit keeps the profile's place in the list, under its new name when
        # it was renamed (the old name used to stay behind as a second profile);
        # only a new profile goes to the end
        replaced = editing or (profile.name if profile.name in names else None)
        if replaced is None:
            self._profiles.append(profile)
        else:
            self._profiles = [profile if p.name == replaced else p for p in self._profiles]
        save_ssh_profiles(self._profiles)
        renamed = (editing, profile.name) if editing and editing != profile.name else None
        if renamed:
            from openmmla.tui.ssh import TARGET_STATES
            if renamed[0] in TARGET_STATES:
                TARGET_STATES[renamed[1]] = TARGET_STATES.pop(renamed[0])
        # the form still shows this profile: further changes are edits of it
        self._editing = profile.name
        self._set_save_mode(editing=True)
        self._set_status(
            f"[green]Profile '{renamed[0]}' renamed to '{profile.name}' and saved.[/green]" if renamed
            else f"[green]Profile '{profile.name}' saved.[/green]"
        )
        await self._rebuild_list(show=profile.name)
        self.post_message(self.ProfilesChanged(renamed=renamed))

    async def _delete_profile(self, name: str) -> None:
        self._profiles = [p for p in self._profiles if p.name != name]
        save_ssh_profiles(self._profiles)
        if self._editing == name:
            self._clear_form()
        self._set_status(f"[red]Profile '{name}' deleted.[/red]")
        await self._rebuild_list()
        self.post_message(self.ProfilesChanged())

    async def _test_connection(self, button: Button | None = None) -> None:
        profile = self._build_profile_from_form()
        if profile is None:
            return
        self._set_status("[yellow]Testing connection...[/yellow]")
        if button is not None:
            button.disabled = True
            button.label = "Testing..."
        try:
            loop = asyncio.get_running_loop()
            success, msg = await loop.run_in_executor(None, ssh_test_connection, profile)
            if success:
                self._set_status(f"[green]{msg}[/green]")
                self.post_message(self.ConnectionTested())
            else:
                self._set_status(f"[red]{msg}[/red]")
        finally:
            if button is not None:
                button.disabled = False
                button.label = "Test Connection"

    async def _test_profile_row(self, name: str, button: Button) -> None:
        """test one saved profile and update its shared host state."""
        profile = next((p for p in self._profiles if p.name == name), None)
        if profile is None:
            return
        self._set_status(f"[yellow]Testing connection to '{name}'...[/yellow]")
        button.disabled = True
        button.label = "..."
        try:
            success, msg = await asyncio.to_thread(ssh_test_connection, profile)
            from openmmla.tui.ssh import TARGET_STATES
            TARGET_STATES[name] = "online" if success else "offline"
            if success:
                self._set_status(f"[green]'{name}': {msg}[/green]")
            else:
                self._set_status(f"[red]'{name}': {msg}[/red]")
            self.post_message(self.ConnectionTested())
        finally:
            button.disabled = False
            button.label = "Test"

    async def _rebuild_list(self, show: str | None = None) -> None:
        """reload the profile list display by re-mounting; `show` names the
        profile to bring into view (the list is short, and a saved profile that
        sits below the fold looks like a save that did nothing)."""
        try:
            container = self.query_one("#ssh-profile-list")
            await container.remove_children()
            shown = None
            for p in self._profiles:
                h = Horizontal(
                    Static(_profile_row_text(p), classes="profile-entry-name"),
                    *_profile_row_buttons(p.name),
                    classes="profile-entry",
                )
                await container.mount(h)
                if p.name == show:
                    shown = h
            if shown is not None:
                self.call_after_refresh(shown.scroll_visible, animate=False)
        except Exception:
            pass
