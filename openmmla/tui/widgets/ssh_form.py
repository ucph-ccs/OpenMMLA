from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label

from openmmla.tui.ssh import (
    SSHProfile,
    load_ssh_profiles,
    save_ssh_profiles,
    ssh_test_connection,
)


class SSHForm(Widget):
    """form widget for creating / editing / deleting SSH profiles."""

    class ProfilesChanged(Message):
        """posted when the profile list is modified."""

    DEFAULT_CSS = """
    SSHForm {
        height: auto;
        padding: 1 2;
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
        height: auto;
    }
    SSHForm .profile-entry {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
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
        with Vertical():
            yield Static("[b]SSH Profiles[/b]", classes="ssh-title")

            with Vertical(classes="ssh-profile-list", id="ssh-profile-list"):
                for p in self._profiles:
                    auth = "password" if p.password else ("key" if p.key_path else "default")
                    with Horizontal(classes="profile-entry"):
                        yield Static(
                            f"{p.name}  ({p.user}@{p.host}:{p.port})  [{auth}]",
                            classes="profile-entry-name",
                        )
                        yield Button("Edit", id=f"ssh-edit-{p.name}")
                        yield Button("Delete", variant="error", id=f"ssh-del-{p.name}")

            yield Static("[b]Add / Edit Profile[/b]", classes="ssh-title")

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

            with Horizontal(classes="ssh-actions"):
                yield Button("Save Profile", variant="primary", id="ssh-save")
                yield Button("Test Connection", variant="warning", id="ssh-test")
                yield Button("Clear Form", variant="default", id="ssh-clear")

            yield Static("", id="ssh-status", classes="ssh-status")

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

    def _clear_form(self) -> None:
        self.query_one("#ssh-name", Input).value = ""
        self.query_one("#ssh-host", Input).value = ""
        self.query_one("#ssh-user", Input).value = ""
        self.query_one("#ssh-port", Input).value = "22"
        self.query_one("#ssh-password", Input).value = ""
        self.query_one("#ssh-key", Input).value = ""
        self.query_one("#ssh-remote-path", Input).value = "~/OpenMMLA"
        self._editing = None
        self._set_status("")

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
            self._test_connection()
        elif btn_id == "ssh-clear":
            self._clear_form()
        elif btn_id.startswith("ssh-edit-"):
            name = btn_id[len("ssh-edit-"):]
            profile = next((p for p in self._profiles if p.name == name), None)
            if profile:
                self._fill_form(profile)
                self._set_status(f"Editing profile '{name}'.")
        elif btn_id.startswith("ssh-del-"):
            name = btn_id[len("ssh-del-"):]
            await self._delete_profile(name)

    async def _save_profile(self) -> None:
        profile = self._build_profile_from_form()
        if profile is None:
            return
        self._profiles = [p for p in self._profiles if p.name != profile.name]
        self._profiles.append(profile)
        save_ssh_profiles(self._profiles)
        self._editing = None
        self._set_status(f"[green]Profile '{profile.name}' saved.[/green]")
        await self._rebuild_list()
        self.post_message(self.ProfilesChanged())

    async def _delete_profile(self, name: str) -> None:
        self._profiles = [p for p in self._profiles if p.name != name]
        save_ssh_profiles(self._profiles)
        if self._editing == name:
            self._clear_form()
        self._set_status(f"[red]Profile '{name}' deleted.[/red]")
        await self._rebuild_list()
        self.post_message(self.ProfilesChanged())

    def _test_connection(self) -> None:
        profile = self._build_profile_from_form()
        if profile is None:
            return
        self._set_status("[yellow]Testing connection...[/yellow]")
        success, msg = ssh_test_connection(profile)
        if success:
            self._set_status(f"[green]{msg}[/green]")
        else:
            self._set_status(f"[red]{msg}[/red]")

    async def _rebuild_list(self) -> None:
        """reload the profile list display by re-mounting."""
        try:
            container = self.query_one("#ssh-profile-list")
            await container.remove_children()
            for p in self._profiles:
                auth = "password" if p.password else ("key" if p.key_path else "default")
                h = Horizontal(
                    Static(
                        f"{p.name}  ({p.user}@{p.host}:{p.port})  [{auth}]",
                        classes="profile-entry-name",
                    ),
                    Button("Edit", id=f"ssh-edit-{p.name}"),
                    Button("Delete", variant="error", id=f"ssh-del-{p.name}"),
                    classes="profile-entry",
                )
                await container.mount(h)
        except Exception:
            pass
