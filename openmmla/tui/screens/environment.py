from __future__ import annotations

import asyncio
import subprocess

from rich.markup import escape as rich_escape
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, DataTable, Button, Select, Label

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, ssh_run_async,
    git_remote_url, wrap_remote,
)
from openmmla.tui.widgets.command_session import CommandSession, _list_conda_envs_sync, _parse_conda_envs


ENV_GROUPS = [
    {"group": "asr-base", "env": "asr-base", "python": "3.10",
     "description": "ASR base station"},
    {"group": "asr-server", "env": "asr-server", "python": "3.10",
     "description": "ASR server (NeMo backend)"},
    {"group": "asr-server-wespeaker", "env": "asr-server-wespeaker", "python": "3.10",
     "description": "ASR server (WeSpeaker backend)"},
    {"group": "vfa-base", "env": "vfa-base", "python": "3.10",
     "description": "VFA base station"},
    {"group": "vfa-server", "env": "vfa-server", "python": "3.10",
     "description": "VFA server"},
    {"group": "ips-base", "env": "ips-base", "python": "3.10",
     "description": "IPS base station"},
    {"group": "uber-base", "env": "uber-base", "python": "3.10",
     "description": "Analysis framework"},
    {"group": "uber-server", "env": "uber-server", "python": "3.10",
     "description": "Dashboard & infrastructure services"},
    {"group": "tui", "env": "tui", "python": "3.10",
     "description": "TUI management console"},
]


class EnvironmentPanel(Widget):

    DEFAULT_CSS = """
    EnvironmentPanel {
        width: 1fr;
        height: 1fr;
    }
    #env-header {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }
    #env-table {
        height: 1fr;
    }
    #env-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #env-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #env-target-bar Select {
        width: 1fr;
    }
    #env-actions {
        height: auto;
        padding: 1;
    }
    #env-actions Button {
        margin: 0 1;
        min-width: 16;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._conda_envs: set[str] = set()
        self._selected_group: str | None = None

    def compose(self) -> ComposeResult:
        from openmmla.tui.ssh import load_ssh_profiles
        target_options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        with Vertical():
            yield Static(
                "Conda Environment Manager — select a target and a row, then use actions",
                id="env-header",
            )
            yield DataTable(id="env-table")
            with Horizontal(id="env-target-bar"):
                yield Label("Target:")
                yield Select(target_options, value="local", id="env-target-select")
            with Horizontal(id="env-actions"):
                yield Button("Connect", variant="primary", id="btn-connect")
                yield Button("Refresh", variant="primary", id="btn-env-refresh")
                yield Button("Git Clone", variant="warning", id="btn-git-clone")
                yield Button("Git Pull", variant="warning", id="btn-git-pull")
                yield Button("Create Env", variant="success", id="btn-create-env")
                yield Button("Install Deps", variant="success", id="btn-install-deps")
            yield CommandSession(show_target=False, id="env-cmd-session")

    @property
    def _cmd(self) -> CommandSession:
        return self.query_one("#env-cmd-session", CommandSession)

    @property
    def _is_remote(self) -> bool:
        return self._cmd.is_remote

    def _get_target(self) -> str:
        return self._cmd.get_target()

    def _log(self, msg: str) -> None:
        self._cmd.log(msg)

    def on_mount(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.add_columns("Conda Env", "Dep Group", "Python", "Status", "Description")
        table.cursor_type = "row"
        self._refresh_table()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "env-target-select":
            val = event.value
            if val is Select.BLANK or val is None:
                val = "local"
            self._cmd.set_target(str(val))
            self._refresh_table()

    # -- table -----------------------------------------------------------------

    def _refresh_table(self) -> None:
        if self._is_remote:
            self.run_worker(self._refresh_table_remote(), exclusive=True)
        else:
            self._conda_envs = _list_conda_envs_sync()
            self._populate_table()

    def _populate_table(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.clear()
        for eg in ENV_GROUPS:
            exists = eg["env"] in self._conda_envs
            status = "Exists" if exists else "Missing"
            table.add_row(
                eg["env"], eg["group"], eg["python"], status, eg["description"],
            )

    async def _refresh_table_remote(self) -> None:
        profile = get_profile_by_name(self._get_target())
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[yellow]Checking conda envs on '{self._get_target()}'...[/yellow]")
        proc = await ssh_run_async(profile, wrap_remote("conda env list"))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        self._conda_envs = _parse_conda_envs(output)
        self._populate_table()
        self._log("[green]Remote env list refreshed.[/green]")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#env-table", DataTable)
        try:
            row = table.get_row(event.row_key)
            self._selected_group = str(row[1])
        except Exception:
            pass

    # -- buttons ---------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-connect":
            self._cmd.connect()
            self._refresh_table()
        elif bid == "btn-env-refresh":
            self._refresh_table()
            if not self._is_remote:
                self._log("[green]Refreshed.[/green]")
        elif bid == "btn-create-env":
            self._create_selected_env()
        elif bid == "btn-install-deps":
            self._install_selected_deps()
        elif bid == "btn-git-clone":
            self._git_clone()
        elif bid == "btn-git-pull":
            self._git_pull()

    # -- git operations -------------------------------------------------------

    def _git_clone(self) -> None:
        if not self._is_remote:
            self._log("[yellow]Git Clone is for remote targets only.[/yellow]")
            return
        profile = get_profile_by_name(self._get_target())
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        repo_url = git_remote_url()
        if not repo_url:
            self._log("[red]Could not determine git remote URL.[/red]")
            return
        self._log(f"[green]Cloning repo on '{profile.name}'...[/green]")
        self.run_worker(self._run_git_clone(profile.name, repo_url), exclusive=True)

    async def _run_git_clone(self, profile_name: str, repo_url: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        cmd = f"git clone {repo_url} {remote_path}"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log("[green]Git clone completed.[/green]")
        else:
            self._log(f"[red]Git clone failed (exit {rc}).[/red]")

    def _git_pull(self) -> None:
        if not self._is_remote:
            self._log("[yellow]Git Pull is for remote targets only.[/yellow]")
            return
        profile = get_profile_by_name(self._get_target())
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[green]Pulling latest on '{profile.name}'...[/green]")
        self.run_worker(self._run_git_pull(profile.name), exclusive=True)

    async def _run_git_pull(self, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        cmd = f"cd {remote_path} && git pull"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log("[green]Git pull completed.[/green]")
        else:
            self._log(f"[red]Git pull failed (exit {rc}).[/red]")

    # -- conda operations -----------------------------------------------------

    def _get_selected_entry(self) -> dict | None:
        if not self._selected_group:
            self._log("[yellow]Select a row first.[/yellow]")
            return None
        entry = next(
            (e for e in ENV_GROUPS if e["group"] == self._selected_group), None,
        )
        if entry is None:
            self._log(f"[red]Unknown group: {self._selected_group}[/red]")
        return entry

    def _create_selected_env(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        python_ver = entry["python"]
        if env_name in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' already exists.[/yellow]")
            return
        self._log(f"[green]Creating conda env '{env_name}' (python={python_ver})...[/green]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_create(self._get_target(), env_name, python_ver),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_create(env_name, python_ver), exclusive=True)

    async def _run_create(self, env_name: str, python_ver: str) -> None:
        cmd = ["conda", "create", "-n", env_name, f"python={python_ver}", "-y"]
        self._log(f"  $ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' created successfully.[/green]")
        else:
            self._log(f"[red]Failed to create env '{env_name}' (exit {rc}).[/red]")
        self._refresh_table()

    async def _run_remote_create(self, profile_name: str, env_name: str, python_ver: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cmd = f"conda create -n {env_name} python={python_ver} -y"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Env '{env_name}' created remotely.[/green]")
        else:
            self._log(f"[red]Remote env creation failed (exit {rc}).[/red]")
        self._refresh_table()

    def _install_selected_deps(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        group = entry["group"]
        if env_name not in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' does not exist. Create it first.[/yellow]")
            return
        self._log(f"[green]Installing deps '{rich_escape(f'[{group}]')}' into env '{env_name}'...[/green]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_install(self._get_target(), env_name, group),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_install(env_name, group), exclusive=True)

    async def _run_install(self, env_name: str, group: str) -> None:
        cmd = [
            "conda", "run", "-n", env_name,
            "pip", "install", "-e", f".[{group}]",
        ]
        self._log(f"  $ {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self._root,
        )
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Dependencies '{rich_escape(f'[{group}]')}' installed successfully.[/green]")
        else:
            self._log(f"[red]Installation failed (exit {rc}).[/red]")

    async def _run_remote_install(self, profile_name: str, env_name: str, group: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_path = profile.remote_project_path
        cmd = f"cd {remote_path} && conda run -n {env_name} pip install -e '.[{group}]'"
        self._log(f"  remote$ {rich_escape(cmd)}")
        proc = await ssh_run_async(profile, wrap_remote(cmd))
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode(errors="replace").rstrip()
            if text:
                for line in text.splitlines():
                    self._log(rich_escape(line))
        rc = await proc.wait()
        if rc == 0:
            self._log(f"[green]Remote install of '{rich_escape(f'[{group}]')}' completed.[/green]")
        else:
            self._log(f"[red]Remote install failed (exit {rc}).[/red]")
