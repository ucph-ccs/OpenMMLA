from __future__ import annotations

import asyncio
import subprocess

from rich.markup import escape as rich_escape
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, DataTable, RichLog, Button, Select, Label, Input

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, ssh_run_async,
    ssh_test_connection, git_remote_url,
)

_CONDA_INIT = (
    'for _p in ~/miniforge3 ~/miniconda3 ~/anaconda3 ~/mambaforge /opt/conda; do '
    '[ -f "$_p/etc/profile.d/conda.sh" ] && . "$_p/etc/profile.d/conda.sh" && break; '
    'done'
)

_REMOTE_SHELL_INIT = f'export LANG=en_US.UTF-8 PYTHONUNBUFFERED=1; {_CONDA_INIT}'


def _wrap_local(cmd: str, conda_env: str = "") -> str:
    """wrap a local command with conda init and optional activate."""
    activate = f"conda activate {conda_env} && " if conda_env else ""
    escaped = cmd.replace("'", "'\\''")
    return f"bash -c 'export PYTHONUNBUFFERED=1; {_CONDA_INIT}; {activate}{escaped}'"


def _wrap_remote(cmd: str, conda_env: str = "") -> str:
    """wrap a shell command with locale, conda init and optional activate for remote SSH."""
    activate = f"conda activate {conda_env} && " if conda_env else ""
    escaped = cmd.replace("'", "'\\''")
    return f"bash -c '{_REMOTE_SHELL_INIT}; {activate}{escaped}'"


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


def _list_conda_envs() -> set[str]:
    """return set of existing conda environment names."""
    envs = set()
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                envs.add(parts[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return envs


def _parse_remote_conda_envs(output: str) -> set[str]:
    """parse conda env list output into a set of env names."""
    envs = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            envs.add(parts[0])
    return envs


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
    #env-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
    }
    #env-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #env-target-bar Select {
        width: 1fr;
    }
    #env-table {
        height: 1fr;
    }
    #env-actions {
        height: auto;
        padding: 1;
    }
    #env-actions Button {
        margin: 0 1;
        min-width: 16;
    }
    #env-log {
        height: 14;
        border-top: solid $primary;
        padding: 0 1;
    }
    #env-cmd-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #env-cmd-bar Label {
        width: auto;
        min-width: 4;
        padding-top: 1;
        padding-right: 1;
    }
    #env-cmd-input {
        width: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._conda_envs: set[str] = set()
        self._selected_group: str | None = None
        self._connected: bool = False
        self._remote_cwd: str = ""
        self._active_conda_env: str = ""
        self._running_proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        target_options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        with Vertical():
            yield Static(
                "Conda Environment Manager — select a target and a row, then use actions",
                id="env-header",
            )
            with Horizontal(id="env-target-bar"):
                yield Label("Target:")
                yield Select(target_options, value="local", id="env-target-select")
            yield DataTable(id="env-table")
            with Horizontal(id="env-actions"):
                yield Button("Connect", variant="default", id="btn-connect")
                yield Button("Refresh", variant="default", id="btn-env-refresh")
                yield Button("Git Clone", variant="warning", id="btn-git-clone")
                yield Button("Git Pull", variant="warning", id="btn-git-pull")
                yield Button("Create Env", variant="primary", id="btn-create-env")
                yield Button("Install Deps", variant="success", id="btn-install-deps")
            yield RichLog(id="env-log", highlight=True, markup=True)
            with Horizontal(id="env-cmd-bar"):
                yield Label("$", id="env-cmd-label")
                yield Input(
                    placeholder="type a command and press Enter (clear to reset log)",
                    id="env-cmd-input",
                )

    def on_mount(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.add_columns("Conda Env", "Dep Group", "Python", "Status", "Description")
        table.cursor_type = "row"
        self._refresh_table()

    def on_show(self) -> None:
        self._refresh_target_selector()

    def _refresh_target_selector(self) -> None:
        """reload SSH profile names into the target selector."""
        options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        try:
            sel = self.query_one("#env-target-select", Select)
            current = sel.value
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
        except Exception:
            pass

    def _get_target(self) -> str:
        try:
            sel = self.query_one("#env-target-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                return "local"
            return str(val)
        except Exception:
            return "local"

    @property
    def _is_remote(self) -> bool:
        return self._get_target() != "local"

    def _refresh_table(self) -> None:
        if self._is_remote:
            self.run_worker(self._refresh_table_remote(), exclusive=True)
        else:
            self._conda_envs = _list_conda_envs()
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
        self._log(f"[yellow]Checking conda envs on '{profile.name}'...[/yellow]")
        proc = await ssh_run_async(profile, _wrap_remote("conda env list"))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        self._conda_envs = _parse_remote_conda_envs(output)
        self._populate_table()
        self._log("[green]Remote env list refreshed.[/green]")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#env-table", DataTable)
        try:
            row = table.get_row(event.row_key)
            self._selected_group = str(row[1])
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#env-log", RichLog).write(msg)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-connect":
            self._test_connect()
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

    def _test_connect(self) -> None:
        target = self._get_target()
        if target == "local":
            self._connected = True
            self._log("[green]Local shell ready.[/green]")
            self._refresh_table()
            return
        profile = get_profile_by_name(target)
        if profile is None:
            self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[yellow]Testing connection to '{profile.name}'...[/yellow]")
        success, msg = ssh_test_connection(profile)
        if success:
            self._connected = True
            self._log(f"[green]Connected to {profile.name} ({profile.ssh_destination()}) — {msg}[/green]")
            self._refresh_table()
        else:
            self._connected = False
            self._log(f"[red]Connection failed: {msg}[/red]")

    def _get_prompt(self) -> str:
        env_prefix = f"({self._active_conda_env}) " if self._active_conda_env else ""
        target = self._get_target()
        if target == "local":
            return f"{env_prefix}$"
        profile = get_profile_by_name(target)
        cwd = self._remote_cwd or (profile.remote_project_path if profile else "~")
        return f"{env_prefix}{target}:{cwd}$"

    def _update_prompt(self) -> None:
        try:
            label = self.query_one("#env-cmd-label", Label)
            label.update(self._get_prompt())
        except Exception:
            pass

    def _proc_is_running(self) -> bool:
        return self._running_proc is not None and self._running_proc.returncode is None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "env-cmd-input":
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if self._proc_is_running():
            assert self._running_proc is not None
            stdin = self._running_proc.stdin
            if stdin is not None:
                self._log(f"[dim]> {rich_escape(text)}[/dim]")
                stdin.write((text + "\n").encode())
                asyncio.ensure_future(stdin.drain())
            return

        if text == "clear":
            try:
                self.query_one("#env-log", RichLog).clear()
            except Exception:
                pass
            return

        if text.startswith("conda activate "):
            env_name = text[len("conda activate "):].strip()
            if env_name:
                if env_name not in self._conda_envs:
                    self._log(f"[red]Env '{env_name}' not found. Click Refresh to update the env list.[/red]")
                    return
                self._active_conda_env = env_name
                self._log(f"[green]Activated conda env: {env_name}[/green]")
                self._update_prompt()
            return

        if text in ("conda deactivate", "conda deactivate "):
            prev = self._active_conda_env
            self._active_conda_env = ""
            self._log(f"[yellow]Deactivated conda env: {prev}[/yellow]" if prev else "[yellow]No active env.[/yellow]")
            self._update_prompt()
            return

        target = self._get_target()
        if target == "local":
            self._log(f"[bold]$ {rich_escape(text)}[/bold]")
            self.run_worker(self._run_local_cmd(text), exclusive=True)
        else:
            self._log(f"[bold]{self._get_prompt()} {rich_escape(text)}[/bold]")
            if text.startswith("cd ") or text == "cd":
                self.run_worker(self._run_remote_cd(target, text), exclusive=True)
            else:
                self.run_worker(self._run_remote_cmd(target, text), exclusive=True)

    async def _run_local_cmd(self, cmd: str) -> None:
        wrapped = _wrap_local(cmd, self._active_conda_env)
        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self._root,
        )
        self._running_proc = proc
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
        self._running_proc = None
        if rc != 0:
            self._log(f"[red](exit {rc})[/red]")

    async def _run_remote_cd(self, profile_name: str, cmd: str) -> None:
        """handle cd on remote by resolving the new path via pwd."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cwd = self._remote_cwd or profile.remote_project_path
        resolve_cmd = _wrap_remote(f"cd {cwd} && {cmd} && pwd", self._active_conda_env)
        proc = await ssh_run_async(profile, resolve_cmd)
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode()
        rc = await proc.wait()
        if rc == 0:
            new_cwd = output.strip().splitlines()[-1] if output.strip() else cwd
            self._remote_cwd = new_cwd
            self._log(f"[green]→ {new_cwd}[/green]")
            self._update_prompt()
        else:
            self._log(f"[red]{output.strip()}[/red]")

    async def _run_remote_cmd(self, profile_name: str, cmd: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cwd = self._remote_cwd or profile.remote_project_path
        wrapped = _wrap_remote(f"cd {cwd} && {cmd}", self._active_conda_env)
        proc = await ssh_run_async(profile, wrapped, pipe_stdin=True)
        self._running_proc = proc
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
        self._running_proc = None
        if rc != 0:
            self._log(f"[red](exit {rc})[/red]")

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
        proc = await ssh_run_async(profile, _wrap_remote(cmd))
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
        proc = await ssh_run_async(profile, _wrap_remote(cmd))
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
        proc = await ssh_run_async(profile, _wrap_remote(cmd))
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
        proc = await ssh_run_async(profile, _wrap_remote(cmd))
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
