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
     "description": "ASR base station",
     "packages": ["openmmla", "influxdb-client", "paho-mqtt", "pymongo", "pyyaml", "redis"]},
    {"group": "asr-server-nemo", "env": "asr-server-nemo", "python": "3.10",
     "description": "ASR server (NeMo backend)",
     "packages": ["openmmla", "flask", "gunicorn", "nemo-toolkit", "onnxruntime", "whisperx"]},
    {"group": "asr-server-wespeaker", "env": "asr-server-wespeaker", "python": "3.10",
     "description": "ASR server (WeSpeaker backend)",
     "packages": [
         "openmmla", "flask", "gunicorn", "wespeaker", "s3prl", "peft",
         "accelerate", "openai-whisper", "hdbscan", "umap-learn", "librosa",
     ]},
    {"group": "vfa-base", "env": "vfa-base", "python": "3.10",
     "description": "VFA base station",
     "packages": ["openmmla", "cv2", "influxdb-client", "paho-mqtt", "pymongo", "pyyaml", "redis"]},
    {"group": "vfa-server", "env": "vfa-server", "python": "3.10",
     "description": "VFA server wrapper",
     "packages": ["openmmla", "flask", "gunicorn", "openai", "opencv-python", "pupil-apriltags"]},
    {"group": "vfa-vllm-runtime", "env": "vfa-vllm", "python": "3.12",
     "description": "VFA local vLLM runtime",
     "packages": ["openmmla", "qwen-vl-utils", "transformers", "vllm"]},
    {"group": "ips-base", "env": "ips-base", "python": "3.10",
     "description": "IPS base station",
     "packages": ["openmmla", "influxdb-client", "opencv-python", "paho-mqtt", "pupil-apriltags"]},
    {"group": "uber-base", "env": "uber-base", "python": "3.10",
     "description": "Analysis framework",
     "packages": ["openmmla", "influxdb-client", "pandas", "pyecharts", "pymongo", "redis"]},
    {"group": "uber-server", "env": "uber-server", "python": "3.10",
     "description": "Dashboard & infrastructure services",
     "packages": ["openmmla", "celery", "flask", "flask-socketio", "influxdb-client", "redis"]},
    {"group": "tui", "env": "tui", "python": "3.10",
     "description": "TUI management console",
     "packages": ["openmmla", "cryptography", "pyyaml", "textual"]},
]


def _target_options() -> list[tuple[str, str]]:
    return [("Local", "local")] + [
        (p.name, p.name) for p in load_ssh_profiles()
    ]


def _normalize_target(value) -> str:
    if value is Select.BLANK or value is None:
        return "local"
    text = str(value)
    return text if text else "local"


def _normalize_package_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _parse_conda_packages(output: str) -> set[str]:
    packages: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            packages.add(_normalize_package_name(parts[0]))
    return packages


def _list_conda_packages_sync(env_name: str) -> set[str]:
    try:
        result = subprocess.run(
            ["conda", "list", "-n", env_name],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0:
            return _parse_conda_packages(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return set()


def _missing_packages(entry: dict, installed_packages: set[str]) -> list[str]:
    required = [_normalize_package_name(pkg) for pkg in entry.get("packages", [])]
    missing = [pkg for pkg in required if pkg not in installed_packages]
    if "opencv-python" in missing and "cv2" in installed_packages:
        missing.remove("opencv-python")
    if "cv2" in missing and "opencv-python" in installed_packages:
        missing.remove("cv2")
    return missing


def _format_env_status(entry: dict, conda_envs: set[str], env_packages: dict[str, set[str]]) -> str:
    env_name = entry["env"]
    if env_name not in conda_envs:
        return "Missing"
    missing = _missing_packages(entry, env_packages.get(env_name, set()))
    if missing:
        suffix = ", ".join(missing[:2])
        if len(missing) > 2:
            suffix += f" +{len(missing) - 2}"
        return f"Partial: {suffix}"
    return "Ready"


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
        self._env_packages: dict[str, set[str]] = {}
        self._selected_group: str | None = None
        self._pending_delete: tuple[str, str] | None = None
        self._target = "local"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "Conda Environment Manager — select a target and a row, then use actions",
                id="env-header",
            )
            yield DataTable(id="env-table")
            with Horizontal(id="env-target-bar"):
                yield Label("Target:")
                yield Select(_target_options(), value="local", id="env-target-select")
            with Horizontal(id="env-actions"):
                yield Button("Connect", variant="primary", id="btn-connect")
                yield Button("Refresh", variant="primary", id="btn-env-refresh")
                yield Button("Git Clone", variant="warning", id="btn-git-clone")
                yield Button("Git Pull", variant="warning", id="btn-git-pull")
                yield Button("Create Env", variant="success", id="btn-create-env")
                yield Button("Install Deps", variant="success", id="btn-install-deps")
                yield Button("Delete Env", variant="error", id="btn-delete-env")
            yield CommandSession(show_target=False, id="env-cmd-session")

    @property
    def _cmd(self) -> CommandSession:
        return self.query_one("#env-cmd-session", CommandSession)

    @property
    def _is_remote(self) -> bool:
        return self._get_target() != "local"

    def _get_target(self) -> str:
        return self._target

    def _get_selected_target(self) -> str:
        try:
            sel = self.query_one("#env-target-select", Select)
            return _normalize_target(sel.value)
        except Exception:
            return self._target

    def _set_target(self, target: str) -> None:
        normalized = _normalize_target(target)
        if normalized != self._target:
            self._pending_delete = None
        self._target = normalized
        try:
            self._cmd.set_target(self._target)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        self._cmd.log(msg)

    def on_mount(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.add_columns("Conda Env", "Dep Group", "Python", "Status", "Description")
        table.cursor_type = "row"
        self._refresh_table()

    def on_show(self) -> None:
        self._refresh_target_options()

    def _refresh_target_options(self) -> None:
        options = _target_options()
        try:
            sel = self.query_one("#env-target-select", Select)
            current = sel.value
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
                self._set_target(_normalize_target(current))
            else:
                sel.value = "local"
                self._set_target("local")
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "env-target-select":
            self._set_target(_normalize_target(event.value))
            self._refresh_table()

    # -- table -----------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._set_target(self._get_selected_target())
        target = self._get_target()
        if target != "local":
            self.run_worker(self._refresh_table_remote(target), exclusive=True)
        else:
            self.run_worker(self._refresh_table_local(target), exclusive=True)

    async def _refresh_table_local(self, target: str) -> None:
        if target != self._get_target():
            return
        conda_envs = await asyncio.to_thread(_list_conda_envs_sync)
        env_packages = await asyncio.to_thread(self._list_selected_conda_packages, conda_envs)
        if target != self._get_target():
            return
        self._conda_envs = conda_envs
        self._env_packages = env_packages
        self._populate_table()

    def _list_selected_conda_packages(self, conda_envs: set[str]) -> dict[str, set[str]]:
        packages: dict[str, set[str]] = {}
        for entry in ENV_GROUPS:
            env_name = entry["env"]
            if env_name in conda_envs:
                packages[env_name] = _list_conda_packages_sync(env_name)
        return packages

    def _populate_table(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.clear()
        for eg in ENV_GROUPS:
            status = _format_env_status(eg, self._conda_envs, self._env_packages)
            table.add_row(
                eg["env"], eg["group"], eg["python"], status, eg["description"],
            )

    async def _refresh_table_remote(self, profile_name: str) -> None:
        if profile_name != self._get_target():
            return
        profile = get_profile_by_name(profile_name)
        if profile is None:
            if profile_name == self._get_target():
                self._log("[red]SSH profile not found.[/red]")
            return
        self._log(f"[yellow]Checking conda envs on '{profile_name}'...[/yellow]")
        proc = await ssh_run_async(profile, wrap_remote("conda env list"))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        if profile_name != self._get_target():
            return
        self._conda_envs = _parse_conda_envs(output)
        env_packages = await self._list_remote_conda_packages(profile_name, self._conda_envs)
        if profile_name != self._get_target():
            return
        self._env_packages = env_packages
        self._populate_table()
        self._log("[green]Remote env list refreshed.[/green]")

    async def _list_remote_conda_packages(
        self,
        profile_name: str,
        conda_envs: set[str],
    ) -> dict[str, set[str]]:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return {}
        env_names = [
            entry["env"] for entry in ENV_GROUPS
            if entry["env"] in conda_envs
        ]
        if not env_names:
            return {}
        cmd_parts = []
        for env_name in env_names:
            cmd_parts.append(
                f"echo __OPENMMLA_ENV__{env_name}; "
                f"conda list -n {env_name} 2>/dev/null || true"
            )
        proc = await ssh_run_async(profile, wrap_remote("; ".join(cmd_parts)))
        output = ""
        assert proc.stdout is not None
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        packages: dict[str, set[str]] = {}
        current_env: str | None = None
        current_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith("__OPENMMLA_ENV__"):
                if current_env is not None:
                    packages[current_env] = _parse_conda_packages("\n".join(current_lines))
                current_env = line.replace("__OPENMMLA_ENV__", "", 1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_env is not None:
            packages[current_env] = _parse_conda_packages("\n".join(current_lines))
        return packages

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#env-table", DataTable)
        try:
            row = table.get_row(event.row_key)
            self._selected_group = str(row[1])
            self._pending_delete = None
        except Exception:
            pass

    # -- buttons ---------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-connect":
            self._refresh_target_options()
            self._set_target(self._get_selected_target())
            self._cmd.connect()
            self._refresh_table()
        elif bid == "btn-env-refresh":
            self._refresh_target_options()
            self._set_target(self._get_selected_target())
            self._refresh_table()
        elif bid == "btn-create-env":
            self._create_selected_env()
        elif bid == "btn-install-deps":
            self._install_selected_deps()
        elif bid == "btn-delete-env":
            self._delete_selected_env()
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

    def _delete_selected_env(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        env_name = entry["env"]
        if env_name not in self._conda_envs:
            self._log(f"[yellow]Env '{env_name}' does not exist.[/yellow]")
            return
        delete_key = (self._get_target(), env_name)
        if self._pending_delete != delete_key:
            self._pending_delete = delete_key
            target_label = "local" if delete_key[0] == "local" else f"remote target '{delete_key[0]}'"
            self._log(
                f"[yellow]Press Delete Env again to permanently delete '{env_name}' on {target_label}.[/yellow]"
            )
            return
        self._pending_delete = None
        self._log(f"[red]Deleting conda env '{env_name}'...[/red]")
        if self._is_remote:
            self.run_worker(
                self._run_remote_delete(self._get_target(), env_name),
                exclusive=True,
            )
        else:
            self.run_worker(self._run_delete(env_name), exclusive=True)

    async def _run_delete(self, env_name: str) -> None:
        cmd = ["conda", "env", "remove", "-n", env_name, "-y"]
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
            self._log(f"[green]Env '{env_name}' deleted.[/green]")
        else:
            self._log(f"[red]Failed to delete env '{env_name}' (exit {rc}).[/red]")
        self._refresh_table()

    async def _run_remote_delete(self, profile_name: str, env_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cmd = f"conda env remove -n {env_name} -y"
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
            self._log(f"[green]Env '{env_name}' deleted remotely.[/green]")
        else:
            self._log(f"[red]Remote env deletion failed (exit {rc}).[/red]")
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
        self._refresh_table()

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
        self._refresh_table()
