from __future__ import annotations

import asyncio
import os
import subprocess

from rich.markup import escape as rich_escape
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import RichLog, Label, Input, Select

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, ssh_run_async,
    ssh_test_connection, wrap_local, wrap_remote,
)


def _list_conda_envs_sync() -> set[str]:
    """return set of existing conda environment names (local)."""
    envs: set[str] = set()
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


def _parse_conda_envs(output: str) -> set[str]:
    """parse conda env list output into a set of env names."""
    envs: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            envs.add(parts[0])
    return envs


class CommandSession(Widget):
    """reusable interactive command session with target selector, log, and input."""

    class TargetChanged(Message):
        """emitted when the user changes the target selector."""
        def __init__(self, target: str) -> None:
            super().__init__()
            self.target = target

    DEFAULT_CSS = """
    CommandSession {
        height: auto;
        min-height: 8;
    }
    .cmd-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    .cmd-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    .cmd-target-bar Select {
        width: 1fr;
    }
    .cmd-log {
        height: 4;
        border-top: solid $primary;
        padding: 0 1;
    }
    .cmd-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    .cmd-bar Label {
        width: auto;
        min-width: 4;
        padding-top: 1;
        padding-right: 1;
    }
    .cmd-input {
        width: 1fr;
    }
    """

    def __init__(self, show_target: bool = True, id: str | None = None) -> None:
        super().__init__(id=id)
        self._root = _find_project_root()
        self._show_target = show_target
        self._target: str = "local"
        self._local_cwd: str = self._root
        self._remote_cwd: str = ""
        self._active_conda_env: str = ""
        self._running_proc: asyncio.subprocess.Process | None = None
        self._connected: bool = False

    def compose(self) -> ComposeResult:
        target_options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        if self._show_target:
            with Horizontal(classes="cmd-target-bar"):
                yield Label("Target:")
                yield Select(target_options, value="local", id="cmd-target-select")
        yield RichLog(classes="cmd-log", highlight=True, markup=True)
        with Horizontal(classes="cmd-bar"):
            yield Label("$", classes="cmd-prompt")
            yield Input(
                placeholder="type a command and press Enter (clear to reset log)",
                classes="cmd-input",
            )

    def on_show(self) -> None:
        self.refresh_targets()

    def refresh_targets(self) -> None:
        """reload SSH profile names into the target selector."""
        if not self._show_target:
            return
        options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        try:
            sel = self.query_one("#cmd-target-select", Select)
            current = sel.value
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
        except Exception:
            pass

    def get_target(self) -> str:
        if self._show_target:
            try:
                sel = self.query_one("#cmd-target-select", Select)
                val = sel.value
                if val is Select.BLANK or val is None:
                    return "local"
                return str(val)
            except Exception:
                pass
        return self._target

    def set_target(self, target: str) -> None:
        """programmatically set the target and reset state."""
        self._target = target
        if self._show_target:
            try:
                sel = self.query_one("#cmd-target-select", Select)
                sel.value = target
            except Exception:
                pass
        self._reset_state()

    @property
    def is_remote(self) -> bool:
        return self.get_target() != "local"

    def log(self, msg: str) -> None:
        """write a message to the log (public API for parents)."""
        try:
            self.query_one(".cmd-log", RichLog).write(msg)
        except Exception:
            pass

    def run(self, text: str) -> None:
        """programmatically execute a command as if typed by the user."""
        text = text.strip()
        if not text:
            return
        target = self.get_target()
        if target == "local":
            self.log(f"[bold]{self._get_prompt()} {rich_escape(text)}[/bold]")
            if text.startswith("cd ") or text == "cd":
                self.run_worker(self._run_local_cd(text), exclusive=True)
            else:
                self.run_worker(self._run_local_cmd(text), exclusive=True)
        else:
            self.log(f"[bold]{self._get_prompt()} {rich_escape(text)}[/bold]")
            if text.startswith("cd ") or text == "cd":
                self.run_worker(self._run_remote_cd(target, text), exclusive=True)
            else:
                self.run_worker(self._run_remote_cmd(target, text), exclusive=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cmd-target-select":
            self._reset_state()
            self.post_message(self.TargetChanged(self.get_target()))

    def _reset_state(self) -> None:
        self._local_cwd = self._root
        self._remote_cwd = ""
        self._active_conda_env = ""
        self._connected = False
        self._update_prompt()

    # -- prompt ----------------------------------------------------------------

    def _get_prompt(self) -> str:
        env_prefix = f"({self._active_conda_env}) " if self._active_conda_env else ""
        target = self.get_target()
        if target == "local":
            display = self._local_cwd.replace(os.path.expanduser("~"), "~")
            return f"{env_prefix}{display}$"
        profile = get_profile_by_name(target)
        cwd = self._remote_cwd or (profile.remote_project_path if profile else "~")
        return f"{env_prefix}{target}:{cwd}$"

    def _update_prompt(self) -> None:
        try:
            label = self.query_one(".cmd-prompt", Label)
            label.update(self._get_prompt())
        except Exception:
            pass

    # -- connect ---------------------------------------------------------------

    def connect(self) -> None:
        """test connection to the current target."""
        target = self.get_target()
        if target == "local":
            self._connected = True
            self.log("[green]Local shell ready.[/green]")
            self._update_prompt()
            return
        profile = get_profile_by_name(target)
        if profile is None:
            self.log("[red]SSH profile not found.[/red]")
            return
        self.log(f"[yellow]Testing connection to '{profile.name}'...[/yellow]")
        success, msg = ssh_test_connection(profile)
        if success:
            self._connected = True
            self.log(f"[green]Connected to {profile.name} ({profile.ssh_destination()}) — {msg}[/green]")
            self.run_worker(self._resolve_remote_cwd(profile), exclusive=True)
        else:
            self._connected = False
            self.log(f"[red]Connection failed: {msg}[/red]")

    async def _resolve_remote_cwd(self, profile) -> None:
        """check if remote_project_path exists, fall back to ~ if not."""
        check = f"[ -d {profile.remote_project_path} ] && echo EXISTS || echo MISSING"
        proc = await ssh_run_async(profile, check)
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode()
        await proc.wait()
        if "EXISTS" in output:
            self._remote_cwd = profile.remote_project_path
        else:
            self._remote_cwd = "~"
            self.log(f"[yellow]'{profile.remote_project_path}' not found, starting in ~[/yellow]")
        self._update_prompt()

    # -- input handling --------------------------------------------------------

    def _proc_is_running(self) -> bool:
        return self._running_proc is not None and self._running_proc.returncode is None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if "cmd-input" not in (event.input.classes or set()):
            return
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if self._proc_is_running():
            assert self._running_proc is not None
            stdin = self._running_proc.stdin
            if stdin is not None:
                self.log("[dim]> (input sent)[/dim]")
                stdin.write((text + "\n").encode())
                asyncio.ensure_future(stdin.drain())
            return

        if text == "clear":
            try:
                self.query_one(".cmd-log", RichLog).clear()
            except Exception:
                pass
            return

        if text.startswith("conda activate "):
            env_name = text[len("conda activate "):].strip()
            if env_name:
                self.run_worker(self._activate_env(env_name), exclusive=True)
            return

        if text in ("conda deactivate", "conda deactivate "):
            prev = self._active_conda_env
            self._active_conda_env = ""
            self.log(f"[yellow]Deactivated conda env: {prev}[/yellow]" if prev else "[yellow]No active env.[/yellow]")
            self._update_prompt()
            return

        target = self.get_target()
        if target == "local":
            self.log(f"[bold]{self._get_prompt()} {rich_escape(text)}[/bold]")
            if text.startswith("cd ") or text == "cd":
                self.run_worker(self._run_local_cd(text), exclusive=True)
            else:
                self.run_worker(self._run_local_cmd(text), exclusive=True)
        else:
            self.log(f"[bold]{self._get_prompt()} {rich_escape(text)}[/bold]")
            if text.startswith("cd ") or text == "cd":
                self.run_worker(self._run_remote_cd(target, text), exclusive=True)
            else:
                self.run_worker(self._run_remote_cmd(target, text), exclusive=True)

    # -- conda activate --------------------------------------------------------

    async def _activate_env(self, env_name: str) -> None:
        """refresh env list then activate if the env exists."""
        if self.is_remote:
            profile = get_profile_by_name(self.get_target())
            if profile is None:
                return
            proc = await ssh_run_async(profile, wrap_remote("conda env list"))
            output = ""
            assert proc.stdout is not None
            async for line in proc.stdout:
                output += line.decode()
            await proc.wait()
            envs = _parse_conda_envs(output)
        else:
            envs = _list_conda_envs_sync()
        if env_name in envs:
            self._active_conda_env = env_name
            self.log(f"[green]Activated conda env: {env_name}[/green]")
            self._update_prompt()
        else:
            self.log(f"[red]Env '{env_name}' not found.[/red]")

    # -- local commands --------------------------------------------------------

    async def _run_local_cd(self, cmd: str) -> None:
        """handle cd locally by resolving the new path."""
        target_dir = cmd[3:].strip() if cmd.startswith("cd ") else os.path.expanduser("~")
        if not target_dir:
            target_dir = os.path.expanduser("~")
        if not os.path.isabs(target_dir):
            target_dir = os.path.join(self._local_cwd, target_dir)
        resolved = os.path.realpath(target_dir)
        if os.path.isdir(resolved):
            self._local_cwd = resolved
            self.log(f"[green]→ {resolved}[/green]")
            self._update_prompt()
        else:
            self.log(f"[red]cd: no such directory: {target_dir}[/red]")

    async def _run_local_cmd(self, cmd: str) -> None:
        wrapped = wrap_local(cmd, self._active_conda_env)
        proc = await asyncio.create_subprocess_shell(
            wrapped,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self._local_cwd,
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
                    self.log(rich_escape(line))
        rc = await proc.wait()
        self._running_proc = None
        if rc != 0:
            self.log(f"[red](exit {rc})[/red]")
        if any(cmd.startswith(p) for p in ("conda create", "conda remove", "conda env remove", "conda env create")):
            self.post_message(self.TargetChanged(self.get_target()))

    # -- remote commands -------------------------------------------------------

    async def _run_remote_cd(self, profile_name: str, cmd: str) -> None:
        """handle cd on remote by resolving the new path via pwd."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cwd = self._remote_cwd or profile.remote_project_path
        resolve_cmd = wrap_remote(f"cd {cwd} && {cmd} && pwd", self._active_conda_env)
        proc = await ssh_run_async(profile, resolve_cmd)
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode()
        rc = await proc.wait()
        if rc == 0:
            new_cwd = output.strip().splitlines()[-1] if output.strip() else cwd
            self._remote_cwd = new_cwd
            self.log(f"[green]→ {new_cwd}[/green]")
            self._update_prompt()
        else:
            self.log(f"[red]{output.strip()}[/red]")

    async def _run_remote_cmd(self, profile_name: str, cmd: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        cwd = self._remote_cwd or profile.remote_project_path
        wrapped = wrap_remote(f"cd {cwd} && {cmd}", self._active_conda_env)
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
                    self.log(rich_escape(line))
        rc = await proc.wait()
        self._running_proc = None
        if rc != 0:
            self.log(f"[red](exit {rc})[/red]")
        if any(cmd.startswith(p) for p in ("conda create", "conda remove", "conda env remove", "conda env create")):
            self.post_message(self.TargetChanged(self.get_target()))
