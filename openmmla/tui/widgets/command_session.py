from __future__ import annotations

import asyncio
import os
import subprocess

from rich.markup import escape as rich_escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import RichLog, Label, Input, Select, Static

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


class LogResizeHandle(Static):
    """thin grab bar above the command log; drag or scroll to resize the log."""

    DEFAULT_CSS = """
    LogResizeHandle {
        height: 1;
        color: $text-muted;
        text-align: center;
        border-top: solid $primary;
    }
    LogResizeHandle:hover {
        background: $primary 20%;
        color: $text;
    }
    """

    def __init__(self, session: "CommandSession") -> None:
        super().__init__("· · ·  ⇕ drag or scroll to resize log  · · ·")
        self._session = session
        self._drag_start_y: int | None = None
        self._drag_start_height: int = 0

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.capture_mouse()
        self._drag_start_y = event.screen_y
        self._drag_start_height = self._session.log_height
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._drag_start_y is None:
            return
        # dragging the handle up grows the log, down shrinks it
        delta = self._drag_start_y - event.screen_y
        self._session.set_log_height(self._drag_start_height + delta)
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.release_mouse()
        self._drag_start_y = None
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._session.set_log_height(self._session.log_height + 1)
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._session.set_log_height(self._session.log_height - 1)
        event.stop()


class CommandSession(Widget):
    """reusable interactive command session with target selector, log, and input."""

    MIN_LOG_HEIGHT = 2
    MAX_LOG_HEIGHT = 40

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
        self._log_height: int = 4

    @property
    def log_height(self) -> int:
        return self._log_height

    def set_log_height(self, height: int) -> None:
        """resize the log area, clamped to sane bounds."""
        height = max(self.MIN_LOG_HEIGHT, min(self.MAX_LOG_HEIGHT, height))
        if height == self._log_height:
            return
        self._log_height = height
        try:
            log = self.query_one(".cmd-log", RichLog)
            log.styles.height = height
            log.scroll_end(animate=False)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        target_options = [("Local", "local")] + [
            (p.name, p.name) for p in load_ssh_profiles()
        ]
        if self._show_target:
            with Horizontal(classes="cmd-target-bar"):
                yield Label("Host:")
                yield Select(target_options, value="local", id="cmd-target-select")
        yield LogResizeHandle(self)
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
                from openmmla.tui.ssh import is_select_sentinel
                if is_select_sentinel(val):
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
        self.run_worker(self._async_connect(profile), exclusive=True)

    async def _async_connect(self, profile) -> None:
        """run the SSH connectivity test off the UI thread."""
        success, msg = await asyncio.to_thread(ssh_test_connection, profile)
        if success:
            self._connected = True
            self.log(f"[green]Connected to {profile.name} ({profile.ssh_destination()}) — {msg}[/green]")
            await self._resolve_remote_cwd(profile)
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

    # -- sudo password auto-fill -------------------------------------------------

    MAX_AUTO_PASSWORD_SENDS = 3

    @staticmethod
    def _looks_like_password_prompt(text: str) -> bool:
        """detect a sudo/ssh password prompt at the end of an output chunk."""
        tail = text.rstrip().lower()
        return tail.endswith("password:") or "password for" in tail

    def _stored_sudo_password(self) -> str | None:
        """local sudo password from the System Services store (None if unset)."""
        try:
            from openmmla.tui.system_services import get_sudo_password
            return get_sudo_password(self._root)
        except Exception:
            return None

    async def _maybe_send_password(self, proc, chunk_text: str, password: str | None,
                                   sends: int) -> int:
        """auto-fill a password prompt; returns the updated send count."""
        if (
            password
            and sends < self.MAX_AUTO_PASSWORD_SENDS
            and proc.stdin is not None
            and self._looks_like_password_prompt(chunk_text)
        ):
            proc.stdin.write((password + "\n").encode())
            await proc.stdin.drain()
            self.log("[dim]> (password auto-filled from System Services)[/dim]")
            return sends + 1
        return sends

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
        sudo_password = self._stored_sudo_password() if "sudo" in cmd else None
        pw_sends = 0
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            raw = chunk.decode(errors="replace")
            text = raw.rstrip()
            if text:
                for line in text.splitlines():
                    self.log(rich_escape(line))
            pw_sends = await self._maybe_send_password(proc, raw, sudo_password, pw_sends)
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
        # remote sudo prompts are answered with the SSH profile's password
        remote_password = (getattr(profile, "password", "") or None) if "sudo" in cmd else None
        pw_sends = 0
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            raw = chunk.decode(errors="replace")
            text = raw.rstrip()
            if text:
                for line in text.splitlines():
                    self.log(rich_escape(line))
            pw_sends = await self._maybe_send_password(proc, raw, remote_password, pw_sends)
        rc = await proc.wait()
        self._running_proc = None
        if rc != 0:
            self.log(f"[red](exit {rc})[/red]")
        if any(cmd.startswith(p) for p in ("conda create", "conda remove", "conda env remove", "conda env create")):
            self.post_message(self.TargetChanged(self.get_target()))
