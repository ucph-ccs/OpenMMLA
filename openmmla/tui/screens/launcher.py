from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static

from openmmla.tui.schema.loader import _find_project_root
from openmmla.tui.ssh import load_ssh_profiles, get_profile_by_name, ssh_run_sync
from openmmla.tui.widgets.command_session import CommandSession
from openmmla.tui.widgets.service_card import ServiceCard, ServiceDef, ParamDef, ComponentDef


def _build_service_registry(root: str) -> list[ServiceDef]:
    """build the registry of all launchable services."""
    services = []

    services.append(ServiceDef(
        name="ASR Base",
        category="Base Stations",
        conda_env="asr-base",
        config_dir=os.path.join(root, "base_stations", "asr"),
        launch_type="bash",
        description="Real-time audio analysis base stations and synchronizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 3),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-s", "Store Audio", "bool", True),
            ParamDef("-vad", "VAD", "bool", True),
            ParamDef("-nr", "Noise Reduce", "bool", True),
            ParamDef("-tr", "Transcribe", "bool", True),
            ParamDef("-sp", "Speech Separate", "bool", False),
            ParamDef("-d", "Dominant Speaker", "bool", False),
            ParamDef("-hsr", "Half-Scaled Recognition", "bool", True),
        ],
        components=[
            ComponentDef("base", "examples/run_asr_base.py", "-nb",
                         ["-s", "-vad", "-nr", "-tr", "-sp", "-hsr"]),
            ComponentDef("synchronizer", "examples/run_asr_synchronizer.py", "-ns",
                         ["-d", "-sp"]),
        ],
    ))

    services.append(ServiceDef(
        name="VFA Base",
        category="Base Stations",
        conda_env="vfa-base",
        config_dir=os.path.join(root, "base_stations", "vfa"),
        launch_type="bash",
        description="Video frame analysis base stations and synchronizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "examples/run_vfa_base.py", "-nb",
                         ["-g", "-v"]),
            ComponentDef("synchronizer", "examples/run_vfa_synchronizer.py", "-ns"),
        ],
    ))

    services.append(ServiceDef(
        name="IPS Base",
        category="Base Stations",
        conda_env="ips-base",
        config_dir=os.path.join(root, "base_stations", "ips"),
        launch_type="bash",
        description="Indoor positioning system base stations, synchronizer, and visualizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-nv", "Num Visualizers", "int", 1),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-s", "Store", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "examples/run_ips_base.py", "-nb",
                         ["-g", "-s", "-v"]),
            ComponentDef("synchronizer", "examples/run_ips_synchronizer.py", "-ns",
                         ["-v"]),
            ComponentDef("visualizer", "examples/run_ips_visualizer.py", "-nv",
                         ["-s"]),
        ],
    ))

    services.append(ServiceDef(
        name="ASR Server",
        category="Servers",
        conda_env="asr-server",
        config_dir=os.path.join(root, "servers", "asr"),
        launch_type="tmux",
        description="ASR inference services (inferer, resampler, enhancer, transcriber, ...)",
    ))

    services.append(ServiceDef(
        name="VFA Server",
        category="Servers",
        conda_env="vfa-server",
        config_dir=os.path.join(root, "servers", "vfa"),
        launch_type="tmux",
        description="VFA inference services (VLLM frame analyzer, ...)",
    ))

    uber_dir = os.path.join(root, "servers", "uber")
    if os.path.isdir(uber_dir):
        for svc_name, desc in [
            ("InfluxDB", "Time series database"),
            ("Redis", "In-memory data store and message broker"),
            ("Mosquitto", "MQTT message broker"),
            ("Nginx", "Reverse proxy and load balancer"),
            ("Flask", "Dashboard backend API"),
            ("Next.js", "Dashboard frontend"),
            ("Celery", "Async task worker"),
        ]:
            services.append(ServiceDef(
                name=f"Uber: {svc_name}",
                category="Uber Server",
                conda_env="uber-server",
                config_dir=uber_dir,
                launch_type="make",
                description=desc,
            ))

    return services


def _check_tmux_session(session_name: str) -> bool:
    """check if a tmux session with the given name exists."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_config_exists(config_dir: str) -> bool:
    return os.path.isfile(os.path.join(config_dir, "config.yml"))


def _check_conda_env(env_name: str) -> bool:
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith(env_name + " ") or line.strip().startswith(env_name + "\t"):
                return True
            parts = line.strip().split()
            if parts and parts[0] == env_name:
                return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class LauncherPanel(Widget):

    DEFAULT_CSS = """
    LauncherPanel {
        width: 1fr;
        height: 1fr;
    }
    #launcher-scroll {
        width: 1fr;
        height: 1fr;
    }
    .category-header {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        padding: 0 1;
        background: $panel;
    }
    .preflight-warning {
        color: $warning;
        padding: 0 2;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._services = _build_service_registry(self._root)
        self._svc_states: dict[str, bool] = {}
        self._ssh_profile_names: list[str] = [
            p.name for p in load_ssh_profiles()
        ]

    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll(id="launcher-scroll"):
                current_cat = None
                for svc in self._services:
                    if svc.category != current_cat:
                        current_cat = svc.category
                        yield Static(f" {current_cat}", classes="category-header")
                    is_running = self._detect_running(svc)
                    self._svc_states[svc.name] = is_running
                    yield ServiceCard(
                        svc,
                        is_running=is_running,
                        ssh_profile_names=self._ssh_profile_names,
                    )
            yield CommandSession(id="launch-cmd-session")

    def on_show(self) -> None:
        self._ssh_profile_names = [p.name for p in load_ssh_profiles()]
        for card in self.query(ServiceCard):
            card.refresh_targets(self._ssh_profile_names)

    def _detect_running(self, svc: ServiceDef) -> bool:
        if svc.launch_type == "tmux":
            session_name = svc.name.lower().replace(" ", "-")
            if _check_tmux_session(session_name):
                return True
            if "ASR" in svc.name:
                return _check_tmux_session("asr-services")
            if "VFA" in svc.name:
                return _check_tmux_session("vfa-services")
        elif svc.launch_type == "make":
            short = svc.name.replace("Uber: ", "").lower().replace(".", "")
            return _check_tmux_session(short)
        return False

    def _log(self, message: str) -> None:
        try:
            self.query_one("#launch-cmd-session", CommandSession).log(message)
        except Exception:
            pass

    def on_service_card_start_requested(self, event: ServiceCard.StartRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = event.target
        is_remote = target != "local"

        if not is_remote and not _check_config_exists(svc.config_dir):
            self._log(f"[yellow]WARNING: config.yml not found in {svc.config_dir}[/yellow]")
            self._log("[yellow]Please configure this pipeline first in the Config tab.[/yellow]")
            return

        if not is_remote:
            has_conda = shutil.which("conda") is not None
            if has_conda and not _check_conda_env(svc.conda_env):
                self._log(f"[yellow]WARNING: conda env '{svc.conda_env}' not found.[/yellow]")

        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[green]Starting {svc.name} {target_label}...[/green]")

        if is_remote:
            self._launch_remote(svc, event.params, target)
        else:
            self._launch_service(svc, event.params)

    def on_service_card_stop_requested(self, event: ServiceCard.StopRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = event.target
        is_remote = target != "local"
        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[red]Stopping {svc.name} {target_label}...[/red]")

        if is_remote:
            self._stop_remote(svc, target)
        else:
            self._stop_service(svc)

    def _launch_service(self, svc: ServiceDef, params: dict) -> None:
        try:
            if svc.launch_type == "bash":
                self._launch_bash(svc, params)
            elif svc.launch_type == "tmux":
                self._launch_tmux_server(svc)
            elif svc.launch_type == "make":
                self._launch_make(svc)
        except Exception as e:
            self._log(f"[red]Error launching {svc.name}: {e}[/red]")

    def _launch_bash(self, svc: ServiceDef, params: dict) -> None:
        if not svc.components:
            self._log(f"[red]No components defined for {svc.name}[/red]")
            return

        python_path = self._root
        conda_env = svc.conda_env
        preamble = (
            f"export PYTHONPATH={python_path}/:$PYTHONPATH && "
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {conda_env}"
        )

        tab_cmds: list[tuple[str, str]] = []
        for comp in svc.components:
            count = params.get(comp.count_flag, 0)
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 0
            if count <= 0:
                continue

            flag_parts = []
            for f in comp.flags:
                val = params.get(f)
                if val is None:
                    continue
                if isinstance(val, bool):
                    flag_parts.append(f"{f} {'true' if val else 'false'}")
                else:
                    flag_parts.append(f"{f} {val}")
            flag_str = " ".join(flag_parts)

            script_path = os.path.join(svc.config_dir, comp.script)
            py_cmd = f"python3 {script_path}"
            if flag_str:
                py_cmd += f" {flag_str}"

            full_cmd = f"{preamble} && {py_cmd}"

            for i in range(count):
                label = f"{comp.role} {i + 1}" if count > 1 else comp.role
                tab_cmds.append((label, full_cmd))

        if not tab_cmds:
            self._log("[yellow]No components to launch (all counts are 0).[/yellow]")
            return

        self._log(f"  Launching {len(tab_cmds)} tab(s)...")
        for label, cmd in tab_cmds:
            self._log(f"    [{label}] {cmd.split(' && ')[-1]}")

        if sys.platform == "darwin":
            self._open_tabs_mac(tab_cmds)
        elif self._is_ubuntu():
            self._open_tabs_gnome(tab_cmds)
        elif self._is_raspberry_pi():
            self._open_tabs_lxterminal(tab_cmds)
        else:
            self._log("[yellow]Unsupported OS for terminal tab launch.[/yellow]")
            return

        self._log(f"[green]{svc.name} launched in new terminal window.[/green]")

    def _open_tabs_mac(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one Terminal.app window with N tabs on macOS."""
        script_lines = []
        _, first_cmd = tab_cmds[0]
        escaped = first_cmd.replace("\\", "\\\\").replace('"', '\\"')
        script_lines.append(f'tell application "Terminal" to do script "{escaped}"')
        script_lines.append('tell application "Terminal" to activate')

        for _, cmd in tab_cmds[1:]:
            escaped = cmd.replace("\\", "\\\\").replace('"', '\\"')
            script_lines.append("delay 0.5")
            script_lines.append(
                'tell application "System Events" to keystroke "t" using command down'
            )
            script_lines.append("delay 0.3")
            script_lines.append(
                f'tell application "Terminal" to do script "{escaped}" in the front window'
            )

        args = ["osascript"]
        for line in script_lines:
            args.extend(["-e", line])
        subprocess.Popen(args)

    def _open_tabs_gnome(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one gnome-terminal window with N tabs."""
        args = ["gnome-terminal", "--window"]
        for _, cmd in tab_cmds:
            args.extend(["--tab", "--", "bash", "-c", f"{cmd}; exec bash"])
        subprocess.Popen(args)

    def _open_tabs_lxterminal(self, tab_cmds: list[tuple[str, str]]) -> None:
        """open one lxterminal window per component (no multi-tab support)."""
        for _, cmd in tab_cmds:
            subprocess.Popen([
                "lxterminal",
                f'--command=bash -c "{cmd}; exec bash"',
            ])

    @staticmethod
    def _is_ubuntu() -> bool:
        try:
            with open("/etc/os-release") as f:
                return "ID=ubuntu" in f.read()
        except FileNotFoundError:
            return False

    @staticmethod
    def _is_raspberry_pi() -> bool:
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            return "ID=debian" in os_release and "Raspberry Pi" in cpuinfo
        except FileNotFoundError:
            return False

    def _launch_tmux_server(self, svc: ServiceDef) -> None:
        bash_dir = os.path.join(svc.config_dir, "bash")
        services_script = os.path.join(bash_dir, "services.sh")
        if not os.path.isfile(services_script):
            self._log(f"[red]services.sh not found: {services_script}[/red]")
            return

        session_name = svc.name.lower().replace(" ", "-")
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
        )
        subprocess.Popen(
            ["tmux", "new-session", "-d", "-s", session_name,
             f"bash -c 'cd {bash_dir} && bash services.sh; exec bash'"],
            cwd=bash_dir,
        )
        self._log(f"[green]{svc.name} tmux session '{session_name}' started.[/green]")

    def _launch_make(self, svc: ServiceDef) -> None:
        target = svc.name.replace("Uber: ", "").lower().replace(".", "")
        make_dir = svc.config_dir
        makefile = os.path.join(make_dir, "Makefile")
        if not os.path.isfile(makefile):
            self._log(f"[red]Makefile not found: {makefile}[/red]")
            return

        self._log(f"  Running: make {target}")
        subprocess.Popen(
            ["make", target],
            cwd=make_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._log(f"[green]Uber {target} started.[/green]")

    def _stop_service(self, svc: ServiceDef) -> None:
        try:
            if svc.launch_type == "tmux":
                session_name = svc.name.lower().replace(" ", "-")
                subprocess.run(["tmux", "send-keys", "-t", session_name, "C-c"], capture_output=True)
                subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
                self._log(f"[red]{svc.name} stopped.[/red]")
            elif svc.launch_type == "make":
                target = "stop-" + svc.name.replace("Uber: ", "").lower().replace(".", "")
                subprocess.Popen(
                    ["make", target],
                    cwd=svc.config_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._log(f"[red]{svc.name} stop initiated.[/red]")
            elif svc.launch_type == "bash":
                self._log(f"[yellow]Bash-launched services must be stopped from their terminal windows.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {svc.name}: {e}[/red]")

    def _launch_remote(self, svc: ServiceDef, params: dict, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        remote_root = profile.remote_project_path
        try:
            if svc.launch_type == "bash":
                rel_dir = os.path.relpath(svc.config_dir, self._root)
                remote_bash = f"{remote_root}/{rel_dir}/bash"
                flag_str = ""
                for flag, value in params.items():
                    if isinstance(value, bool):
                        flag_str += f" {flag} {'true' if value else 'false'}"
                    else:
                        flag_str += f" {flag} {value}"
                cmd = f"cd {remote_bash} && bash run.sh{flag_str}"
                self._log(f"  Remote: {cmd}")
                result = ssh_run_sync(profile, f"nohup bash -c '{cmd}' >/dev/null 2>&1 &", timeout=15.0)
                if result.returncode == 0:
                    self._log(f"[green]{svc.name} launched remotely.[/green]")
                else:
                    self._log(f"[red]Remote launch failed: {result.stderr.strip()}[/red]")

            elif svc.launch_type == "tmux":
                rel_dir = os.path.relpath(svc.config_dir, self._root)
                remote_bash = f"{remote_root}/{rel_dir}/bash"
                session_name = svc.name.lower().replace(" ", "-")
                cmd = (
                    f"tmux kill-session -t {session_name} 2>/dev/null; "
                    f"tmux new-session -d -s {session_name} "
                    f"'cd {remote_bash} && bash services.sh; exec bash'"
                )
                self._log(f"  Remote: {cmd}")
                result = ssh_run_sync(profile, cmd, timeout=15.0)
                if result.returncode == 0:
                    self._log(f"[green]{svc.name} tmux session started remotely.[/green]")
                else:
                    self._log(f"[red]Remote tmux launch failed: {result.stderr.strip()}[/red]")

            elif svc.launch_type == "make":
                target = svc.name.replace("Uber: ", "").lower().replace(".", "")
                remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                cmd = f"cd {remote_dir} && make {target}"
                self._log(f"  Remote: {cmd}")
                result = ssh_run_sync(profile, cmd, timeout=30.0)
                if result.returncode == 0:
                    self._log(f"[green]Uber {target} started remotely.[/green]")
                else:
                    self._log(f"[red]Remote make failed: {result.stderr.strip()}[/red]")
        except Exception as e:
            self._log(f"[red]Remote launch error: {e}[/red]")

    def _stop_remote(self, svc: ServiceDef, profile_name: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._log(f"[red]SSH profile '{profile_name}' not found.[/red]")
            return
        remote_root = profile.remote_project_path
        try:
            if svc.launch_type == "tmux":
                session_name = svc.name.lower().replace(" ", "-")
                cmd = (
                    f"tmux send-keys -t {session_name} C-c; "
                    f"tmux kill-session -t {session_name}"
                )
                result = ssh_run_sync(profile, cmd, timeout=10.0)
                self._log(f"[red]{svc.name} stopped remotely.[/red]")

            elif svc.launch_type == "make":
                target = "stop-" + svc.name.replace("Uber: ", "").lower().replace(".", "")
                remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                cmd = f"cd {remote_dir} && make {target}"
                result = ssh_run_sync(profile, cmd, timeout=15.0)
                self._log(f"[red]{svc.name} stop initiated remotely.[/red]")

            elif svc.launch_type == "bash":
                self._log(
                    f"[yellow]Remote bash services must be stopped "
                    f"from the remote terminal.[/yellow]"
                )
        except Exception as e:
            self._log(f"[red]Remote stop error: {e}[/red]")
