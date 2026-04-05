from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import (
    Static, Tree, Button, Select, Input, Label, TabbedContent, TabPane,
)

from openmmla.tui.schema.loader import (
    FieldDef as LoaderFieldDef,
    discover_pipelines, load_existing_config, get_nested_value,
    save_config, PipelineDef, _find_project_root, fields_from_config_section,
)
from openmmla.tui.schema.definitions import (
    SHARED_SECTIONS, get_shared_defaults, apply_shared_values,
)
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, ssh_run_sync,
    scp_file_async, ssh_run_async, ssh_check_port, ssh_check_tmux,
)
from openmmla.tui.widgets.command_session import CommandSession
from openmmla.tui.widgets.config_form import ConfigForm
from openmmla.tui.widgets.experiment_form import ExperimentForm
from openmmla.tui.widgets.service_card import ServiceCard, ServiceDef, ParamDef, ComponentDef
from openmmla.tui.widgets.ssh_form import SSHForm
from openmmla.tui.widgets.task_form import TaskForm


_SVC_PIPELINE_NAMES: dict[str, str] = {
    "Uber: Nginx": "Nginx",
    "Uber: Flask": "Flask Backend",
}


def _build_service_registry(root: str) -> list[ServiceDef]:
    """build the registry of all launchable services."""
    services = []

    services.append(ServiceDef(
        name="ASR Base",
        category="ASR",
        conda_env="asr-base",
        config_dir=os.path.join(root, "pipelines", "asr-base"),
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
            ComponentDef("base", "mmla asr-base", "-nb",
                         ["-s", "-vad", "-nr", "-tr", "-sp", "-hsr"]),
            ComponentDef("synchronizer", "mmla asr-sync", "-ns",
                         ["-d", "-sp"]),
        ],
    ))

    services.append(ServiceDef(
        name="VFA Base",
        category="VFA",
        conda_env="vfa-base",
        config_dir=os.path.join(root, "pipelines", "vfa-base"),
        launch_type="bash",
        description="Video frame analysis base stations and synchronizer",
        params=[
            ParamDef("-nb", "Num Bases", "int", 1),
            ParamDef("-ns", "Num Synchronizers", "int", 1),
            ParamDef("-g", "Graphics", "bool", True),
            ParamDef("-v", "Verbose", "bool", True),
        ],
        components=[
            ComponentDef("base", "mmla vfa-base", "-nb",
                         ["-g", "-v"]),
            ComponentDef("synchronizer", "mmla vfa-sync", "-ns"),
        ],
    ))

    services.append(ServiceDef(
        name="IPS Base",
        category="IPS",
        conda_env="ips-base",
        config_dir=os.path.join(root, "pipelines", "ips-base"),
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
            ComponentDef("base", "mmla ips-base", "-nb",
                         ["-g", "-s", "-v"]),
            ComponentDef("synchronizer", "mmla ips-sync", "-ns",
                         ["-v"]),
            ComponentDef("visualizer", "mmla ips-vis", "-nv",
                         ["-s"]),
        ],
    ))

    services.append(ServiceDef(
        name="ASR Server",
        category="ASR",
        conda_env="asr-server",
        config_dir=os.path.join(root, "pipelines", "asr-server"),
        launch_type="tmux",
        description="ASR inference services (inferer, resampler, enhancer, transcriber, ...)",
    ))

    services.append(ServiceDef(
        name="VFA Server",
        category="VFA",
        conda_env="vfa-server",
        config_dir=os.path.join(root, "pipelines", "vfa-server"),
        launch_type="tmux",
        description="VFA inference services (VLLM frame analyzer, ...)",
    ))

    uber_dir = os.path.join(root, "pipelines", "uber-server")
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
                category="Infrastructure",
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


def _capture_tmux_pane(session_name: str, lines: int = 80) -> str:
    """capture recent output from a tmux session."""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-p", "-S", f"-{lines}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout
        return f"(could not capture pane for session '{session_name}')"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "(tmux not available or session not found)"


def _get_system_service_log(service_name: str, lines: int = 80) -> str:
    """get log output for a brew-managed system service."""
    try:
        prefix = subprocess.run(
            ["brew", "--prefix"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        prefix = "/opt/homebrew"

    log_paths: dict[str, list[str]] = {
        "influxdb": [f"{prefix}/var/log/influxdb2/influxd_output.log"],
        "redis": [f"{prefix}/var/log/redis.log"],
        "nginx": [
            f"{prefix}/var/log/nginx/error.log",
            f"{prefix}/var/log/nginx/access.log",
        ],
        "mosquitto": [f"{prefix}/var/log/mosquitto/mosquitto.log"],
    }

    paths = log_paths.get(service_name, [])
    for path in paths:
        if os.path.isfile(path):
            try:
                result = subprocess.run(
                    ["tail", "-n", str(lines), path],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return f"[log file: {path}]\n{result.stdout}"
            except Exception:
                continue

    try:
        proc_name = "influxd" if service_name == "influxdb" else service_name
        result = subprocess.run(
            ["log", "show", "--predicate", f'process == "{proc_name}"',
             "--last", "5m", "--style", "compact"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return f"[macOS system log]\n{result.stdout}"
    except Exception:
        pass

    return f"(no log file found for {service_name})"


def _check_config_exists(config_dir: str) -> bool:
    return os.path.isfile(os.path.join(config_dir, "config.yml"))


_SYSTEM_SVC_PORTS: dict[str, int] = {
    "influxdb": 8086,
    "redis": 6379,
    "mosquitto": 1883,
    "nginx": 8080,
}

_MAKE_TARGET_OVERRIDES: dict[str, str] = {
    "Uber: Next.js": "next",
}

# (port, expected_command) for app services that need port cleanup on stop
_APP_PORT_CMDS: dict[str, tuple[int, str]] = {
    "flask": (5050, "gunicorn"),
    "next": (3000, "node"),
}


def _make_target_for(svc_name: str) -> str:
    """derive the Makefile target name from a service name."""
    if svc_name in _MAKE_TARGET_OVERRIDES:
        return _MAKE_TARGET_OVERRIDES[svc_name]
    return svc_name.replace("Uber: ", "").lower().replace(".", "")


def _check_port_in_use(port: int) -> bool:
    """check if a TCP port is in use on localhost."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


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


class ServicePanel(Widget):

    DEFAULT_CSS = """
    ServicePanel {
        layout: horizontal;
        width: 1fr;
        height: 1fr;
    }
    #svc-sidebar {
        width: 32;
        border-right: solid $primary;
        padding: 1;
        background: $panel;
    }
    #svc-sidebar Tree {
        width: 1fr;
        scrollbar-size: 1 1;
    }
    #svc-main {
        width: 1fr;
        height: 1fr;
    }
    #svc-target-bar {
        layout: horizontal;
        height: auto;
        padding: 0 1;
    }
    #svc-target-bar Label {
        width: 10;
        padding-top: 1;
    }
    #svc-target-bar Select {
        width: 1fr;
    }
    #svc-content-area {
        width: 1fr;
        height: 1fr;
    }
    #svc-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
        text-style: italic;
    }
    .svc-config-scroll {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    .svc-launch-scroll {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    #svc-sub-tabs {
        width: 1fr;
        height: 1fr;
    }
    .sync-bar {
        layout: horizontal;
        height: auto;
        padding: 1 0;
        margin-top: 1;
    }
    .sync-bar Select {
        width: 1fr;
    }
    .sync-bar Button {
        margin: 0 1;
        min-width: 18;
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
        self._svc_map: dict[str, ServiceDef] = {s.name: s for s in self._services}
        self._svc_states: dict[str, bool] = {}
        self._pipelines: list[PipelineDef] = []
        self._pipeline_map: dict[str, PipelineDef] = {}
        self._shared_values: dict[str, object] = get_shared_defaults()
        self._current_pipeline: PipelineDef | None = None
        self._current_form: ConfigForm | None = None
        self._config_container: VerticalScroll | None = None
        self._ssh_profile_names: list[str] = [
            p.name for p in load_ssh_profiles()
        ]

    def compose(self) -> ComposeResult:
        target_options = [("Local", "local")] + [
            (name, name) for name in self._ssh_profile_names
        ]
        with Vertical(id="svc-sidebar"):
            yield Static("[b]Launcher[/b]", classes="status-info")
            tree: Tree[str] = Tree("OpenMMLA", id="svc-tree")
            tree.root.expand()
            yield tree
        with Vertical(id="svc-main"):
            with Horizontal(id="svc-target-bar"):
                yield Label("Target:")
                yield Select(target_options, value="local", id="svc-target-select")
            with Vertical(id="svc-content-area"):
                yield Static(
                    "Select a service from the sidebar.",
                    id="svc-empty",
                )
            yield CommandSession(show_target=False, id="svc-cmd-session")

    def on_mount(self) -> None:
        self._pipelines = discover_pipelines()
        self._pipeline_map = {p.name: p for p in self._pipelines}
        self._build_tree()

    def on_show(self) -> None:
        self._ssh_profile_names = [p.name for p in load_ssh_profiles()]
        try:
            sel = self.query_one("#svc-target-select", Select)
            options = [("Local", "local")] + [
                (name, name) for name in self._ssh_profile_names
            ]
            current = sel.value
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
        except Exception:
            pass

    def _get_panel_target(self) -> str:
        """get the current target from the unified selector."""
        try:
            sel = self.query_one("#svc-target-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                return "local"
            return str(val)
        except Exception:
            return "local"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "svc-target-select":
            val = event.value
            if val is Select.BLANK or val is None:
                val = "local"
            cmd = self.query_one("#svc-cmd-session", CommandSession)
            cmd.set_target(str(val))

    def _pipeline_for_service(self, svc_name: str) -> PipelineDef | None:
        """find the PipelineDef associated with a service, if any."""
        pipeline_name = _SVC_PIPELINE_NAMES.get(svc_name, svc_name)
        return self._pipeline_map.get(pipeline_name)

    # ── sidebar tree ─────────────────────────────────────────────

    def _build_tree(self) -> None:
        tree = self.query_one("#svc-tree", Tree)
        tree.clear()

        shared_node = tree.root.add("Global Defaults", data="__shared__")
        shared_node.expand()
        for sec in SHARED_SECTIONS:
            shared_node.add_leaf(sec, data=f"__shared__{sec}")
        shared_node.add_leaf("Experiments", data="__experiments__")
        shared_node.add_leaf("Tasks", data="__tasks__")
        shared_node.add_leaf("SSH Profiles", data="__ssh_profiles__")

        categories: dict[str, list[ServiceDef]] = {}
        for svc in self._services:
            categories.setdefault(svc.category, []).append(svc)

        _PIPELINE_CATS = ["ASR", "VFA", "IPS"]

        pipeline_node = tree.root.add("Pipelines", data="__cat_Pipelines")
        pipeline_node.expand()
        for cat in _PIPELINE_CATS:
            svcs = categories.get(cat, [])
            if not svcs:
                continue
            cat_node = pipeline_node.add(cat, data=f"__cat_{cat}")
            cat_node.expand()
            for svc in svcs:
                cat_node.add_leaf(f"{svc.name}{self._svc_markers(svc)}", data=svc.name)

        infra_svcs = categories.get("Infrastructure", [])
        if infra_svcs:
            infra_node = tree.root.add("Infrastructure", data="__cat_Infrastructure")
            infra_node.expand()
            for svc in infra_svcs:
                infra_node.add_leaf(f"{svc.name}{self._svc_markers(svc)}", data=svc.name)

    def _svc_markers(self, svc: ServiceDef) -> str:
        """build status marker string for a service tree leaf."""
        is_running = self._detect_running(svc)
        self._svc_states[svc.name] = is_running
        pipeline = self._pipeline_for_service(svc.name)
        markers = ""
        if pipeline and os.path.isfile(pipeline.config_path):
            markers += " [green]\\[OK][/green]"
        if is_running:
            markers += " [green](R)[/green]"
        return markers

    # ── tree node selection ──────────────────────────────────────

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = event.node.data
        if node_data is None:
            return
        node_str = str(node_data)

        if node_str.startswith("__cat_") or node_str == "__shared__":
            return

        content_area = self.query_one("#svc-content-area", Vertical)
        await content_area.remove_children()
        self._current_pipeline = None
        self._current_form = None
        self._config_container = None

        if node_str.startswith("__shared__"):
            section_name = node_str.replace("__shared__", "")
            scroll = VerticalScroll(classes="svc-config-scroll")
            await content_area.mount(scroll)
            self._config_container = scroll
            self._show_shared_form(scroll, section_name)
            return

        if node_str == "__experiments__":
            await content_area.mount(ExperimentForm())
            return

        if node_str == "__tasks__":
            await content_area.mount(TaskForm())
            return

        if node_str == "__ssh_profiles__":
            scroll = VerticalScroll(classes="svc-config-scroll")
            await content_area.mount(scroll)
            await scroll.mount(SSHForm())
            return

        svc = self._svc_map.get(node_str)
        if svc is None:
            return

        pipeline = self._pipeline_for_service(svc.name)
        is_running = self._svc_states.get(svc.name, False)

        if pipeline:
            self._current_pipeline = pipeline
            tabs = TabbedContent(id="svc-sub-tabs")
            await content_area.mount(tabs)

            launch_scroll = VerticalScroll(classes="svc-launch-scroll")
            card = ServiceCard(
                svc,
                is_running=is_running,
            )
            launch_pane = TabPane("Launch", launch_scroll, id="svc-tab-launch")
            await tabs.add_pane(launch_pane)
            await launch_scroll.mount(card)

            config_scroll = VerticalScroll(classes="svc-config-scroll")
            config_pane = TabPane("Config", config_scroll, id="svc-tab-config")
            await tabs.add_pane(config_pane)
            self._config_container = config_scroll
            self._show_pipeline_form(config_scroll, pipeline)
        else:
            scroll = VerticalScroll(classes="svc-launch-scroll")
            await content_area.mount(scroll)
            await scroll.mount(ServiceCard(
                svc,
                is_running=is_running,
            ))

    # ── config logic ─────────────────────────────────────────────

    def _show_shared_form(self, container: VerticalScroll, section_name: str) -> None:
        sec_info = SHARED_SECTIONS.get(section_name, {})
        fields = []
        for key, fdef in sec_info.get("fields", {}).items():
            fields.append(LoaderFieldDef(
                path=f"{section_name}.{key}",
                field_type=fdef["field_type"],
                default=fdef["default"],
                description=fdef["description"],
                required=True,
                section=section_name,
            ))
        values = {f.path: self._shared_values.get(f.path, f.default) for f in fields}
        form = ConfigForm(f"shared:{section_name}", fields, values)
        container.mount(form)

    def _show_pipeline_form(self, container: VerticalScroll, pipeline: PipelineDef) -> None:
        existing = load_existing_config(pipeline.config_path)
        apply_shared_values(pipeline.fields, self._shared_values)

        values = {}
        for f in pipeline.fields:
            existing_val = get_nested_value(existing, f.path)
            if existing_val is not None:
                values[f.path] = existing_val
            else:
                values[f.path] = self._shared_values.get(f.path, f.default)

        dynamic_sections: dict[str, list[LoaderFieldDef]] = {}
        if pipeline.base_template and pipeline.base_section:
            base_data = existing.get(pipeline.base_section, {})
            if isinstance(base_data, dict):
                for device_name in base_data:
                    section_name = f"{pipeline.base_section}.{device_name}"
                    fields = self._clone_template_fields(pipeline, section_name)
                    dynamic_sections[section_name] = fields
                    for f in fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val

        if pipeline.name == "Nginx":
            existing_upstreams = existing.get("upstreams", {})
            template_names = {
                f.path.split(".", 1)[1]
                for f in pipeline.fields if f.path.startswith("upstreams.")
            }
            for svc_name, entries in existing_upstreams.items():
                if svc_name not in template_names and isinstance(entries, list):
                    values[f"upstreams.{svc_name}"] = entries

        known_sections = {f.path.split(".")[0] for f in pipeline.fields}
        if pipeline.base_section:
            known_sections.add(pipeline.base_section)
        for key, value in existing.items():
            if key not in known_sections and isinstance(value, dict):
                extra_fields = fields_from_config_section(key, value)
                if extra_fields:
                    dynamic_sections[key] = extra_fields
                    for f in extra_fields:
                        val = get_nested_value(existing, f.path)
                        if val is not None:
                            values[f.path] = val

        group_add_buttons = {}
        if pipeline.base_template and pipeline.base_section:
            group_add_buttons[pipeline.base_section] = ("+ Add Base", "btn-add-base")

        form = ConfigForm(pipeline.name, pipeline.fields, values, dynamic_sections,
                          group_add_buttons=group_add_buttons)
        container.mount(form)
        self._current_form = form

    def on_config_form_saved(self, event: ConfigForm.Saved) -> None:
        if event.pipeline_name.startswith("shared:"):
            for path, val in event.values.items():
                self._shared_values[path] = val
            self._show_status("Shared defaults updated")
            return

        pipeline = self._pipeline_map.get(event.pipeline_name)
        if pipeline is None:
            return

        form = self._current_form
        all_fields = form.all_fields if form else pipeline.fields
        save_config(pipeline.config_path, all_fields, event.values)

        self._show_status(f"Saved to {pipeline.config_path}")
        self._show_sync_bar(pipeline)
        self._build_tree()

    def _show_status(self, message: str) -> None:
        container = self._config_container
        if container is None:
            return
        for old in container.query(".status-saved"):
            old.remove()
        container.mount(Static(f" {message}", classes="status-saved"))

    def _show_sync_bar(self, pipeline: PipelineDef) -> None:
        profiles = load_ssh_profiles()
        if not profiles:
            return
        container = self._config_container
        if container is None:
            return
        for old in container.query(".sync-bar"):
            old.remove()
        options = [(p.name, p.name) for p in profiles]
        bar = Horizontal(
            Select(options, prompt="Select SSH profile...", id="sync-profile-select"),
            Button("Sync to Remote", variant="warning", id="btn-sync-remote"),
            classes="sync-bar",
        )
        container.mount(bar)

    def _clone_template_fields(self, pipeline: PipelineDef, new_name: str) -> list[LoaderFieldDef]:
        """clone base_template fields with paths rewritten to a new section name."""
        old_prefix = pipeline.base_template_prefix
        fields = []
        for f in pipeline.base_template:
            new_path = new_name + f.path[len(old_prefix):]
            new_section = new_name + f.section[len(old_prefix):]
            fields.append(LoaderFieldDef(
                path=new_path,
                field_type=f.field_type,
                default=f.default,
                description=f.description,
                required=f.required,
                section=new_section,
                choices=list(f.choices),
            ))
        return fields

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-sync-remote":
            self._sync_to_remote()
        elif event.button.id == "btn-add-base":
            self._show_add_base_input()
        elif event.button.id == "btn-confirm-add-base":
            self._confirm_add_base()
        elif event.button.id == "btn-cancel-add-base":
            self._cancel_add_base()

    def _show_add_base_input(self) -> None:
        form = self._current_form
        if form is None:
            return
        try:
            form.query_one("#btn-add-base").remove()
        except Exception:
            pass
        bar = Horizontal(
            Input(placeholder="Device name (e.g. Jabra)", id="add-base-input"),
            Button("Add", variant="success", id="btn-confirm-add-base"),
            Button("Cancel", id="btn-cancel-add-base"),
            classes="add-base-bar",
        )
        try:
            form.mount(bar, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(bar)

    def _confirm_add_base(self) -> None:
        try:
            inp = self.query_one("#add-base-input", Input)
            name = inp.value.strip()
        except Exception:
            return
        if not name:
            return

        pipeline = self._current_pipeline
        form = self._current_form
        if not pipeline or not pipeline.base_template or not form:
            return

        section_name = f"{pipeline.base_section}.{name}"
        fields = self._clone_template_fields(pipeline, section_name)
        form.add_section(section_name, fields, {})
        self._restore_add_base_button()

    def _cancel_add_base(self) -> None:
        self._restore_add_base_button()

    def _restore_add_base_button(self) -> None:
        form = self._current_form
        if form is None:
            return
        for old in form.query(".add-base-bar"):
            old.remove()
        btn = Button("+ Add Base", variant="success", id="btn-add-base")
        try:
            form.mount(btn, before=form.query_one(".form-actions"))
        except Exception:
            form.mount(btn)

    def _sync_to_remote(self) -> None:
        if self._current_pipeline is None:
            return
        try:
            sel = self.query_one("#sync-profile-select", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                self._show_status("Select an SSH profile first.")
                return
            profile_name = str(val)
        except Exception:
            return

        profile = get_profile_by_name(profile_name)
        if profile is None:
            self._show_status(f"SSH profile '{profile_name}' not found.")
            return

        local_path = self._current_pipeline.config_path
        if not os.path.isfile(local_path):
            self._show_status("Local config.yml not found. Save first.")
            return

        rel_dir = os.path.relpath(
            os.path.dirname(local_path), self._root,
        )
        remote_path = f"{profile.remote_project_path}/{rel_dir}/config.yml"

        self._show_status(f"Syncing to {profile_name}:{remote_path} ...")
        self.run_worker(
            self._run_scp(profile_name, local_path, remote_path),
            exclusive=True,
        )

    async def _run_scp(self, profile_name: str, local_path: str, remote_path: str) -> None:
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return
        remote_dir = remote_path.rsplit("/", 1)[0]
        mkdir_proc = await ssh_run_async(profile, f"mkdir -p {remote_dir}")
        await mkdir_proc.wait()

        proc = await scp_file_async(profile, local_path, remote_path)
        assert proc.stdout is not None
        output = ""
        async for line in proc.stdout:
            output += line.decode()
        rc = await proc.wait()
        if rc == 0:
            self._show_status(f"Synced to {profile_name}:{remote_path}")
        else:
            self._show_status(f"Sync failed: {output.strip()}")

    # ── launch logic ─────────────────────────────────────────────

    @property
    def _cmd(self) -> CommandSession:
        return self.query_one("#svc-cmd-session", CommandSession)

    def _log(self, message: str) -> None:
        try:
            self._cmd.log(message)
        except Exception:
            pass

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
            target = _make_target_for(svc.name)
            port = _SYSTEM_SVC_PORTS.get(target)
            if port is not None:
                return _check_port_in_use(port)
            return _check_tmux_session(target)
        return False

    def on_service_card_start_requested(self, event: ServiceCard.StartRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"

        if not is_remote and svc.launch_type != "make" and not _check_config_exists(svc.config_dir):
            self._log(f"[yellow]WARNING: config.yml not found in {svc.config_dir}[/yellow]")
            self._log("[yellow]Please configure this pipeline first.[/yellow]")
            return

        if not is_remote and svc.launch_type != "make":
            has_conda = shutil.which("conda") is not None
            if has_conda and not _check_conda_env(svc.conda_env):
                self._log(f"[red]conda env '{svc.conda_env}' not found. Aborting launch.[/red]")
                self._log(f"[yellow]Create it with: conda create -n {svc.conda_env} python=3.10[/yellow]")
                return
        elif not is_remote and svc.launch_type == "make" and svc.conda_env:
            has_conda = shutil.which("conda") is not None
            need_env = _make_target_for(svc.name) in ("flask", "next", "celery", "nginx")
            if need_env and has_conda and not _check_conda_env(svc.conda_env):
                self._log(f"[red]conda env '{svc.conda_env}' not found. Aborting launch.[/red]")
                self._log(f"[yellow]Create it with: conda create -n {svc.conda_env} python=3.10[/yellow]")
                return

        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[green]Starting {svc.name} {target_label}...[/green]")

        if is_remote:
            self._launch_remote(svc, event.params, target)
        else:
            self._launch_service(svc, event.params)
        self.set_timer(2.0, self._build_tree)

    def on_service_card_stop_requested(self, event: ServiceCard.StopRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"
        target_label = f"on '{target}'" if is_remote else "locally"
        self._log(f"[red]Stopping {svc.name} {target_label}...[/red]")

        if is_remote:
            self._stop_remote(svc, target)
        else:
            self._stop_service(svc)
        self.set_timer(2.0, self._build_tree)

    def on_service_card_refresh_requested(self, event: ServiceCard.RefreshRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return
        target = self._get_panel_target()
        if target == "local":
            is_running = self._detect_running(svc)
        else:
            is_running = self._detect_running_remote(svc, target)
        self._svc_states[svc.name] = is_running
        for card in self.query(ServiceCard):
            if card.service_def.name == svc.name:
                card.update_status(is_running)
        self._build_tree()
        status = "[green]Running[/green]" if is_running else "[red]Stopped[/red]"
        self._log(f"{svc.name} ({target}): {status}")

    def _detect_running_remote(self, svc: ServiceDef, profile_name: str) -> bool:
        """check if a service is running on a remote host via SSH."""
        profile = get_profile_by_name(profile_name)
        if profile is None:
            return False
        if svc.launch_type == "make":
            target = _make_target_for(svc.name)
            port = _SYSTEM_SVC_PORTS.get(target)
            if port is not None:
                return ssh_check_port(profile, port)
            return ssh_check_tmux(profile, target)
        elif svc.launch_type == "tmux":
            session_name = svc.name.lower().replace(" ", "-")
            return ssh_check_tmux(profile, session_name)
        return False

    def on_service_card_view_logs_requested(self, event: ServiceCard.ViewLogsRequested) -> None:
        svc = next((s for s in self._services if s.name == event.service_name), None)
        if svc is None:
            return

        target = self._get_panel_target()
        is_remote = target != "local"

        if is_remote:
            is_running = self._detect_running_remote(svc, target)
        else:
            is_running = self._detect_running(svc)

        if not is_running:
            self._log(f"[yellow]{svc.name} is not running ({target}). Start the service first.[/yellow]")
            return

        if is_remote:
            self._view_logs_remote(svc)
        else:
            self._view_logs_local(svc)

    def _view_logs_local(self, svc: ServiceDef) -> None:
        if svc.launch_type == "make":
            target = _make_target_for(svc.name)
            if target in _SYSTEM_SVC_PORTS:
                self._log(f"[cyan]── Logs for {svc.name} ──[/cyan]")
                output = _get_system_service_log(target)
                for line in output.splitlines():
                    self._log(line)
                self._log(f"[cyan]── End of logs ──[/cyan]")
                return
            session_name = target
        elif svc.launch_type == "tmux":
            session_name = svc.name.lower().replace(" ", "-")
        else:
            self._log(f"[yellow]No logs available for {svc.name}[/yellow]")
            return

        self._log(f"[cyan]── Logs for {svc.name} (session: {session_name}) ──[/cyan]")
        output = _capture_tmux_pane(session_name)
        for line in output.splitlines():
            self._log(line)
        self._log(f"[cyan]── End of logs ──[/cyan]")

    def _view_logs_remote(self, svc: ServiceDef) -> None:
        if svc.launch_type == "make":
            target = _make_target_for(svc.name)
            if target in _SYSTEM_SVC_PORTS:
                log_cmds = {
                    "influxdb": "journalctl -u influxdb -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/influxdb/influxd.log 2>/dev/null || echo '(no influxdb logs found)'",
                    "redis": "journalctl -u redis-server -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/redis/redis-server.log 2>/dev/null || echo '(no redis logs found)'",
                    "mosquitto": "journalctl -u mosquitto -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/mosquitto/mosquitto.log 2>/dev/null || echo '(no mosquitto logs found)'",
                    "nginx": "journalctl -u nginx -n 80 --no-pager 2>/dev/null || tail -n 80 /var/log/nginx/error.log 2>/dev/null || echo '(no nginx logs found)'",
                }
                cmd = log_cmds.get(target, f"echo '(no log command for {target})'")
                self._cmd.run(cmd)
                return
            session_name = target
        elif svc.launch_type == "tmux":
            session_name = svc.name.lower().replace(" ", "-")
        else:
            self._log(f"[yellow]No logs available for {svc.name}[/yellow]")
            return

        self._cmd.run(f"tmux capture-pane -t {session_name} -p -S -80")

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

            if os.path.sep in comp.script or comp.script.endswith(".py"):
                script_path = os.path.join(svc.config_dir, comp.script)
                py_cmd = f"python3 {script_path}"
            else:
                py_cmd = comp.script
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
        target = _make_target_for(svc.name)
        make_dir = svc.config_dir
        makefile = os.path.join(make_dir, "Makefile")
        if not os.path.isfile(makefile):
            self._log(f"[red]Makefile not found: {makefile}[/red]")
            return

        if target in _SYSTEM_SVC_PORTS:
            self._log(f"  Running: make {target} (may ask for sudo password)")
            self._cmd.run(f'make -C {make_dir} {target} SUDO="sudo -S"')
        else:
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
                make_name = _make_target_for(svc.name)
                stop_target = "stop-" + make_name
                if make_name in _SYSTEM_SVC_PORTS:
                    self._cmd.run(f'make -C {svc.config_dir} {stop_target} SUDO="sudo -S"')
                else:
                    subprocess.run(
                        ["make", stop_target],
                        cwd=svc.config_dir,
                        capture_output=True,
                        timeout=15,
                    )
                    self._log(f"[red]{svc.name} stopped.[/red]")
                    kill_info = _APP_PORT_CMDS.get(make_name)
                    if kill_info:
                        self._kill_port(kill_info[0], kill_info[1])
            elif svc.launch_type == "bash":
                self._log(f"[yellow]Bash-launched services must be stopped from their terminal windows.[/yellow]")
        except Exception as e:
            self._log(f"[red]Error stopping {svc.name}: {e}[/red]")

    def _kill_port(self, port: int, expected_cmd: str) -> None:
        """force-kill processes on a port whose command name matches expected_cmd."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip()
            if not pids:
                return
            killed = 0
            for pid in pids.splitlines():
                pid = pid.strip()
                if not pid:
                    continue
                ps = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True, text=True, timeout=5,
                )
                cmd = ps.stdout.strip().lower()
                if expected_cmd.lower() not in cmd:
                    continue
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                killed += 1
            if killed:
                self._log(f"  Cleaned {killed} process(es) on port {port}")
        except Exception:
            pass

    # ── remote operations ────────────────────────────────────────

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
                target = _make_target_for(svc.name)
                remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                self._cmd.run(f"cd {remote_dir} && make {target}")
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
                target = "stop-" + _make_target_for(svc.name)
                remote_dir = f"{remote_root}/{os.path.relpath(svc.config_dir, self._root)}"
                self._cmd.run(f"cd {remote_dir} && make {target}")

            elif svc.launch_type == "bash":
                self._log(
                    f"[yellow]Remote bash services must be stopped "
                    f"from the remote terminal.[/yellow]"
                )
        except Exception as e:
            self._log(f"[red]Remote stop error: {e}[/red]")
