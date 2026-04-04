from __future__ import annotations

import os

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll, Vertical
from textual.widget import Widget
from textual.widgets import Static, Tree, Button, Select, Input

from openmmla.tui.schema.loader import (
    FieldDef as LoaderFieldDef,
    discover_pipelines, load_existing_config, get_nested_value,
    save_config, PipelineDef, _find_project_root,
)
from openmmla.tui.schema.definitions import (
    SHARED_SECTIONS, get_shared_defaults, apply_shared_values,
)
from openmmla.tui.ssh import (
    load_ssh_profiles, get_profile_by_name, scp_file_async, ssh_run_async,
)
from openmmla.tui.widgets.config_form import ConfigForm
from openmmla.tui.widgets.ssh_form import SSHForm


class ConfigPanel(Widget):

    DEFAULT_CSS = """
    ConfigPanel {
        layout: horizontal;
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
    """

    def __init__(self) -> None:
        super().__init__()
        self._root = _find_project_root()
        self._pipelines: list[PipelineDef] = []
        self._pipeline_map: dict[str, PipelineDef] = {}
        self._shared_values: dict[str, object] = get_shared_defaults()
        self._current_pipeline: PipelineDef | None = None
        self._current_form: ConfigForm | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="sidebar"):
            yield Static("[b]Pipelines[/b]", classes="status-info")
            tree: Tree[str] = Tree("OpenMMLA", id="pipeline-tree")
            tree.root.expand()
            yield tree
        with VerticalScroll(id="main-area"):
            yield Static(
                "Select a pipeline from the sidebar\nto edit its configuration.",
                id="empty-state",
            )

    def on_mount(self) -> None:
        self._pipelines = discover_pipelines()
        self._pipeline_map = {p.name: p for p in self._pipelines}
        self._build_tree()

    def _build_tree(self) -> None:
        tree = self.query_one("#pipeline-tree", Tree)
        tree.clear()

        shared_node = tree.root.add("Global Defaults", data="__shared__")
        shared_node.expand()
        for sec in SHARED_SECTIONS:
            shared_node.add_leaf(sec, data=f"__shared__{sec}")

        bases_node = tree.root.add("Base Stations", data="__group_bases__")
        bases_node.expand()
        servers_node = tree.root.add("Servers", data="__group_servers__")
        servers_node.expand()

        for p in self._pipelines:
            config_exists = os.path.isfile(p.config_path)
            marker = " [OK]" if config_exists else ""
            label = f"{p.name}{marker}"
            if "Base" in p.name:
                bases_node.add_leaf(label, data=p.name)
            else:
                servers_node.add_leaf(label, data=p.name)

        tree.root.add_leaf("SSH Profiles", data="__ssh_profiles__")

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node_data = event.node.data
        if node_data is None or str(node_data).startswith("__group"):
            return

        main_area = self.query_one("#main-area", VerticalScroll)
        await main_area.remove_children()

        node_str = str(node_data)

        if node_str == "__shared__":
            return

        if node_str.startswith("__shared__"):
            section_name = node_str.replace("__shared__", "")
            self._show_shared_form(main_area, section_name)
            return

        if node_str == "__ssh_profiles__":
            main_area.mount(SSHForm())
            return

        pipeline = self._pipeline_map.get(node_str)
        if pipeline is None:
            return

        self._current_pipeline = pipeline
        self._show_pipeline_form(main_area, pipeline)

    def _show_shared_form(self, container, section_name: str) -> None:
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

    def _show_pipeline_form(self, container, pipeline: PipelineDef) -> None:
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
        if pipeline.base_template:
            base_sections = [k for k in existing if k.startswith("Base_")]
            for base_name in base_sections:
                fields = self._clone_template_fields(pipeline.base_template, base_name)
                dynamic_sections[base_name] = fields
                for f in fields:
                    val = get_nested_value(existing, f.path)
                    if val is not None:
                        values[f.path] = val

        form = ConfigForm(pipeline.name, pipeline.fields, values, dynamic_sections)
        container.mount(form)
        self._current_form = form

        if pipeline.base_template:
            container.mount(Button("+ Add Base", variant="success", id="btn-add-base"))

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
        main_area = self.query_one("#main-area", VerticalScroll)
        for old in main_area.query(".status-saved"):
            old.remove()
        main_area.mount(Static(f" {message}", classes="status-saved"))

    def _show_sync_bar(self, pipeline: PipelineDef) -> None:
        profiles = load_ssh_profiles()
        if not profiles:
            return
        main_area = self.query_one("#main-area", VerticalScroll)
        for old in main_area.query(".sync-bar"):
            old.remove()
        options = [(p.name, p.name) for p in profiles]
        bar = Horizontal(
            Select(options, prompt="Select SSH profile...", id="sync-profile-select"),
            Button("Sync to Remote", variant="warning", id="btn-sync-remote"),
            classes="sync-bar",
        )
        main_area.mount(bar)

    def _clone_template_fields(self, base_template: list, new_name: str) -> list[LoaderFieldDef]:
        """clone base_template fields with paths rewritten to a new section name."""
        fields = []
        for f in base_template:
            old_top = f.path.split(".")[0]
            new_path = new_name + f.path[len(old_top):]
            old_sec_top = f.section.split(".")[0]
            new_section = new_name + f.section[len(old_sec_top):]
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
        main_area = self.query_one("#main-area", VerticalScroll)
        try:
            main_area.query_one("#btn-add-base").remove()
        except Exception:
            pass
        bar = Horizontal(
            Input(placeholder="Device name (e.g. Jabra)", id="add-base-input"),
            Button("Add", variant="success", id="btn-confirm-add-base"),
            Button("Cancel", id="btn-cancel-add-base"),
            classes="add-base-bar",
        )
        main_area.mount(bar)

    def _confirm_add_base(self) -> None:
        try:
            inp = self.query_one("#add-base-input", Input)
            name = inp.value.strip()
        except Exception:
            return
        if not name:
            return

        section_name = f"Base_{name}"
        pipeline = self._current_pipeline
        form = self._current_form
        if not pipeline or not pipeline.base_template or not form:
            return

        fields = self._clone_template_fields(pipeline.base_template, section_name)
        form.add_section(section_name, fields, {})
        self._restore_add_base_button()

    def _cancel_add_base(self) -> None:
        self._restore_add_base_button()

    def _restore_add_base_button(self) -> None:
        main_area = self.query_one("#main-area", VerticalScroll)
        for old in main_area.query(".add-base-bar"):
            old.remove()
        main_area.mount(Button("+ Add Base", variant="success", id="btn-add-base"))

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
