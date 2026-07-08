from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Select, TabbedContent, TabPane


def _safe_id(raw: str) -> str:
    """sanitize a string to be a valid textual widget id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw)


@dataclass
class ComponentDef:
    role: str
    script: str
    count_flag: str
    flags: list = field(default_factory=list)


@dataclass
class ServiceDef:
    name: str
    category: str
    conda_env: str
    config_dir: str
    launch_type: str
    description: str = ""
    params: list = field(default_factory=list)
    components: list = field(default_factory=list)
    artifact_pipeline: str = ""


@dataclass
class ParamDef:
    flag: str
    label: str
    param_type: str
    default: Any
    choices: list[str] = field(default_factory=list)


# flags whose Select lists artifact/collection sessions; these get an inline
# "↻" button so the list can be re-queried on demand (bypassing the cache)
_SESSION_PARAM_FLAGS = {"-sid", "--session-id", "--artifact-session-id"}


class ServiceCard(Widget):
    """card widget displaying a service with start/stop controls and parameters."""

    class StartRequested(Message):
        def __init__(self, service_name: str, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params

    class StopRequested(Message):
        def __init__(self, service_name: str, params: dict | None = None) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params or {}

    class ViewLogsRequested(Message):
        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class RefreshRequested(Message):
        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class SessionRefreshRequested(Message):
        """user clicked the ↻ next to a session Select: re-query the session list."""

        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class DownloadRequested(Message):
        def __init__(self, service_name: str, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params

    class DeleteFilesRequested(Message):
        def __init__(self, service_name: str, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params

    DEFAULT_CSS = """
    ServiceCard {
        height: auto;
        border: solid $primary;
        padding: 0 1;
        margin: 0 1 1 1;
    }
    /* the card's content wrapper must size to its rows: Vertical defaults to
       height:1fr, which would clamp the card to the scroll viewport and clip
       the lower param rows inside the border */
    ServiceCard Vertical {
        height: auto;
    }
    ServiceCard .card-title {
        text-style: bold;
        color: $text;
        margin-bottom: 0;
    }
    ServiceCard .card-meta {
        color: $text-muted;
    }
    ServiceCard .card-status {
        margin: 0 0 1 0;
    }
    ServiceCard .card-params {
        margin: 0;
    }
    /* counts wrap to the next row when the card is too narrow (grid column
       count is recomputed on resize in on_resize); height auto so the card
       grows instead of clipping the steppers */
    ServiceCard .card-counts {
        layout: grid;
        grid-size: 3;
        grid-rows: 3;
        grid-columns: 38;
        grid-gutter: 0 2;
        height: auto;
        margin: 0 0 1 0;
    }
    ServiceCard .param-cell {
        layout: horizontal;
        width: 100%;
        height: 3;
    }
    ServiceCard .param-row {
        layout: horizontal;
        height: auto;
        min-height: 3;
        margin-bottom: 0;
        padding: 0;
    }
    ServiceCard .param-label {
        width: 22;
        height: 3;
        content-align: left middle;
    }
    ServiceCard .param-step {
        width: 3;
        min-width: 3;
        height: 3;
        margin-right: 1;
    }
    ServiceCard .param-value {
        width: 5;
        height: 3;
        margin-right: 1;
        content-align: center middle;
        text-style: bold;
        background: $boost;
    }
    ServiceCard .param-toggle {
        width: 9;
        min-width: 9;
        height: 3;
    }
    ServiceCard .param-input {
        width: 44;
        min-width: 22;
        height: 3;
    }
    ServiceCard .param-select {
        width: 44;
        min-width: 22;
        height: 3;
    }
    ServiceCard .param-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    /* action buttons wrap to the next row when the card is too narrow (grid
       column count is recomputed on resize in on_resize) */
    ServiceCard .card-actions {
        layout: grid;
        grid-size: 5;
        grid-rows: 3;
        grid-columns: 16;
        grid-gutter: 0 1;
        height: auto;
        margin: 0 0 1 0;
    }
    ServiceCard .card-actions Button {
        width: 100%;
        height: 3;
    }
    """

    # approximate widths (cols) used to compute how many items fit per row
    _ACTION_BTN_W = 16
    _ACTION_GUTTER = 1
    _COUNT_CELL_W = 38
    _COUNT_GUTTER = 2

    def on_resize(self, event) -> None:
        self._reflow_rows(event.size.width)

    def _reflow_rows(self, width: int) -> None:
        """Recompute grid column counts so action buttons / count steppers wrap
        to the next row instead of being clipped when the card is too narrow."""
        # usable inner width (border + padding ≈ 4 cols)
        avail = max(1, int(width) - 4)
        act_cols = max(1, (avail + self._ACTION_GUTTER) // (self._ACTION_BTN_W + self._ACTION_GUTTER))
        cnt_cols = max(1, (avail + self._COUNT_GUTTER) // (self._COUNT_CELL_W + self._COUNT_GUTTER))
        try:
            for row in self.query(".card-actions"):
                n = len(list(row.children))
                row.styles.grid_size_columns = max(1, min(act_cols, n)) if n else 1
        except Exception:
            pass
        try:
            for row in self.query(".card-counts"):
                n = len(list(row.children))
                row.styles.grid_size_columns = max(1, min(cnt_cols, n)) if n else 1
        except Exception:
            pass

    def __init__(
        self,
        service_def: ServiceDef,
        is_running: bool = False,
        stack_components: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.service_def = service_def
        self._is_running = is_running
        self._param_values = {
            param.flag: self._initial_param_value(param)
            for param in self.service_def.params
        }
        # sub-services of a stack service (e.g. AudioInferer, SpeechTranscriber);
        # each gets a launch toggle, all enabled by default.
        self.stack_components = list(stack_components or [])
        self._component_enabled = {name: True for name in self.stack_components}

    def _component_toggle_id(self, name: str) -> str:
        return _safe_id(f"component_toggle__{self.service_def.name}__{name}")

    def enabled_components(self) -> list[str]:
        return [n for n in self.stack_components if self._component_enabled.get(n, True)]

    def _toggle_component(self, name: str) -> None:
        self._component_enabled[name] = not self._component_enabled.get(name, True)
        try:
            button = self.query_one(f"#{self._component_toggle_id(name)}", Button)
            value = self._component_enabled[name]
            button.label = self._bool_label(value)
            button.variant = self._bool_variant(value)
        except Exception:
            pass

    @property
    def _is_interactive(self) -> bool:
        """bash services run in their own terminal window and aren't tracked,
        so they can't show a live status or be stopped from the TUI."""
        return self.service_def.launch_type == "bash"

    def _status_markup(self) -> str:
        if self._is_interactive:
            return "[yellow]Interactive (runs in its own terminal)[/yellow]"
        return "[green]Running[/green]" if self._is_running else "[red]Stopped[/red]"

    def compose(self) -> ComposeResult:
        if self.service_def.launch_type == "collection":
            yield from self._compose_collection()
            return

        status_text = self._status_markup()

        with Vertical():
            yield Static(f"[b]{self.service_def.name}[/b]", classes="card-title")
            yield Static(
                f"  env: {self.service_def.conda_env}  |  type: {self.service_def.launch_type}",
                classes="card-meta",
            )
            if self.service_def.description:
                yield Static(f"  {self.service_def.description}", classes="card-meta")
            yield Static(f"  Status: {status_text}", classes="card-status")

            if self.stack_components:
                yield Static("  Services to launch:", classes="card-meta")
                with Vertical(classes="card-params"):
                    for name in self.stack_components:
                        with Horizontal(classes="param-row"):
                            yield Static(f"{name}:", classes="param-label")
                            enabled = self._component_enabled.get(name, True)
                            yield Button(
                                self._bool_label(enabled),
                                variant=self._bool_variant(enabled),
                                compact=True,
                                id=self._component_toggle_id(name),
                                classes="param-toggle",
                            )

            if self.service_def.params:
                count_params = [p for p in self.service_def.params if p.param_type == "int"]
                option_params = [p for p in self.service_def.params if p.param_type != "int"]

                if count_params:
                    with Horizontal(classes="card-counts"):
                        for p in count_params:
                            with Horizontal(classes="param-cell"):
                                for widget in self._param_widgets(p):
                                    yield widget

            with Horizontal(classes="card-actions"):
                yield Button(
                    "Start",
                    variant="success",
                    compact=True,
                    id=_safe_id(f"start__{self.service_def.name}"),
                )
                # interactive (bash) services are stopped from their own terminal,
                # so don't offer a Stop button that can't do anything
                if not self._is_interactive:
                    yield Button(
                        "Stop",
                        variant="error",
                        compact=True,
                        id=_safe_id(f"stop__{self.service_def.name}"),
                    )
                yield Button(
                    "Logs",
                    variant="primary",
                    compact=True,
                    id=_safe_id(f"logs__{self.service_def.name}"),
                )
                yield Button(
                    "Refresh",
                    variant="primary",
                    compact=True,
                    id=_safe_id(f"refresh__{self.service_def.name}"),
                )
                if self.service_def.launch_type == "collection" or self.service_def.artifact_pipeline:
                    yield Button(
                        "Download" if self.service_def.launch_type == "collection" else "Artifacts",
                        variant="warning",
                        compact=True,
                        id=_safe_id(f"download__{self.service_def.name}"),
                    )
                if self.service_def.launch_type == "collection":
                    yield Button(
                        "Delete Remote",
                        variant="error",
                        compact=True,
                        id=_safe_id(f"delete_files__{self.service_def.name}"),
                    )

            if self.service_def.params:
                option_params = [p for p in self.service_def.params if p.param_type != "int"]
                if option_params:
                    with Vertical(classes="card-params"):
                        for p in option_params:
                            with Horizontal(classes="param-row"):
                                for widget in self._param_widgets(p):
                                    yield widget

    def _compose_collection(self) -> ComposeResult:
        status_text = "[green]Running[/green]" if self._is_running else "[red]Stopped[/red]"
        with Vertical():
            yield Static(f"[b]{self.service_def.name}[/b]", classes="card-title")
            yield Static(
                f"  env: {self.service_def.conda_env}  |  type: {self.service_def.launch_type}",
                classes="card-meta",
            )
            if self.service_def.description:
                yield Static(f"  {self.service_def.description}", classes="card-meta")
            yield Static(f"  Status: {status_text}", classes="card-status")

            with TabbedContent(id=_safe_id(f"collection_tabs__{self.service_def.name}")):
                for label, role in (("Audio", "audio"), ("Video", "video")):
                    with TabPane(label, id=_safe_id(f"collection_tab__{self.service_def.name}__{role}")):
                        yield from self._compose_collection_role(role)

    def _compose_collection_role(self, role: str) -> ComposeResult:
        count_param = self._collection_count_param(role)
        if count_param is not None:
            with Horizontal(classes="card-counts"):
                with Horizontal(classes="param-cell"):
                    for widget in self._param_widgets(
                        count_param,
                        param_id=lambda kind, flag: self._collection_param_id(role, kind, flag),
                    ):
                        yield widget

        with Horizontal(classes="card-actions"):
            title_role = role.title()
            yield Button(
                f"Start {title_role}",
                variant="success",
                compact=True,
                id=_safe_id(f"start_collection__{self.service_def.name}__{role}"),
            )
            yield Button(
                "Stop",
                variant="error",
                compact=True,
                id=_safe_id(f"stop_collection__{self.service_def.name}__{role}"),
            )
            yield Button(
                "Logs",
                variant="primary",
                compact=True,
                id=_safe_id(f"logs_collection__{self.service_def.name}__{role}"),
            )
            yield Button(
                "Refresh",
                variant="primary",
                compact=True,
                id=_safe_id(f"refresh_collection__{self.service_def.name}__{role}"),
            )
            yield Button(
                "Download",
                variant="warning",
                compact=True,
                id=_safe_id(f"download_collection__{self.service_def.name}__{role}"),
            )
            yield Button(
                "Delete Remote",
                variant="error",
                compact=True,
                id=_safe_id(f"delete_collection__{self.service_def.name}__{role}"),
            )

        with Vertical(classes="card-params"):
            for param in self._collection_params_for_role(role):
                with Horizontal(classes="param-row"):
                    for widget in self._param_widgets(
                        param,
                        param_id=lambda kind, flag, role=role: self._collection_param_id(role, kind, flag),
                    ):
                        yield widget

    def _param_widgets(self, param: ParamDef, param_id=None) -> list:
        """build the widgets for one launch parameter control."""
        param_id = param_id or self._param_id
        widgets = [Static(f"{param.label}:", classes="param-label")]
        if param.param_type == "bool":
            value = bool(self._param_values[param.flag])
            widgets.append(
                Button(
                    self._bool_label(value),
                    variant=self._bool_variant(value),
                    compact=True,
                    id=param_id("toggle", param.flag),
                    classes="param-toggle",
                )
            )
        elif param.param_type == "int":
            widgets.extend(
                [
                    Button(
                        "-",
                        compact=True,
                        id=param_id("dec", param.flag),
                        classes="param-step",
                    ),
                    Static(
                        str(self._param_values[param.flag]),
                        id=param_id("value", param.flag),
                        classes="param-value",
                    ),
                    Button(
                        "+",
                        compact=True,
                        id=param_id("inc", param.flag),
                        classes="param-step",
                    ),
                ]
            )
        elif param.choices:
            options = [(str(choice), str(choice)) for choice in param.choices]
            value = self._param_values[param.flag]
            if not any(option_value == value for _, option_value in options):
                value = Select.NULL
            widgets.append(
                Select(
                    options,
                    value=value,
                    prompt=f"Select {param.label.lower()}...",
                    id=param_id("select", param.flag),
                    classes="param-select",
                )
            )
            if param.flag in _SESSION_PARAM_FLAGS:
                widgets.append(
                    Button(
                        "↻",
                        variant="primary",
                        compact=True,
                        id=param_id("sessionrefresh", param.flag),
                        classes="param-refresh",
                    )
                )
        else:
            widgets.append(
                Input(
                    value=str(self._param_values[param.flag] or ""),
                    id=param_id("input", param.flag),
                    classes="param-input",
                )
            )
        return widgets

    def collect_params(self) -> dict:
        """gather current launch parameter values."""
        for param in self.service_def.params:
            if param.param_type in ("bool", "int"):
                continue
            try:
                if param.choices:
                    sel = self.query_one(f"#{self._param_id('select', param.flag)}", Select)
                    self._param_values[param.flag] = "" if sel.value is Select.NULL else str(sel.value)
                else:
                    inp = self.query_one(f"#{self._param_id('input', param.flag)}", Input)
                    self._param_values[param.flag] = inp.value
            except Exception:
                pass
        return dict(self._param_values)

    def _initial_param_value(self, param: ParamDef) -> Any:
        if param.param_type == "bool":
            return param.default if isinstance(param.default, bool) else str(param.default).lower() == "true"
        if param.param_type == "int":
            try:
                return max(0, int(param.default))
            except (TypeError, ValueError):
                return 0
        return param.default

    def _param_id(self, kind: str, flag: str) -> str:
        return _safe_id(f"param_{kind}__{self.service_def.name}__{flag}")

    def _collection_param_id(self, role: str, kind: str, flag: str) -> str:
        return _safe_id(f"param_{kind}__{self.service_def.name}__{role}__{flag}")

    def _collection_count_param(self, role: str) -> ParamDef | None:
        count_flag = next(
            (component.count_flag for component in self.service_def.components if component.role == role),
            "",
        )
        return next((param for param in self.service_def.params if param.flag == count_flag), None)

    def _collection_params_for_role(self, role: str) -> list[ParamDef]:
        component = next((component for component in self.service_def.components if component.role == role), None)
        if component is None:
            return []
        flags = [
            "--session-id",
            "--experiment-group",
            "--output-root",
            "--host-label",
            *[
                flag for flag in component.flags
                if flag not in {"--session-id", "--experiment-group", "--output-root", "--host-label"}
            ],
        ]
        params = {param.flag: param for param in self.service_def.params}
        return [params[flag] for flag in flags if flag in params]

    @staticmethod
    def _bool_label(value: bool) -> str:
        return "true" if value else "false"

    @staticmethod
    def _bool_variant(value: bool) -> str:
        return "success" if value else "default"

    def _change_int_param(self, flag: str, delta: int) -> None:
        self._param_values[flag] = max(0, int(self._param_values.get(flag, 0)) + delta)
        try:
            self.query_one(f"#{self._param_id('value', flag)}", Static).update(str(self._param_values[flag]))
        except Exception:
            pass

    def _change_collection_int_param(self, role: str, flag: str, delta: int) -> None:
        self._param_values[flag] = max(0, int(self._param_values.get(flag, 0)) + delta)
        try:
            self.query_one(f"#{self._collection_param_id(role, 'value', flag)}", Static).update(
                str(self._param_values[flag])
            )
        except Exception:
            pass

    def _toggle_bool_param(self, flag: str) -> None:
        self._param_values[flag] = not bool(self._param_values.get(flag, False))
        try:
            button = self.query_one(f"#{self._param_id('toggle', flag)}", Button)
            value = bool(self._param_values[flag])
            button.label = self._bool_label(value)
            button.variant = self._bool_variant(value)
        except Exception:
            pass

    def _toggle_collection_bool_param(self, role: str, flag: str) -> None:
        self._param_values[flag] = not bool(self._param_values.get(flag, False))
        try:
            button = self.query_one(f"#{self._collection_param_id(role, 'toggle', flag)}", Button)
            value = bool(self._param_values[flag])
            button.label = self._bool_label(value)
            button.variant = self._bool_variant(value)
        except Exception:
            pass

    def _collect_param_values(self, params: list[ParamDef], param_id) -> dict:
        values: dict[str, Any] = {}
        for param in params:
            if param.param_type in ("bool", "int"):
                values[param.flag] = self._param_values[param.flag]
                continue
            try:
                if param.choices:
                    sel = self.query_one(f"#{param_id('select', param.flag)}", Select)
                    values[param.flag] = "" if sel.value is Select.NULL else str(sel.value)
                else:
                    inp = self.query_one(f"#{param_id('input', param.flag)}", Input)
                    values[param.flag] = inp.value
            except Exception:
                values[param.flag] = self._param_values.get(param.flag, param.default)
        return values

    def _collect_collection_params(self, role: str, *, start: bool = False) -> dict:
        params = self._collection_params_for_role(role)
        values = self._collect_param_values(
            params,
            lambda kind, flag: self._collection_param_id(role, kind, flag),
        )
        count_param = self._collection_count_param(role)
        if count_param is not None:
            count = int(self._param_values.get(count_param.flag, 0))
            values[count_param.flag] = max(0, count)
        if start:
            for component in self.service_def.components:
                if component.role != role:
                    values[component.count_flag] = 0
        return values

    def update_status(self, is_running: bool) -> None:
        """update the displayed status."""
        self._is_running = is_running
        # bash/interactive services keep the "Interactive" label regardless
        try:
            self.query_one(".card-status", Static).update(f"  Status: {self._status_markup()}")
        except Exception:
            pass

    def update_stack_status(self, up: int, total: int) -> None:
        """update status for a stack service as a running-count, e.g. '3/6'."""
        self._is_running = total > 0 and up == total
        if total <= 0:
            status_text = "[red]Stopped[/red]"
        elif up == 0:
            status_text = f"[red]Stopped (0/{total})[/red]"
        elif up < total:
            status_text = f"[yellow]Partial ({up}/{total})[/yellow]"
        else:
            status_text = f"[green]Running ({up}/{total})[/green]"
        try:
            self.query_one(".card-status", Static).update(f"  Status: {status_text}")
        except Exception:
            pass

    def update_service_def(self, service_def: ServiceDef) -> None:
        """update service metadata displayed by an already-mounted card."""
        existing_params = {param.flag: param for param in self.service_def.params}
        params = []
        for param in service_def.params:
            existing = existing_params.get(param.flag)
            if existing and existing.choices and not param.choices:
                params.append(replace(param, choices=existing.choices, default=existing.default))
            else:
                params.append(param)
        self.service_def = replace(service_def, params=params)
        try:
            metas = list(self.query(".card-meta"))
            if metas:
                metas[0].update(
                    f"  env: {service_def.conda_env}  |  type: {service_def.launch_type}"
                )
            if len(metas) > 1:
                metas[1].update(f"  {service_def.description}")
        except Exception:
            pass

    def update_session_choices(self, choices: list[str]) -> None:
        """replace the session Select options in place, keeping the current
        selection when it is still in the fresh list (other params untouched)."""
        choice_strs = [str(c) for c in choices if c]
        params = []
        for param in self.service_def.params:
            if param.flag not in _SESSION_PARAM_FLAGS:
                params.append(param)
                continue
            params.append(replace(param, choices=list(choice_strs)))
            if self.service_def.launch_type == "collection":
                widget_ids = [
                    self._collection_param_id(role, "select", param.flag)
                    for role in ("audio", "video")
                ]
            else:
                widget_ids = [self._param_id("select", param.flag)]
            for widget_id in widget_ids:
                try:
                    sel = self.query_one(f"#{widget_id}", Select)
                except Exception:
                    continue
                current = sel.value
                sel.set_options((c, c) for c in choice_strs)
                if current is not Select.NULL and str(current) in choice_strs:
                    sel.value = str(current)
        self.service_def = replace(self.service_def, params=params)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if self.service_def.launch_type == "collection":
            for role in ("audio", "video"):
                if btn_id == _safe_id(f"start_collection__{self.service_def.name}__{role}"):
                    params = self._collect_collection_params(role, start=True)
                    self.post_message(self.StartRequested(self.service_def.name, params))
                    return
                if btn_id == _safe_id(f"stop_collection__{self.service_def.name}__{role}"):
                    params = self._collect_collection_params(role)
                    self.post_message(self.StopRequested(self.service_def.name, params))
                    return
                if btn_id == _safe_id(f"logs_collection__{self.service_def.name}__{role}"):
                    self.post_message(self.ViewLogsRequested(self.service_def.name))
                    return
                if btn_id == _safe_id(f"refresh_collection__{self.service_def.name}__{role}"):
                    self.post_message(self.RefreshRequested(self.service_def.name))
                    return
                if btn_id == _safe_id(f"download_collection__{self.service_def.name}__{role}"):
                    params = self._collect_collection_params(role)
                    self.post_message(self.DownloadRequested(self.service_def.name, params))
                    return
                if btn_id == _safe_id(f"delete_collection__{self.service_def.name}__{role}"):
                    params = self._collect_collection_params(role)
                    self.post_message(self.DeleteFilesRequested(self.service_def.name, params))
                    return
                for param in self.service_def.params:
                    if btn_id == self._collection_param_id(role, "inc", param.flag):
                        self._change_collection_int_param(role, param.flag, 1)
                        return
                    if btn_id == self._collection_param_id(role, "dec", param.flag):
                        self._change_collection_int_param(role, param.flag, -1)
                        return
                    if btn_id == self._collection_param_id(role, "toggle", param.flag):
                        self._toggle_collection_bool_param(role, param.flag)
                        return
                    if btn_id == self._collection_param_id(role, "sessionrefresh", param.flag):
                        self.post_message(self.SessionRefreshRequested(self.service_def.name))
                        return

        for name in self.stack_components:
            if btn_id == self._component_toggle_id(name):
                self._toggle_component(name)
                return

        if btn_id.startswith("start__"):
            params = self.collect_params()
            if self.stack_components:
                params["__components__"] = self.enabled_components()
            self.post_message(self.StartRequested(self.service_def.name, params))
        elif btn_id.startswith("stop__"):
            params = self.collect_params()
            self.post_message(self.StopRequested(self.service_def.name, params))
        elif btn_id.startswith("logs__"):
            self.post_message(self.ViewLogsRequested(self.service_def.name))
        elif btn_id.startswith("refresh__"):
            self.post_message(self.RefreshRequested(self.service_def.name))
        elif btn_id.startswith("download__"):
            params = self.collect_params()
            self.post_message(self.DownloadRequested(self.service_def.name, params))
        elif btn_id.startswith("delete_files__"):
            params = self.collect_params()
            self.post_message(self.DeleteFilesRequested(self.service_def.name, params))
        else:
            for param in self.service_def.params:
                if btn_id == self._param_id("inc", param.flag):
                    self._change_int_param(param.flag, 1)
                    return
                if btn_id == self._param_id("dec", param.flag):
                    self._change_int_param(param.flag, -1)
                    return
                if btn_id == self._param_id("toggle", param.flag):
                    self._toggle_bool_param(param.flag)
                    return
                if btn_id == self._param_id("sessionrefresh", param.flag):
                    self.post_message(self.SessionRefreshRequested(self.service_def.name))
                    return
