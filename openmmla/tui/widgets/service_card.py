from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button


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


@dataclass
class ParamDef:
    flag: str
    label: str
    param_type: str
    default: Any


class ServiceCard(Widget):
    """card widget displaying a service with start/stop controls and parameters."""

    class StartRequested(Message):
        def __init__(self, service_name: str, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params

    class StopRequested(Message):
        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class ViewLogsRequested(Message):
        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class RefreshRequested(Message):
        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    DEFAULT_CSS = """
    ServiceCard {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    ServiceCard .card-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    ServiceCard .card-meta {
        color: $text-muted;
    }
    ServiceCard .card-status {
        margin: 1 0;
    }
    ServiceCard .card-params {
        margin: 1 0;
    }
    ServiceCard .card-counts {
        layout: horizontal;
        height: auto;
        margin: 1 0;
    }
    ServiceCard .param-cell {
        layout: horizontal;
        width: 1fr;
        min-width: 36;
        height: 3;
        margin-right: 2;
    }
    ServiceCard .param-row {
        layout: horizontal;
        height: auto;
        min-height: 3;
        margin-bottom: 1;
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
    ServiceCard .card-actions {
        height: auto;
        margin-top: 1;
    }
    ServiceCard .card-actions Button {
        margin-right: 1;
        width: 12;
        min-width: 12;
        height: 3;
    }
    """

    def __init__(
        self,
        service_def: ServiceDef,
        is_running: bool = False,
    ) -> None:
        super().__init__()
        self.service_def = service_def
        self._is_running = is_running
        self._param_values = {
            param.flag: self._initial_param_value(param)
            for param in self.service_def.params
        }

    def compose(self) -> ComposeResult:
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

            if self.service_def.params:
                option_params = [p for p in self.service_def.params if p.param_type != "int"]
                if option_params:
                    with Vertical(classes="card-params"):
                        for p in option_params:
                            with Horizontal(classes="param-row"):
                                for widget in self._param_widgets(p):
                                    yield widget

    def _param_widgets(self, param: ParamDef) -> list:
        """build the widgets for one launch parameter control."""
        widgets = [Static(f"{param.label}:", classes="param-label")]
        if param.param_type == "bool":
            value = bool(self._param_values[param.flag])
            widgets.append(
                Button(
                    self._bool_label(value),
                    variant=self._bool_variant(value),
                    compact=True,
                    id=self._param_id("toggle", param.flag),
                    classes="param-toggle",
                )
            )
        elif param.param_type == "int":
            widgets.extend(
                [
                    Button(
                        "-",
                        compact=True,
                        id=self._param_id("dec", param.flag),
                        classes="param-step",
                    ),
                    Static(
                        str(self._param_values[param.flag]),
                        id=self._param_id("value", param.flag),
                        classes="param-value",
                    ),
                    Button(
                        "+",
                        compact=True,
                        id=self._param_id("inc", param.flag),
                        classes="param-step",
                    ),
                ]
            )
        else:
            widgets.append(Static(str(self._param_values[param.flag]), classes="param-value"))
        return widgets

    def collect_params(self) -> dict:
        """gather current launch parameter values."""
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

    def _toggle_bool_param(self, flag: str) -> None:
        self._param_values[flag] = not bool(self._param_values.get(flag, False))
        try:
            button = self.query_one(f"#{self._param_id('toggle', flag)}", Button)
            value = bool(self._param_values[flag])
            button.label = self._bool_label(value)
            button.variant = self._bool_variant(value)
        except Exception:
            pass

    def update_status(self, is_running: bool) -> None:
        """update the displayed status."""
        self._is_running = is_running
        status_text = "[green]Running[/green]" if is_running else "[red]Stopped[/red]"
        try:
            self.query_one(".card-status", Static).update(f"  Status: {status_text}")
        except Exception:
            pass

    def update_service_def(self, service_def: ServiceDef) -> None:
        """update service metadata displayed by an already-mounted card."""
        self.service_def = service_def
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("start__"):
            params = self.collect_params()
            self.post_message(self.StartRequested(self.service_def.name, params))
        elif btn_id.startswith("stop__"):
            self.post_message(self.StopRequested(self.service_def.name))
        elif btn_id.startswith("logs__"):
            self.post_message(self.ViewLogsRequested(self.service_def.name))
        elif btn_id.startswith("refresh__"):
            self.post_message(self.RefreshRequested(self.service_def.name))
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
