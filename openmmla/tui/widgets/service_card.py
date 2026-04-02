from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Switch, Label, Select


def _safe_id(raw: str) -> str:
    """sanitize a string to be a valid textual widget id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw)


@dataclass
class ServiceDef:
    name: str
    category: str
    conda_env: str
    config_dir: str
    launch_type: str
    description: str = ""
    params: list = field(default_factory=list)


@dataclass
class ParamDef:
    flag: str
    label: str
    param_type: str
    default: Any


class ServiceCard(Widget):
    """card widget displaying a service with start/stop controls and parameters."""

    class StartRequested(Message):
        def __init__(self, service_name: str, params: dict, target: str = "local") -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params
            self.target = target

    class StopRequested(Message):
        def __init__(self, service_name: str, target: str = "local") -> None:
            super().__init__()
            self.service_name = service_name
            self.target = target

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
    ServiceCard .param-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 0;
    }
    ServiceCard .param-label {
        width: 20;
    }
    ServiceCard .param-input {
        width: 1fr;
    }
    ServiceCard .card-target {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }
    ServiceCard .target-label {
        width: 20;
        padding-top: 1;
    }
    ServiceCard .target-select {
        width: 1fr;
    }
    ServiceCard .card-actions {
        height: auto;
    }
    ServiceCard .card-actions Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def __init__(
        self,
        service_def: ServiceDef,
        is_running: bool = False,
        ssh_profile_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.service_def = service_def
        self._is_running = is_running
        self._ssh_profile_names = ssh_profile_names or []

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

            target_options = [("Local", "local")] + [
                (name, name) for name in self._ssh_profile_names
            ]
            target_id = _safe_id(f"target__{self.service_def.name}")
            with Horizontal(classes="card-target"):
                yield Label("Target:", classes="target-label")
                yield Select(target_options, value="local", id=target_id, classes="target-select")

            if self.service_def.params:
                with Vertical(classes="card-params"):
                    for p in self.service_def.params:
                        with Horizontal(classes="param-row"):
                            yield Label(f"{p.label}:", classes="param-label")
                            widget_id = _safe_id(f"param__{self.service_def.name}__{p.flag}")
                            if p.param_type == "bool":
                                val = p.default if isinstance(p.default, bool) else (str(p.default).lower() == "true")
                                yield Switch(value=val, id=widget_id)
                            elif p.param_type == "int":
                                yield Input(
                                    value=str(p.default),
                                    id=widget_id,
                                    classes="param-input",
                                )
                            else:
                                yield Input(
                                    value=str(p.default),
                                    id=widget_id,
                                    classes="param-input",
                                )

            with Horizontal(classes="card-actions"):
                if self._is_running:
                    yield Button(
                        "Stop",
                        variant="error",
                        id=_safe_id(f"stop__{self.service_def.name}"),
                    )
                else:
                    yield Button(
                        "Start",
                        variant="success",
                        id=_safe_id(f"start__{self.service_def.name}"),
                    )

    def refresh_targets(self, profile_names: list[str]) -> None:
        """update the target selector with fresh SSH profile names."""
        target_id = _safe_id(f"target__{self.service_def.name}")
        try:
            sel = self.query_one(f"#{target_id}", Select)
            current = sel.value
            options = [("Local", "local")] + [(n, n) for n in profile_names]
            sel.set_options(options)
            if any(v == current for _, v in options):
                sel.value = current
        except Exception:
            pass

    def collect_params(self) -> dict:
        """gather current parameter values from the card inputs."""
        result = {}
        for p in self.service_def.params:
            widget_id = _safe_id(f"param__{self.service_def.name}__{p.flag}")
            try:
                w = self.query_one(f"#{widget_id}")
            except Exception:
                result[p.flag] = p.default
                continue
            if isinstance(w, Switch):
                result[p.flag] = w.value
            elif isinstance(w, Input):
                raw = w.value.strip()
                if p.param_type == "int":
                    try:
                        result[p.flag] = int(raw)
                    except ValueError:
                        result[p.flag] = p.default
                else:
                    result[p.flag] = raw
            else:
                result[p.flag] = p.default
        return result

    def _get_target(self) -> str:
        target_id = _safe_id(f"target__{self.service_def.name}")
        try:
            sel = self.query_one(f"#{target_id}", Select)
            val = sel.value
            if val is Select.BLANK or val is None:
                return "local"
            return str(val)
        except Exception:
            return "local"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        target = self._get_target()
        if btn_id.startswith("start__"):
            params = self.collect_params()
            self.post_message(self.StartRequested(self.service_def.name, params, target))
        elif btn_id.startswith("stop__"):
            self.post_message(self.StopRequested(self.service_def.name, target))
