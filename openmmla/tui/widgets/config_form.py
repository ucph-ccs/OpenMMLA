from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static, Input, Switch, Button, Rule
from textual.widget import Widget

from openmmla.tui.schema.loader import FieldDef


def _safe_id(raw: str) -> str:
    """sanitize a string to be a valid textual widget id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw)


class FieldRow(Widget):
    """a single config field with label and input."""

    DEFAULT_CSS = """
    FieldRow {
        layout: vertical;
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }
    """

    def __init__(self, field_def: FieldDef, initial_value=None) -> None:
        super().__init__()
        self.field_def = field_def
        self._initial = initial_value if initial_value is not None else field_def.default

    def compose(self) -> ComposeResult:
        short_name = self.field_def.path.split(".")[-1]
        required_marker = " *" if self.field_def.required else ""
        label_text = f"{short_name}{required_marker}"
        widget_id = _safe_id(f"field__{self.field_def.path}")

        yield Static(label_text, classes="field-label")
        if self.field_def.description:
            yield Static(
                self.field_def.description,
                classes="field-desc",
            )
        if self.field_def.field_type == "bool":
            val = self._initial if isinstance(self._initial, bool) else False
            yield Switch(value=val, id=widget_id)
        else:
            display = self._to_display(self._initial)
            yield Input(
                value=display,
                placeholder=self.field_def.description or short_name,
                id=widget_id,
                classes="field-input",
            )

    def _to_display(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    @property
    def current_value(self):
        """get the current value from the input widget, converted to the expected type."""
        widget_id = _safe_id(f"field__{self.field_def.path}")
        try:
            w = self.query_one(f"#{widget_id}")
        except Exception:
            return self.field_def.default

        if isinstance(w, Switch):
            return w.value

        raw = w.value.strip()
        if not raw:
            return self.field_def.default

        return _parse_value(raw, self.field_def.field_type)


def _parse_value(raw: str, field_type: str):
    """convert a raw string to the appropriate python type."""
    if field_type == "bool":
        return raw.lower() in ("true", "1", "yes")
    if field_type == "int":
        try:
            return int(raw)
        except ValueError:
            return raw
    if field_type == "float":
        try:
            return float(raw)
        except ValueError:
            return raw
    if field_type == "list":
        items = [s.strip().strip("'\"") for s in raw.split(",") if s.strip()]
        for i, item in enumerate(items):
            try:
                items[i] = int(item)
            except ValueError:
                try:
                    items[i] = float(item)
                except ValueError:
                    pass
        return items
    return raw


class ConfigForm(Widget):
    """form widget that renders a list of FieldDefs grouped by section."""

    class Saved(Message):
        def __init__(self, pipeline_name: str, values: dict) -> None:
            super().__init__()
            self.pipeline_name = pipeline_name
            self.values = values

    DEFAULT_CSS = """
    ConfigForm {
        height: auto;
        padding: 1 2;
    }
    """

    def __init__(self, pipeline_name: str, fields: list[FieldDef], values: dict | None = None) -> None:
        super().__init__()
        self.pipeline_name = pipeline_name
        self._fields = fields
        self._values = values or {}

    def compose(self) -> ComposeResult:
        current_section = None
        for f in self._fields:
            if f.section != current_section:
                current_section = f.section
                yield Rule()
                yield Static(f" [{current_section}]", classes="section-header")
            initial = self._values.get(f.path, f.default)
            yield FieldRow(f, initial_value=initial)

        with Horizontal(classes="form-actions"):
            yield Button("Save", variant="primary", id="btn-save")
            yield Button("Reset to Defaults", variant="warning", id="btn-reset")

    def collect_values(self) -> dict:
        """gather current values from all field rows."""
        values = {}
        for row in self.query(FieldRow):
            values[row.field_def.path] = row.current_value
        return values

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            values = self.collect_values()
            self.post_message(self.Saved(self.pipeline_name, values))
        elif event.button.id == "btn-reset":
            self._reset_to_defaults()

    def _reset_to_defaults(self) -> None:
        for row in self.query(FieldRow):
            widget_id = _safe_id(f"field__{row.field_def.path}")
            try:
                w = self.query_one(f"#{widget_id}")
            except Exception:
                continue
            if isinstance(w, Switch):
                w.value = row.field_def.default if isinstance(row.field_def.default, bool) else False
            elif isinstance(w, Input):
                w.value = row._to_display(row.field_def.default)
