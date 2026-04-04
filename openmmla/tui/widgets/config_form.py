from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Static, Input, Switch, Button, Collapsible, TextArea
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
            if "\n" in display or len(display) > 120:
                yield TextArea(
                    display,
                    id=widget_id,
                    classes="field-textarea",
                )
            else:
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

        raw = w.text.strip() if isinstance(w, TextArea) else w.value.strip()
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


def _auto_parse(raw: str):
    """try to convert a string to int or float, otherwise return as-is."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


class DictListField(Widget):
    """widget for editing a list of dicts (e.g., upstream server entries)."""

    DEFAULT_CSS = """
    DictListField {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, field_def: FieldDef, initial_value=None) -> None:
        super().__init__()
        self.field_def = field_def
        self._entries: list[dict] = list(initial_value or field_def.default or [])
        self._schema: dict = field_def.entry_schema or {}
        self._next_idx = 0

    def compose(self) -> ComposeResult:
        short_name = self.field_def.path.split(".")[-1]
        yield Static(f"[b]{short_name}[/b]", classes="field-label")
        if self.field_def.description:
            yield Static(self.field_def.description, classes="field-desc")

        for entry in self._entries:
            yield from self._build_entry(entry)

        add_id = _safe_id(f"dle-add__{self.field_def.path}")
        yield Button("+ Add Entry", variant="success", id=add_id, classes="dle-add-btn")

    def _build_entry(self, entry: dict) -> ComposeResult:
        idx = self._next_idx
        self._next_idx += 1
        container_id = _safe_id(f"dle__{self.field_def.path}__{idx}")
        with Vertical(classes="dict-entry", id=container_id):
            yield Static(f"Entry {idx + 1}", classes="dict-entry-header")
            for key in self._schema:
                val = entry.get(key, "")
                widget_id = _safe_id(f"dle__{self.field_def.path}__{idx}__{key}")
                with Horizontal(classes="dict-entry-row"):
                    yield Static(f"{key}:", classes="dict-entry-label")
                    yield Input(
                        value=str(val) if val is not None else "",
                        id=widget_id,
                        classes="dict-entry-input",
                    )
            rm_id = _safe_id(f"dle-rm__{self.field_def.path}__{idx}")
            yield Button("Remove", variant="error", id=rm_id, classes="dle-rm-btn")

    def add_entry(self) -> None:
        """mount a new empty entry before the add button."""
        defaults = {}
        for key, typ in self._schema.items():
            defaults[key] = "" if typ == "str" else 0
        idx = self._next_idx
        self._next_idx += 1
        container_id = _safe_id(f"dle__{self.field_def.path}__{idx}")
        children: list[Widget] = [Static(f"Entry {idx + 1}", classes="dict-entry-header")]
        for key in self._schema:
            val = defaults.get(key, "")
            widget_id = _safe_id(f"dle__{self.field_def.path}__{idx}__{key}")
            row = Horizontal(
                Static(f"{key}:", classes="dict-entry-label"),
                Input(value=str(val), id=widget_id, classes="dict-entry-input"),
                classes="dict-entry-row",
            )
            children.append(row)
        rm_id = _safe_id(f"dle-rm__{self.field_def.path}__{idx}")
        children.append(Button("Remove", variant="error", id=rm_id, classes="dle-rm-btn"))
        container = Vertical(*children, classes="dict-entry", id=container_id)
        add_btn = self.query_one(f"#{_safe_id(f'dle-add__{self.field_def.path}')}", Button)
        self.mount(container, before=add_btn)

    @property
    def current_value(self) -> list[dict]:
        """collect all remaining entry dicts from the widget tree."""
        result = []
        prefix = _safe_id(f"dle__{self.field_def.path}__")
        for container in self.query(".dict-entry"):
            if not container.id or not container.id.startswith(prefix):
                continue
            entry = {}
            for key in self._schema:
                try:
                    inp = container.query_one(
                        f"#{_safe_id(f'{container.id}__{key}')}", Input
                    )
                    entry[key] = _auto_parse(inp.value.strip())
                except Exception:
                    entry[key] = ""
            result.append(entry)
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        add_id = _safe_id(f"dle-add__{self.field_def.path}")
        rm_prefix = _safe_id(f"dle-rm__{self.field_def.path}__")

        if btn_id == add_id:
            event.stop()
            self.add_entry()
        elif btn_id.startswith(rm_prefix):
            event.stop()
            entry_idx_part = btn_id[len(rm_prefix):]
            container_id = _safe_id(f"dle__{self.field_def.path}__{entry_idx_part}")
            try:
                self.query_one(f"#{container_id}").remove()
            except Exception:
                pass


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

    def __init__(
        self,
        pipeline_name: str,
        fields: list[FieldDef],
        values: dict | None = None,
        dynamic_sections: dict[str, list[FieldDef]] | None = None,
    ) -> None:
        super().__init__()
        self.pipeline_name = pipeline_name
        self._fields = fields
        self._values = values or {}
        self._dynamic_sections: dict[str, list[FieldDef]] = dict(dynamic_sections or {})

    def _yield_field(self, f: FieldDef) -> ComposeResult:
        """yield the appropriate widget for a single field."""
        initial = self._values.get(f.path, f.default)
        if f.field_type == "list_of_dicts":
            yield DictListField(f, initial_value=initial)
        else:
            yield FieldRow(f, initial_value=initial)

    def compose(self) -> ComposeResult:
        tree = self._build_section_tree()
        for top, direct, subs in tree:
            with Collapsible(title=top, collapsed=True):
                for f in direct:
                    yield from self._yield_field(f)
                for sub_name, sub_fields in subs:
                    with Collapsible(title=sub_name, collapsed=True):
                        for f in sub_fields:
                            yield from self._yield_field(f)

        for section_name, fields in self._dynamic_sections.items():
            coll_id = _safe_id(f"dyn-{section_name}")
            btn_id = _safe_id(f"btn-remove-{section_name}")
            with Collapsible(title=section_name, collapsed=True, id=coll_id):
                yield from self._yield_nested_fields(section_name, fields)
                yield Button("Remove", variant="error", id=btn_id)

        with Horizontal(classes="form-actions"):
            yield Button("Save", variant="primary", id="btn-save")
            yield Button("Reset to Defaults", variant="warning", id="btn-reset")

    def _build_section_tree(self) -> list[list]:
        """build hierarchical section tree from static fields.

        Returns list of [top_section, direct_fields, [(sub_name, sub_fields), ...]]
        """
        grouped: list[tuple[str, list[FieldDef]]] = []
        current_section = None
        current_fields: list[FieldDef] = []
        for f in self._fields:
            if f.section != current_section:
                if current_section is not None:
                    grouped.append((current_section, current_fields))
                current_section = f.section
                current_fields = []
            current_fields.append(f)
        if current_section is not None:
            grouped.append((current_section, current_fields))

        entries: list[list] = []
        entry_map: dict[str, list] = {}
        for section, fields in grouped:
            parts = section.split(".")
            top = parts[0]
            if top not in entry_map:
                entry: list = [top, [], []]
                entry_map[top] = entry
                entries.append(entry)
            e = entry_map[top]
            if len(parts) == 1:
                e[1].extend(fields)
            else:
                sub_name = ".".join(parts[1:])
                e[2].append((sub_name, fields))
        return entries

    def _yield_nested_fields(self, section_name: str, fields: list[FieldDef]) -> ComposeResult:
        """yield FieldRows with nested Collapsibles for subsections."""
        direct = [f for f in fields if f.section == section_name]
        for f in direct:
            initial = self._values.get(f.path, f.default)
            yield FieldRow(f, initial_value=initial)

        sub_groups: dict[str, list[FieldDef]] = {}
        for f in fields:
            if f.section != section_name and f.section.startswith(section_name + "."):
                sub_name = f.section[len(section_name) + 1:]
                sub_groups.setdefault(sub_name, []).append(f)

        for sub_name, sub_flds in sub_groups.items():
            with Collapsible(title=sub_name, collapsed=True):
                for f in sub_flds:
                    initial = self._values.get(f.path, f.default)
                    yield FieldRow(f, initial_value=initial)

    def add_section(self, section_name: str, fields: list[FieldDef], values: dict) -> None:
        """dynamically add a new collapsible section with fields and a remove button."""
        self._dynamic_sections[section_name] = fields
        direct = [f for f in fields if f.section == section_name]
        children: list[Widget] = []
        for f in direct:
            children.append(FieldRow(f, initial_value=values.get(f.path, f.default)))

        sub_groups: dict[str, list[FieldDef]] = {}
        for f in fields:
            if f.section != section_name and f.section.startswith(section_name + "."):
                sub_name = f.section[len(section_name) + 1:]
                sub_groups.setdefault(sub_name, []).append(f)
        for sub_name, sub_flds in sub_groups.items():
            sub_children = [FieldRow(f, initial_value=values.get(f.path, f.default)) for f in sub_flds]
            children.append(Collapsible(*sub_children, title=sub_name, collapsed=True))

        btn_id = _safe_id(f"btn-remove-{section_name}")
        children.append(Button("Remove", variant="error", id=btn_id))
        coll_id = _safe_id(f"dyn-{section_name}")
        collapsible = Collapsible(*children, title=section_name, collapsed=False, id=coll_id)
        form_actions = self.query_one(".form-actions", Horizontal)
        self.mount(collapsible, before=form_actions)

    def remove_section(self, section_name: str) -> None:
        """remove a dynamic section by name."""
        self._dynamic_sections.pop(section_name, None)
        coll_id = _safe_id(f"dyn-{section_name}")
        try:
            self.query_one(f"#{coll_id}").remove()
        except Exception:
            pass

    @property
    def all_fields(self) -> list[FieldDef]:
        """return all fields including dynamic sections."""
        result = list(self._fields)
        for fields in self._dynamic_sections.values():
            result.extend(fields)
        return result

    def collect_values(self) -> dict:
        """gather current values from all field rows and dict-list fields."""
        values = {}
        for row in self.query(FieldRow):
            values[row.field_def.path] = row.current_value
        for dl in self.query(DictListField):
            values[dl.field_def.path] = dl.current_value
        return values

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            values = self.collect_values()
            self.post_message(self.Saved(self.pipeline_name, values))
        elif event.button.id == "btn-reset":
            self._reset_to_defaults()
        elif event.button.id and event.button.id.startswith("btn-remove-"):
            for section_name in list(self._dynamic_sections.keys()):
                if _safe_id(f"btn-remove-{section_name}") == event.button.id:
                    self.remove_section(section_name)
                    break

    def _reset_to_defaults(self) -> None:
        for row in self.query(FieldRow):
            widget_id = _safe_id(f"field__{row.field_def.path}")
            try:
                w = self.query_one(f"#{widget_id}")
            except Exception:
                continue
            if isinstance(w, Switch):
                w.value = row.field_def.default if isinstance(row.field_def.default, bool) else False
            elif isinstance(w, TextArea):
                w.load_text(row._to_display(row.field_def.default))
            elif isinstance(w, Input):
                w.value = row._to_display(row.field_def.default)
