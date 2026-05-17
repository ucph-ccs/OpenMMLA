from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, Input, Switch, Button, Collapsible, TextArea, Select
from textual.widget import Widget

from openmmla.tui.schema.loader import FieldDef

_SENSITIVE_SUFFIXES = {"api_key", "token", "password", "secret", "secret_key", "subscription_key"}


def _safe_id(raw: str) -> str:
    """sanitize a string to be a valid textual widget id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw)


def _is_sensitive_field(path: str) -> bool:
    """check if a field path ends with a sensitive key name."""
    last = path.rsplit(".", 1)[-1].lower()
    return last in _SENSITIVE_SUFFIXES


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
        if self.field_def.choices:
            options = [(c, c) for c in self.field_def.choices]
            initial = str(self._initial) if self._initial else ""
            if initial and initial in self.field_def.choices:
                yield Select(
                    options,
                    value=initial,
                    prompt=self.field_def.description or short_name,
                    allow_blank=True,
                    id=widget_id,
                )
            else:
                yield Select(
                    options,
                    prompt=self.field_def.description or short_name,
                    allow_blank=True,
                    id=widget_id,
                )
        elif self.field_def.field_type == "bool":
            val = self._initial if isinstance(self._initial, bool) else False
            yield Switch(value=val, id=widget_id)
        else:
            display = self._to_display(self._initial)
            sensitive = _is_sensitive_field(self.field_def.path)
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
                    password=sensitive,
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

        if isinstance(w, Select):
            val = w.value
            if val is Select.BLANK or val is getattr(Select, "NULL", None) or val is None:
                return self.field_def.default
            return str(val)

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
    return _auto_parse(raw)


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


class _UpstreamService(Widget):
    """a single upstream service with editable name, entries, and remove button."""

    DEFAULT_CSS = """
    _UpstreamService {
        height: auto;
        padding: 0 1;
    }
    .ue-name-row {
        height: auto;
        margin-bottom: 1;
    }
    """

    def __init__(self, svc_id: str, name: str, entries: list[dict], schema: dict) -> None:
        super().__init__(id=svc_id)
        self._name = name
        self._entries = list(entries) if entries else []
        self._schema = schema
        self._entry_idx = 0

    def compose(self) -> ComposeResult:
        with Horizontal(classes="ue-name-row"):
            yield Static("name:", classes="dict-entry-label")
            yield Input(value=self._name, id=f"{self.id}--name", classes="dict-entry-input")
        for entry in self._entries:
            yield from self._build_entry(entry)
        yield Button("+ Add Entry", variant="success", id=f"{self.id}--add", classes="dle-add-btn")
        yield Button("Remove Service", variant="error", id=f"{self.id}--rm")

    def _build_entry(self, entry: dict) -> ComposeResult:
        eidx = self._entry_idx
        self._entry_idx += 1
        eid = f"{self.id}--e{eidx}"
        with Vertical(classes="dict-entry", id=eid):
            yield Static(f"Entry {eidx + 1}", classes="dict-entry-header")
            for key in self._schema:
                val = entry.get(key, "")
                with Horizontal(classes="dict-entry-row"):
                    yield Static(f"{key}:", classes="dict-entry-label")
                    yield Input(
                        value=str(val) if val is not None else "",
                        id=f"{eid}--{_safe_id(key)}",
                        classes="dict-entry-input",
                    )
            yield Button("Remove", variant="error", id=f"{eid}--rm", classes="dle-rm-btn")

    def add_entry(self) -> None:
        eidx = self._entry_idx
        self._entry_idx += 1
        eid = f"{self.id}--e{eidx}"
        children: list[Widget] = [Static(f"Entry {eidx + 1}", classes="dict-entry-header")]
        for key in self._schema:
            val = "" if self._schema[key] == "str" else 0
            children.append(Horizontal(
                Static(f"{key}:", classes="dict-entry-label"),
                Input(value=str(val), id=f"{eid}--{_safe_id(key)}", classes="dict-entry-input"),
                classes="dict-entry-row",
            ))
        children.append(Button("Remove", variant="error", id=f"{eid}--rm", classes="dle-rm-btn"))
        container = Vertical(*children, classes="dict-entry", id=eid)
        add_btn = self.query_one(f"#{self.id}--add", Button)
        self.mount(container, before=add_btn)

    @property
    def service_name(self) -> str:
        try:
            return self.query_one(f"#{self.id}--name", Input).value.strip()
        except Exception:
            return self._name

    @property
    def entries(self) -> list[dict]:
        result = []
        for container in self.query(".dict-entry"):
            cid = container.id or ""
            if not cid.startswith(f"{self.id}--e"):
                continue
            entry = {}
            for key in self._schema:
                try:
                    inp = container.query_one(f"#{cid}--{_safe_id(key)}", Input)
                    entry[key] = _auto_parse(inp.value.strip())
                except Exception:
                    entry[key] = ""
            result.append(entry)
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == f"{self.id}--add":
            event.stop()
            self.add_entry()
        elif btn_id.startswith(f"{self.id}--e") and btn_id.endswith("--rm"):
            event.stop()
            entry_id = btn_id[:-4]
            try:
                self.query_one(f"#{entry_id}").remove()
            except Exception:
                pass


class UpstreamsEditor(Widget):
    """editor for a group of upstream services with nested collapsibles."""

    DEFAULT_CSS = """
    UpstreamsEditor {
        height: auto;
    }
    """

    def __init__(self, section_key: str, services: dict[str, list[dict]], entry_schema: dict) -> None:
        super().__init__()
        self.section_key = section_key
        self._services = dict(services)
        self._schema = entry_schema
        self._svc_idx = 0

    def compose(self) -> ComposeResult:
        for svc_name, entries in self._services.items():
            yield from self._build_service(svc_name, entries)
        yield Button("+ Add Upstream", variant="success", id="ue-add-svc")

    def _build_service(self, name: str, entries: list[dict]) -> ComposeResult:
        svc_id = f"ue-svc-{self._svc_idx}"
        self._svc_idx += 1
        svc = _UpstreamService(svc_id, name, entries, self._schema)
        yield Collapsible(svc, title=name, collapsed=True, id=f"{svc_id}-coll")

    def _add_service(self) -> None:
        svc_id = f"ue-svc-{self._svc_idx}"
        self._svc_idx += 1
        svc = _UpstreamService(svc_id, "", [], self._schema)
        coll = Collapsible(svc, title="(new)", collapsed=False, id=f"{svc_id}-coll")
        add_btn = self.query_one("#ue-add-svc", Button)
        self.mount(coll, before=add_btn)

    @property
    def current_value(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for svc in self.query(_UpstreamService):
            name = svc.service_name
            if name:
                result[name] = svc.entries
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "ue-add-svc":
            event.stop()
            self._add_service()
        elif btn_id.endswith("--rm") and "--e" not in btn_id:
            event.stop()
            svc_id = btn_id[:-4]
            try:
                self.query_one(f"#{svc_id}-coll").remove()
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
        height: 1fr;
        padding: 1 2;
    }
    .form-scroll {
        height: 1fr;
    }
    .grp-inner {
        height: auto;
    }
    .form-actions {
        height: auto;
        padding: 1 0;
    }
    """

    def __init__(
        self,
        pipeline_name: str,
        fields: list[FieldDef],
        values: dict | None = None,
        dynamic_sections: dict[str, list[FieldDef]] | None = None,
        group_add_buttons: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self.pipeline_name = pipeline_name
        self._fields = fields
        self._values = values or {}
        self._dynamic_sections: dict[str, list[FieldDef]] = dict(dynamic_sections or {})
        self._group_add_buttons = group_add_buttons or {}

    def _yield_field(self, f: FieldDef) -> ComposeResult:
        """yield the appropriate widget for a single field."""
        initial = self._values.get(f.path, f.default)
        if f.field_type == "list_of_dicts":
            yield DictListField(f, initial_value=initial)
        else:
            yield FieldRow(f, initial_value=initial)

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="form-scroll"):
            tree = self._build_section_tree()
            for top, direct, subs in tree:
                if self._is_upstream_style(direct, subs):
                    services: dict[str, list[dict]] = {}
                    schema: dict = {}
                    for f in direct:
                        svc_name = f.path.split(".")[-1]
                        services[svc_name] = self._values.get(f.path, f.default) or []
                        if not schema and f.entry_schema:
                            schema = dict(f.entry_schema)
                    for k, v in self._values.items():
                        if k.startswith(f"{top}.") and isinstance(v, list):
                            svc_name = k.split(".", 1)[1]
                            if svc_name not in services:
                                services[svc_name] = v
                    with Collapsible(title=top, collapsed=True):
                        yield UpstreamsEditor(top, services, schema)
                else:
                    nested_subs = self._nest_subs(subs)
                    with Collapsible(title=top, collapsed=True):
                        for f in direct:
                            yield from self._yield_field(f)
                        yield from self._render_nested_subs(nested_subs)

            dyn_groups: dict[str, list[str]] = {}
            dyn_group_order: list[str] = []
            ungrouped: list[str] = []
            for section_name in self._dynamic_sections:
                parts = section_name.split(".", 1)
                if len(parts) > 1:
                    group = parts[0]
                    if group not in dyn_groups:
                        dyn_groups[group] = []
                        dyn_group_order.append(group)
                    dyn_groups[group].append(section_name)
                else:
                    ungrouped.append(section_name)

            for grp in self._group_add_buttons:
                if grp not in dyn_groups:
                    dyn_groups[grp] = []
                    dyn_group_order.append(grp)

            for group_name in dyn_group_order:
                group_coll_id = _safe_id(f"grp-{group_name}")
                inner_id = _safe_id(f"grp-inner-{group_name}")
                with Collapsible(title=group_name, collapsed=False, id=group_coll_id):
                    with Vertical(id=inner_id, classes="grp-inner"):
                        for section_name in dyn_groups[group_name]:
                            child_name = section_name.split(".", 1)[1]
                            coll_id = _safe_id(f"dyn-{section_name}")
                            btn_id = _safe_id(f"btn-remove-{section_name}")
                            with Collapsible(title=child_name, collapsed=True, id=coll_id):
                                yield from self._yield_nested_fields(section_name, self._dynamic_sections[section_name])
                                yield Button("Remove", variant="error", id=btn_id)
                    if group_name in self._group_add_buttons:
                        label, bid = self._group_add_buttons[group_name]
                        yield Button(label, variant="success", id=bid)

            for section_name in ungrouped:
                coll_id = _safe_id(f"dyn-{section_name}")
                btn_id = _safe_id(f"btn-remove-{section_name}")
                with Collapsible(title=section_name, collapsed=True, id=coll_id):
                    yield from self._yield_nested_fields(section_name, self._dynamic_sections[section_name])
                    yield Button("Remove", variant="error", id=btn_id)

        with Horizontal(classes="form-actions"):
            yield Button("Save", variant="primary", id="btn-save")
            yield Button("Reset to Defaults", variant="warning", id="btn-reset")

    @staticmethod
    def _is_upstream_style(direct: list[FieldDef], subs: list) -> bool:
        """detect sections where all direct fields are list_of_dicts with no subsections."""
        return (
            len(direct) > 0
            and not subs
            and all(f.field_type == "list_of_dicts" for f in direct)
        )

    @staticmethod
    def _nest_subs(subs: list[tuple]) -> list[tuple]:
        """group flat sub-sections into a recursive tree.

        Input:  [("vllm", fields), ("vllm.VLMExtraBody", fields), ("ollama", fields)]
        Output: [("vllm", fields, [("VLMExtraBody", fields, [])]), ("ollama", fields, [])]
        """
        groups: dict[str, list] = {}
        order: list[str] = []
        for sub_name, sub_fields in subs:
            parts = sub_name.split(".", 1)
            top = parts[0]
            if top not in groups:
                groups[top] = [[], []]
                order.append(top)
            if len(parts) == 1:
                groups[top][0] = sub_fields
            else:
                groups[top][1].append((parts[1], sub_fields))
        result = []
        for name in order:
            fields, children_flat = groups[name]
            nested = ConfigForm._nest_subs(children_flat) if children_flat else []
            result.append((name, fields, nested))
        return result

    def _render_nested_subs(self, subs: list[tuple]) -> ComposeResult:
        """recursively render nested sub-section collapsibles."""
        for name, fields, children in subs:
            with Collapsible(title=name, collapsed=True):
                for f in fields:
                    yield from self._yield_field(f)
                yield from self._render_nested_subs(children)

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
        """yield field widgets with nested Collapsibles for subsections."""
        direct = [f for f in fields if f.section == section_name]
        for f in direct:
            yield from self._yield_field(f)

        sub_groups: dict[str, list[FieldDef]] = {}
        for f in fields:
            if f.section != section_name and f.section.startswith(section_name + "."):
                sub_name = f.section[len(section_name) + 1:]
                sub_groups.setdefault(sub_name, []).append(f)

        for sub_name, sub_flds in sub_groups.items():
            with Collapsible(title=sub_name, collapsed=True):
                for f in sub_flds:
                    yield from self._yield_field(f)

    def _make_field_widget(self, f: FieldDef) -> Widget:
        """create the appropriate widget for a single field."""
        initial = self._values.get(f.path, f.default)
        if f.field_type == "list_of_dicts":
            return DictListField(f, initial_value=initial)
        return FieldRow(f, initial_value=initial)

    def add_section(self, section_name: str, fields: list[FieldDef], values: dict) -> None:
        """dynamically add a new collapsible section with fields and a remove button."""
        self._dynamic_sections[section_name] = fields
        for f in fields:
            if f.path in values:
                self._values[f.path] = values[f.path]
        direct = [f for f in fields if f.section == section_name]
        children: list[Widget] = []
        for f in direct:
            children.append(self._make_field_widget(f))

        sub_groups: dict[str, list[FieldDef]] = {}
        for f in fields:
            if f.section != section_name and f.section.startswith(section_name + "."):
                sub_name = f.section[len(section_name) + 1:]
                sub_groups.setdefault(sub_name, []).append(f)
        for sub_name, sub_flds in sub_groups.items():
            sub_children = [self._make_field_widget(f) for f in sub_flds]
            children.append(Collapsible(*sub_children, title=sub_name, collapsed=True))

        btn_id = _safe_id(f"btn-remove-{section_name}")
        children.append(Button("Remove", variant="error", id=btn_id))
        coll_id = _safe_id(f"dyn-{section_name}")

        parts = section_name.split(".", 1)
        if len(parts) > 1:
            child_name = parts[1]
            inner_id = _safe_id(f"grp-inner-{parts[0]}")
            collapsible = Collapsible(*children, title=child_name, collapsed=False, id=coll_id)
            try:
                inner = self.query_one(f"#{inner_id}", Vertical)
                inner.mount(collapsible)
                return
            except Exception:
                pass

        collapsible = Collapsible(*children, title=section_name, collapsed=False, id=coll_id)
        try:
            scroll = self.query_one(".form-scroll", VerticalScroll)
            scroll.mount(collapsible)
        except Exception:
            self.mount(collapsible)

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
        """return all fields including dynamic sections and upstream editors."""
        ue_sections = {ue.section_key for ue in self.query(UpstreamsEditor)}
        result = [f for f in self._fields if not any(f.path.startswith(s + ".") for s in ue_sections)]
        for fields in self._dynamic_sections.values():
            result.extend(fields)
        for ue in self.query(UpstreamsEditor):
            for svc_name in ue.current_value:
                result.append(FieldDef(
                    path=f"{ue.section_key}.{svc_name}",
                    field_type="list_of_dicts",
                    default=[],
                    description="",
                    required=False,
                    section=ue.section_key,
                ))
        return result

    def collect_values(self) -> dict:
        """gather current values from all field rows, dict-list fields, and upstream editors."""
        values = {}
        for row in self.query(FieldRow):
            values[row.field_def.path] = row.current_value
        for dl in self.query(DictListField):
            values[dl.field_def.path] = dl.current_value
        for ue in self.query(UpstreamsEditor):
            for svc_name, entries in ue.current_value.items():
                values[f"{ue.section_key}.{svc_name}"] = entries
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
            elif isinstance(w, Select):
                w.clear()
            elif isinstance(w, TextArea):
                w.load_text(row._to_display(row.field_def.default))
            elif isinstance(w, Input):
                w.value = row._to_display(row.field_def.default)
