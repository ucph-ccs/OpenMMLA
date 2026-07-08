from __future__ import annotations

import os
import re

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Static, Input, Switch, Button, Collapsible, TextArea, Select, DirectoryTree, Label,
)
from textual.widget import Widget

from openmmla.tui.schema.loader import FieldDef

# media files the file-browser highlights (others are still shown, greyed)
_MEDIA_EXTS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
)


class FileBrowserModal(ModalScreen):
    """A simple local file browser; dismisses with the chosen file path (or None)."""

    DEFAULT_CSS = """
    FileBrowserModal {
        align: center middle;
    }
    FileBrowserModal #fb-box {
        width: 80%;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    FileBrowserModal #fb-tree {
        height: 1fr;
        margin: 1 0;
    }
    FileBrowserModal #fb-actions {
        height: auto;
    }
    FileBrowserModal #fb-actions Button {
        margin-right: 2;
    }
    """

    def __init__(self, start_dir: str | None = None) -> None:
        super().__init__()
        d = start_dir if start_dir and os.path.isdir(start_dir) else os.path.expanduser("~")
        self._start_dir = os.path.abspath(d)
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="fb-box"):
            yield Label(f"Select a file — browsing {self._start_dir}")
            yield DirectoryTree(self._start_dir, id="fb-tree")
            yield Static("", id="fb-pick", classes="dict-entry-hint")
            with Horizontal(id="fb-actions"):
                yield Button("Choose", variant="primary", id="fb-choose", disabled=True)
                yield Button("Cancel", id="fb-cancel")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        self._selected = str(event.path)
        try:
            self.query_one("#fb-pick", Static).update(f"Selected: {self._selected}")
            self.query_one("#fb-choose", Button).disabled = False
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "fb-choose":
            self.dismiss(self._selected)
        else:
            self.dismiss(None)

_SENSITIVE_SUFFIXES = {"api_key", "token", "password", "secret", "secret_key", "subscription_key"}


def _safe_id(raw: str) -> str:
    """sanitize a string to be a valid textual widget id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', raw)


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


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

    def __init__(self, field_def: FieldDef, initial_value=None, source: str | None = None,
                 read_only: bool = False) -> None:
        super().__init__()
        self.field_def = field_def
        self._initial = initial_value if initial_value is not None else field_def.default
        self._source = source
        self._read_only = read_only

    def compose(self) -> ComposeResult:
        short_name = self.field_def.path.split(".")[-1]
        required_marker = " *" if self.field_def.required else ""
        label_text = f"{short_name}{required_marker}"
        if self._source:
            label_text = f"{label_text}  {self._source}"
        widget_id = _safe_id(f"field__{self.field_def.path}")

        yield Static(label_text, classes="field-label")
        if self.field_def.description:
            yield Static(
                self.field_def.description,
                classes="field-desc",
            )
        ro = self._read_only
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
                    disabled=ro,
                )
            else:
                yield Select(
                    options,
                    prompt=self.field_def.description or short_name,
                    allow_blank=True,
                    id=widget_id,
                    disabled=ro,
                )
        elif self.field_def.field_type == "bool":
            val = self._initial if isinstance(self._initial, bool) else False
            yield Switch(value=val, id=widget_id, disabled=ro)
        else:
            display = self._to_display(self._initial)
            sensitive = _is_sensitive_field(self.field_def.path)
            if "\n" in display or len(display) > 120:
                yield TextArea(
                    display,
                    id=widget_id,
                    classes="field-textarea",
                    read_only=ro,
                )
            else:
                yield Input(
                    value=display,
                    placeholder=self.field_def.description or short_name,
                    password=sensitive,
                    id=widget_id,
                    classes="field-input",
                    disabled=ro,
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
    DictListField .dict-entry-hint {
        color: $text-muted;
        padding-left: 24;
        margin-bottom: 1;
    }
    """

    # what source_index means for each source type (drives the dynamic hint
    # and the widget type — see _source_index_widget)
    _SOURCE_INDEX_HINTS = {
        "opencv": "→ which local camera (0-based index)",
        "rtmp": "→ which RTMP stream from Streams (0-based index)",
        "file": "→ pick a video file from file_dir",
        "pyaudio": "→ PyAudio input device index",
        "lsl": "→ LSL stream name (resolved by name)",
        "udp": "→ not used for udp (set 'port' instead)",
        "tcp": "→ not used for tcp (set 'port' instead)",
    }

    @classmethod
    def _source_index_hint(cls, source) -> str:
        return cls._SOURCE_INDEX_HINTS.get(str(source).strip().lower(), "")

    @staticmethod
    def _source_unused(source) -> bool:
        # udp/tcp carry their binding in 'port', not source_index; lsl now uses
        # source_index for the stream name and file uses it for the file pick
        return str(source).strip().lower() in ("udp", "tcp")

    class FileDirChosen(Message):
        """bubbled to ConfigForm when the file browser picks a file, so the
        Base.file_dir field can be updated to the chosen file's directory."""

        def __init__(self, file_dir: str) -> None:
            super().__init__()
            self.file_dir = file_dir

    @staticmethod
    def _list_media_files(dir_path: str) -> list[str]:
        """list media file names in a local directory (for the file dropdown)."""
        if not dir_path or not os.path.isdir(dir_path):
            return []
        try:
            names = sorted(os.listdir(dir_path))
        except OSError:
            return []
        return [n for n in names if n.lower().endswith(_MEDIA_EXTS)]

    def __init__(self, field_def: FieldDef, initial_value=None) -> None:
        super().__init__()
        self.field_def = field_def
        self._entries: list[dict] = list(initial_value or field_def.default or [])
        self._schema: dict = field_def.entry_schema or {}
        # per-key dropdown choices (e.g. {"camera": [...], "base_type": [...]})
        self._choices: dict = field_def.entry_field_choices or {}
        self._next_idx = 0
        # directory last chosen via the file browser (seeds the next browse)
        self._last_file_dir: str | None = None

    def _entry_widget(self, key: str, val, widget_id: str):
        """build a Switch (bool), Select (choices) or Input for one entry field."""
        choices = self._choices.get(key)
        if choices:
            options = [(str(c), str(c)) for c in choices]
            cur = str(val) if val not in (None, "") else None
            valid = cur if cur and any(ov == cur for _, ov in options) else None
            kwargs = {"value": valid} if valid is not None else {}
            return Select(
                options, prompt=f"Select {key}...", id=widget_id,
                classes="dict-entry-input", **kwargs,
            )
        if self._schema.get(key) == "bool":
            return Switch(value=_to_bool(val), id=widget_id, classes="param-toggle")
        return Input(
            value=str(val) if val is not None else "",
            id=widget_id, classes="dict-entry-input",
        )

    def _source_index_widget(self, source, val, widget_id: str):
        """source_index means different things per source type, so the widget
        adapts: file → a dropdown of video files in file_dir; lsl → a text
        input for the stream name; everything else → a plain index/text input
        (disabled for udp/tcp, which bind via 'port')."""
        s = str(source).strip().lower()
        if s == "file":
            files = self._choices.get("source_index") or []
            options = [(str(f), str(f)) for f in files]
            if options:
                cur = str(val) if val not in (None, "") else None
                valid = cur if cur and any(ov == cur for _, ov in options) else None
                kwargs = {"value": valid} if valid is not None else {}
                return Select(
                    options, prompt="Select file...", id=widget_id,
                    classes="dict-entry-input", **kwargs,
                )
            # no files discovered (e.g. file_dir unset/empty/remote) → free text
            w = Input(
                value=str(val) if val is not None else "",
                id=widget_id, classes="dict-entry-input",
            )
            w.placeholder = "filename in file_dir"
            return w
        if s == "lsl":
            w = Input(
                value=str(val) if val is not None else "",
                id=widget_id, classes="dict-entry-input",
            )
            w.placeholder = "LSL stream name"
            return w
        w = Input(
            value=str(val) if val is not None else "",
            id=widget_id, classes="dict-entry-input",
        )
        return w

    def compose(self) -> ComposeResult:
        # for a top-level list (path == section, e.g. Bases) the enclosing
        # Collapsible already shows the name, so skip the redundant inner label
        if "." in self.field_def.path:
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
        src_val = str(entry.get("source", "")) if "source" in self._schema else ""
        with Vertical(classes="dict-entry", id=container_id):
            yield Static(f"Entry {idx + 1}", classes="dict-entry-header")
            for key in self._schema:
                val = entry.get(key, "")
                widget_id = _safe_id(f"dle__{self.field_def.path}__{idx}__{key}")
                with Horizontal(classes="dict-entry-row"):
                    yield Static(f"{key}:", classes="dict-entry-label")
                    if key == "source_index":
                        widget = self._source_index_widget(src_val, val, widget_id)
                    else:
                        widget = self._entry_widget(key, val, widget_id)
                    if key == "source_index" and self._source_unused(src_val):
                        widget.disabled = True
                    yield widget
                    if key == "source_index":
                        browse = Button(
                            "Browse…",
                            id=_safe_id(f"dle-browse__{self.field_def.path}__{idx}"),
                            classes="dle-browse-btn",
                        )
                        browse.disabled = str(src_val).strip().lower() != "file"
                        yield browse
                if key == "source_index" and "source" in self._schema:
                    yield Static(
                        self._source_index_hint(src_val),
                        id=_safe_id(f"{widget_id}__hint"), classes="dict-entry-hint",
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
        src_val = str(defaults.get("source", "")) if "source" in self._schema else ""
        children: list[Widget] = [Static(f"Entry {idx + 1}", classes="dict-entry-header")]
        for key in self._schema:
            val = defaults.get(key, "")
            widget_id = _safe_id(f"dle__{self.field_def.path}__{idx}__{key}")
            if key == "source_index":
                widget = self._source_index_widget(src_val, val, widget_id)
            else:
                widget = self._entry_widget(key, val, widget_id)
            if key == "source_index" and self._source_unused(src_val):
                widget.disabled = True
            row_children: list[Widget] = [Static(f"{key}:", classes="dict-entry-label"), widget]
            if key == "source_index":
                browse = Button(
                    "Browse…",
                    id=_safe_id(f"dle-browse__{self.field_def.path}__{idx}"),
                    classes="dle-browse-btn",
                )
                browse.disabled = str(src_val).strip().lower() != "file"
                row_children.append(browse)
            row = Horizontal(*row_children, classes="dict-entry-row")
            children.append(row)
            if key == "source_index" and "source" in self._schema:
                children.append(Static(
                    self._source_index_hint(src_val),
                    id=_safe_id(f"{widget_id}__hint"), classes="dict-entry-hint",
                ))
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
                    w = container.query_one(f"#{_safe_id(f'{container.id}__{key}')}")
                    if isinstance(w, Switch):
                        entry[key] = bool(w.value)
                    else:
                        raw = w.value
                        if raw is Select.BLANK or raw is None:
                            raw = ""
                        entry[key] = _auto_parse(str(raw).strip())
                except Exception:
                    entry[key] = ""
            result.append(entry)
        return result

    async def on_select_changed(self, event: Select.Changed) -> None:
        """when an entry's 'source' changes, rebuild that entry's source_index
        widget to match the new source type (file → file dropdown, lsl → stream
        name input, others → index input; udp/tcp disabled), and refresh the
        hint."""
        sel_id = event.select.id or ""
        if not sel_id.endswith("__source"):  # only the per-entry source dropdown
            return
        # find the enclosing entry container
        container = event.select
        while container is not None and "dict-entry" not in getattr(container, "classes", ()):
            container = container.parent
        if container is None:
            return
        new_source = "" if event.value is Select.BLANK else str(event.value)

        # preserve the current source_index value where it still makes sense
        cur = ""
        for w in container.query():
            if w.id and w.id.endswith("__source_index"):
                try:
                    v = w.value
                    cur = "" if v is Select.BLANK or v is None else str(v)
                except Exception:
                    cur = ""
                break
        await self._rebuild_source_index(container, new_source, cur)

    @staticmethod
    def _find_in_container(container, suffix):
        for w in container.query():
            if w.id and w.id.endswith(suffix):
                return w
        return None

    async def _rebuild_source_index(self, container, new_source, value) -> None:
        """rebuild an entry's source_index widget to match its source type,
        refresh the hint, and enable the Browse button only for 'file'."""
        unused = self._source_unused(new_source)

        # hint line
        for w in container.query(Static):
            if w.id and w.id.endswith("source_index__hint"):
                w.update(self._source_index_hint(new_source))

        # Browse button is only meaningful for file sources
        for w in container.query(Button):
            if w.id and "dle-browse__" in w.id:
                w.disabled = str(new_source).strip().lower() != "file"

        old = self._find_in_container(container, "__source_index")
        if old is None:
            return
        widget_id = old.id
        row = old.parent
        new_widget = self._source_index_widget(new_source, value, widget_id)
        if unused:
            new_widget.disabled = True
        try:
            # await removal before mounting so the reused id is free (avoids a
            # duplicate-id error on the new widget)
            await old.remove()
            # keep the Browse button last in the row
            browse = self._find_in_container(row, "dle-browse__")
            if browse is not None:
                await row.mount(new_widget, before=browse)
            else:
                await row.mount(new_widget)
        except Exception:
            pass

    def _open_file_browser(self, container) -> None:
        """open the local file browser; on pick, set the entry's source_index to
        the file name, re-list the directory, and update Base.file_dir."""
        start = None
        files = self._choices.get("source_index") or []
        # seed the browser at the previously-listed dir if we can infer it
        if isinstance(self._last_file_dir, str) and os.path.isdir(self._last_file_dir):
            start = self._last_file_dir

        def _done(path: str | None) -> None:
            if not path:
                return
            file_dir = os.path.dirname(path)
            filename = os.path.basename(path)
            self._last_file_dir = file_dir
            # re-list the chosen directory so the dropdown reflects it
            self._choices["source_index"] = self._list_media_files(file_dir)
            self.run_worker(self._rebuild_source_index(container, "file", filename))
            # ask ConfigForm to update the Base.file_dir field
            self.post_message(self.FileDirChosen(file_dir))

        try:
            self.app.push_screen(FileBrowserModal(start), _done)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        add_id = _safe_id(f"dle-add__{self.field_def.path}")
        rm_prefix = _safe_id(f"dle-rm__{self.field_def.path}__")
        browse_prefix = _safe_id(f"dle-browse__{self.field_def.path}__")

        if btn_id == add_id:
            event.stop()
            self.add_entry()
        elif btn_id.startswith(browse_prefix):
            event.stop()
            container = event.button
            while container is not None and "dict-entry" not in getattr(container, "classes", ()):
                container = container.parent
            if container is not None:
                self._open_file_browser(container)
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

    class OverrideToggled(Message):
        def __init__(self, section_name: str) -> None:
            super().__init__()
            self.section_name = section_name

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
        base_section: str | None = None,
        sources: dict | None = None,
        readonly_paths: set | None = None,
        shared_sections: set | None = None,
        overridden_sections: set | None = None,
        allow_override_toggle: bool = False,
    ) -> None:
        super().__init__()
        self.pipeline_name = pipeline_name
        self._fields = fields
        self._values = values or {}
        self._sources = sources or {}
        self._readonly_paths = readonly_paths or set()
        self._shared_sections = shared_sections or set()
        self._overridden_sections = overridden_sections or set()
        self._allow_override_toggle = allow_override_toggle
        self._dynamic_sections: dict[str, list[FieldDef]] = dict(dynamic_sections or {})
        self._group_add_buttons = group_add_buttons or {}
        # dynamic base group (e.g. ASR "Base") is rendered first, before the
        # static sections, so it sits at the top alongside Bases
        self._base_section = base_section

    def _yield_field(self, f: FieldDef) -> ComposeResult:
        """yield the appropriate widget for a single field."""
        initial = self._values.get(f.path, f.default)
        if f.field_type == "list_of_dicts":
            yield DictListField(f, initial_value=initial)
        else:
            yield FieldRow(f, initial_value=initial, source=self._sources.get(f.path),
                           read_only=f.path in self._readonly_paths)

    def _render_dyn_group(self, group_name: str, dyn_groups: dict[str, list[str]]) -> ComposeResult:
        """render one dynamic section group (e.g. ASR 'Base', 'Streams').
        Collapsed by default so the form doesn't open fully expanded."""
        group_coll_id = _safe_id(f"grp-{group_name}")
        inner_id = _safe_id(f"grp-inner-{group_name}")
        with Collapsible(title=group_name, collapsed=True, id=group_coll_id):
            with Vertical(id=inner_id, classes="grp-inner"):
                for section_name in dyn_groups.get(group_name, []):
                    child_name = section_name.split(".", 1)[1]
                    coll_id = _safe_id(f"dyn-{section_name}")
                    btn_id = _safe_id(f"btn-remove-{section_name}")
                    with Collapsible(title=child_name, collapsed=True, id=coll_id):
                        yield from self._yield_nested_fields(section_name, self._dynamic_sections[section_name])
                        yield Button("Remove", variant="error", id=btn_id)
            if group_name in self._group_add_buttons:
                label, bid = self._group_add_buttons[group_name]
                yield Button(label, variant="success", id=bid)

    def compose(self) -> ComposeResult:
        # group dynamic sections (Base.<device>, Streams.<name>) by their group
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

        with VerticalScroll(classes="form-scroll"):
            # the dynamic base group (e.g. ASR 'Base') renders first, at the top
            if self._base_section and self._base_section in dyn_groups:
                yield from self._render_dyn_group(self._base_section, dyn_groups)

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
                        if self._allow_override_toggle and top in self._shared_sections:
                            if top in self._overridden_sections:
                                yield Button(
                                    "Manage centrally (remove override)",
                                    variant="warning",
                                    id=_safe_id(f"btn-ssoverride-{top}"),
                                    classes="ss-override-btn",
                                )
                            else:
                                yield Button(
                                    "Override here (edit on this pipeline)",
                                    variant="primary",
                                    id=_safe_id(f"btn-ssoverride-{top}"),
                                    classes="ss-override-btn",
                                )

            # remaining dynamic groups (e.g. Streams) after the static sections
            for group_name in dyn_group_order:
                if group_name == self._base_section:
                    continue
                yield from self._render_dyn_group(group_name, dyn_groups)

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
        """detect upstream-style sections: named children (dotted paths) that are
        all list_of_dicts (e.g. Server.asr / Server.vfa). A top-level list_of_dicts
        section (e.g. Bases, path == section, no dot) is NOT upstream-style — it
        renders as a plain editable DictListField instead.
        """
        return (
            len(direct) > 0
            and not subs
            and all(f.field_type == "list_of_dicts" for f in direct)
            and all("." in f.path for f in direct)
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
        return FieldRow(f, initial_value=initial, source=self._sources.get(f.path),
                        read_only=f.path in self._readonly_paths)

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

    def on_dict_list_field_file_dir_chosen(self, event: "DictListField.FileDirChosen") -> None:
        """the file browser picked a file under event.file_dir — reflect it in the
        Base.file_dir field so source resolution finds the file at runtime."""
        event.stop()
        for row in self.query(FieldRow):
            if row.field_def.path.endswith(".file_dir"):
                try:
                    w = self.query_one(f"#{_safe_id(f'field__{row.field_def.path}')}")
                    if isinstance(w, Input):
                        w.value = event.file_dir
                except Exception:
                    pass
                break

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
        elif event.button.id and event.button.id.startswith("btn-ssoverride-"):
            for section_name in self._shared_sections:
                if _safe_id(f"btn-ssoverride-{section_name}") == event.button.id:
                    event.stop()
                    self.post_message(self.OverrideToggled(section_name))
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
