from __future__ import annotations

import os
import re
from typing import Callable

import yaml

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Static, Input, Switch, Button, Collapsible, TextArea, Select, DirectoryTree, Label,
)
from textual.widget import Widget

from rich.text import Text

from openmmla.tui.schema.loader import REMOVED_SECTION, FieldDef
from openmmla.utils.constants import normalize_source

# media files the file-browser highlights (others are still shown, greyed)
_MEDIA_EXTS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm",
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac",
)


def _no_choice(value) -> bool:
    """a Select with nothing picked: Select.NULL on Textual 8 (where
    Select.BLANK is False), Select.BLANK before it."""
    return value is None or value is Select.BLANK or value is getattr(Select, "NULL", None)


def _media_files(folder: str) -> list[str]:
    """the media files of a folder on this machine, by name; none when it
    cannot be listed."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [name for name in names
            if name.lower().endswith(_MEDIA_EXTS) and os.path.isfile(os.path.join(folder, name))]


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


# the description stands above the field once: it used to be repeated as the
# first row of every dropdown and as the placeholder of every empty input
_EMPTY_CHOICE = "(empty)"


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
            yield self._select(self.field_def.choices, self._initial, widget_id)
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
                    placeholder="" if self.field_def.description else short_name,
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

    def _select(self, choices: list, initial, widget_id: str) -> Select:
        """the dropdown of a field with choices: plain values, or (label, value)
        pairs. A stored value the list does not offer (an SSH profile that was
        renamed or deleted, a device not detected now) is shown and kept: it
        used to open blank, and the next Save wrote the blank over it."""
        options = [(str(c[0]), str(c[1])) if isinstance(c, (tuple, list)) else (str(c), str(c)) for c in choices]
        initial = str(initial) if initial not in (None, "") else ""
        if initial in ("Select.NULL", "Select.BLANK", "None", "null"):
            initial = ""  # what an empty Select once left behind in a config
        if initial and not any(value == initial for _, value in options):
            options.append((f"{initial}  (not in the list now)", initial))
        kwargs = {"value": initial} if initial else {}
        return Select(options, prompt=_EMPTY_CHOICE, allow_blank=True, id=widget_id, disabled=self._read_only,
                      **kwargs)

    def set_description(self, text: str) -> None:
        """the line under the label, for a description that follows the value."""
        self.field_def.description = text
        try:
            self.query_one(".field-desc", Static).update(text)
        except Exception:
            # composed without one: the description was empty then
            try:
                self.mount(Static(text, classes="field-desc"), after=self.query_one(".field-label"))
            except Exception:
                pass

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
        if "[" in raw:
            # a list of lists, e.g. a camera's K: "[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]".
            # Split at the commas it used to come back as strings like '[fx' and 'cx]'
            # (and that is also how a config saved then reads now: joined, it is parsed right)
            try:
                nested = yaml.safe_load(f"[{raw}]")
            except yaml.YAMLError:
                nested = None
            if isinstance(nested, list):
                return nested
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

    # what source_index is for each source: the hint under its row, which
    # follows the source dropdown
    _SOURCE_INDEX_HINTS = {
        "opencv": "→ which local camera (0-based index)",
        "stream": "→ which stream of this config's Streams it pulls, by name (a new one is listed once the Streams are saved)",
        "file": "→ the file to replay, by its full path (Browse…); the list holds the other files of its folder",
        "pyaudio": "→ the input device, by PyAudio's index of it",
        "lsl": "→ LSL stream name (resolved by name)",
    }
    # the sources whose source_index is a device of the card's host, offered
    # in a dropdown once the host has said what it has (set_source_choices)
    _DEVICE_SOURCES = ("pyaudio", "opencv")
    # the fields of a Bases entry that only some sources use, and which: the
    # row of one the entry's source does not use is hidden, and left out of
    # the saved entry. A field not listed here is for every source. This is
    # what the bases read: asr_base takes port, host and packet_format for
    # udp/tcp, channel_select for pyaudio and source_index for the rest; the
    # video bases source_index alone
    _FIELDS_BY_SOURCE = {
        "source_index": {"pyaudio", "opencv", "stream", "lsl", "file"},
        "channel_select": {"pyaudio"},
        "port": {"udp", "tcp"},
        "host": {"udp", "tcp"},
        "packet_format": {"udp", "tcp"},
    }
    # what a base takes for a field left empty, shown in its row so the entry
    # says what runs (and saved with it)
    _FIELD_DEFAULTS = {"host": "0.0.0.0", "packet_format": "auto"}
    _FIELD_HINTS = {
        "port": "→ the port this base listens on: the badge or FFmpeg stream pushes to it",
        "host": "→ the address it listens on: 0.0.0.0 is every interface",
        "packet_format": "→ auto: the first packet tells; timestamped: a badge (18-byte header + PCM); "
                         "raw: header-less PCM, as an FFmpeg stream sends",
        "channel_select": "→ which channel of a multi-channel device this base keeps; empty: all of them "
                          "(stream_kwargs.channels is how many the device has)",
        "camera_angle": "→ what this camera sees, by a name of Base.angle_config (a new one is listed once the "
                        "Base is saved)",
    }
    # a field's earlier name: an entry that still holds it shows its value under
    # the new name, and Save writes the new name alone
    _RENAMED_FIELDS = {"channel_select": "channel"}

    @classmethod
    def _source_index_hint(cls, source) -> str:
        return cls._SOURCE_INDEX_HINTS.get(normalize_source(source), "")

    def _applies(self, key: str, source) -> bool:
        """whether an entry with `source` uses the field `key` (every field
        does in a list whose entries have no source)."""
        if "source" not in self._schema:
            return True
        sources = self._FIELDS_BY_SOURCE.get(key)
        return sources is None or normalize_source(source) in sources

    def _field_hint(self, key: str, source) -> str | None:
        """the hint under the row of `key`; None for a row that has none."""
        if "source" not in self._schema:
            return None
        if key == "source_index":
            return self._hint(source)
        return self._FIELD_HINTS.get(key)

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
        # what the host said about its devices, under the pyaudio and opencv rows
        self._device_notes: dict[str, str] = {}
        # what the Stream Server says of the pullable streams: name -> "live" |
        # "idle", and a line for below the stream entries (see set_stream_states)
        self._stream_states: dict[str, str] = {}
        self._stream_note = ""

    def _entry_widget(self, key: str, val, widget_id: str):
        """build a Switch (bool), Select (choices) or Input for one entry field."""
        choices = self._choices.get(key)
        if choices:
            options = [(str(c), str(c)) for c in choices]
            cur = str(val) if val not in (None, "") else None
            if cur and not any(ov == cur for _, ov in options):
                # a value the list no longer offers (a camera renamed, an angle
                # taken out of angle_config) is shown and kept: an entry that
                # opened blank used to be saved blank
                options.append((f"{cur}  (not in the list now)", cur))
            kwargs = {"value": cur} if cur else {}
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

    def _files_here(self) -> bool:
        """whether the files of the card's host are this machine's, so that a
        file entry's folder can be listed and browsed from here."""
        return bool(self._choices.get("source_index:file_here", True))

    def _source_index_widget(self, source, val, widget_id: str):
        """source_index means different things per source type, so the widget
        adapts: file → a dropdown of the files in the folder of the entry's
        file; lsl → a text input for the stream name; everything else → a
        plain index/text input (disabled for udp/tcp, which bind via 'port')."""
        s = normalize_source(source)
        if s == "stream":
            streams, cur = self._stream_options(str(val).strip() if val not in (None, "") else "")
            if streams:
                kwargs = {"value": cur} if cur else {}
                return Select(
                    streams, prompt="Select stream...", id=widget_id,
                    classes="dict-entry-input", **kwargs,
                )
            w = Input(
                value=str(val) if val is not None else "",
                id=widget_id, classes="dict-entry-input",
            )
            w.placeholder = "no pullable stream in Streams yet: add one there and Save"
            return w
        if s == "file":
            cur = str(val).strip() if val not in (None, "") else ""
            if cur and self._files_here():
                # the entry's file among the other files of its folder; the file
                # itself is named with its folder, and kept when it is not there
                folder = os.path.dirname(cur) if os.path.isabs(cur) else ""
                options = [(f"{name}  ({folder})" if os.path.join(folder, name) == cur else name,
                            os.path.join(folder, name)) for name in (_media_files(folder) if folder else [])]
                if not any(value == cur for _, value in options):
                    options.insert(0, (f"{os.path.basename(cur)}  ({folder})" if folder else cur, cur))
                return Select(options, prompt="Select file...", id=widget_id,
                              classes="dict-entry-input", value=cur)
            # nothing picked yet, or a file on another machine: its path, typed
            w = Input(value=cur, id=widget_id, classes="dict-entry-input")
            w.placeholder = ("full path of the file (Browse…)" if self._files_here()
                             else "full path of the file on the card's host")
            return w
        if s == "lsl":
            w = Input(
                value=str(val) if val is not None else "",
                id=widget_id, classes="dict-entry-input",
            )
            w.placeholder = "LSL stream name"
            return w
        if s in self._DEVICE_SOURCES:
            options = [(str(label), str(value)) for label, value in (self._choices.get(f"source_index:{s}") or [])]
            cur = str(val).strip() if val not in (None, "") else ""
            if options:
                if cur and not any(value == cur for _, value in options):
                    options.append((f"{cur}  (not detected now)", cur))  # kept rather than dropped
                kwargs = {"value": cur} if cur else {}
                return Select(options, prompt="Select device...", id=widget_id, classes="dict-entry-input", **kwargs)
        w = Input(
            value=str(val) if val is not None else "",
            id=widget_id, classes="dict-entry-input",
        )
        return w

    def set_source_choices(self, source: str, options: list[tuple[str, str]], note: str = "") -> None:
        """the devices of the card's host for the entries whose source is
        `source` (pyaudio, opencv): their source_index becomes a dropdown of
        them, what it showed kept, with `note` under it."""
        source = normalize_source(source)
        self._choices[f"source_index:{source}"] = [(str(label), str(value)) for label, value in options]
        self._device_notes[source] = note
        for container in self.query(".dict-entry"):
            picked = self._find_in_container(container, "__source")
            value = getattr(picked, "value", "") if picked is not None else ""
            if value is Select.BLANK or normalize_source(value) != source:
                continue
            widget = self._find_in_container(container, "__source_index")
            current = ""
            if widget is not None:
                raw = getattr(widget, "value", "")
                current = "" if raw in (None, Select.BLANK) else str(raw)
            self.run_worker(self._rebuild_source_index(container, source, current), exclusive=False)

    @property
    def stream_choices(self) -> list[tuple[str, str]]:
        """(name, url) of the streams a 'stream' entry can pull."""
        return list(self._choices.get("source_index:stream") or [])

    def _stream_label(self, name: str, url: str):
        state = self._stream_states.get(name)
        if state == "live":
            return Text.assemble(("● ", "green"), f"{name}  ({url})")
        if state == "idle":
            return Text.assemble(("○ ", "dim"), f"{name}  ({url})", ("  not publishing now", "dim"))
        return f"{name}  ({url})"

    def _stream_options(self, cur: str) -> tuple[list, str]:
        """the options of a stream dropdown and the value it shows. The
        pullable Streams entries come as (name, url); a base finds its stream
        by name, so the name is what is stored."""
        pullable = self.stream_choices
        options = [(self._stream_label(name, url), name) for name, url in pullable]
        names = [name for name, _url in pullable]
        if cur and cur not in names:
            # what a base also reads: an index (older configs), a URL or its last segment
            same = [name for i, (name, url) in enumerate(pullable)
                    if cur in (str(i), url, url.rstrip("/").rsplit("/", 1)[-1])]
            if same:
                cur = same[0]
            else:
                # kept rather than dropped on the next Save
                options.append((f"{cur}  (not a pullable stream of this config)", cur))
        return options, cur

    def _hint(self, source) -> str:
        text = self._source_index_hint(source)
        if normalize_source(source) == "file" and not self._files_here():
            text = "→ the file to replay, by its full path on the card's host"
        if normalize_source(source) == "stream" and self._stream_note:
            text = f"{text}\n{self._stream_note}"
        note = self._device_notes.get(normalize_source(source))
        if note:
            text = f"{text}\n{note}"
        return text

    def set_stream_states(self, states: dict[str, str], note: str = "") -> None:
        """what the Stream Server says, in the dropdowns of the entries that
        pull a stream (● live, ○ not publishing now) and below them (it does
        not answer, or has live paths no Streams entry names)."""
        self._stream_states = dict(states)
        self._stream_note = note
        for container in self.query(".dict-entry"):
            source = self._find_in_container(container, "__source")
            value = getattr(source, "value", "") if source is not None else ""
            if value is Select.BLANK or normalize_source(value) != "stream":
                continue
            widget = self._find_in_container(container, "__source_index")
            if isinstance(widget, Select):
                self._show_stream_states_in(widget)
            for hint in container.query(Static):
                if hint.id and hint.id.endswith("source_index__hint"):
                    hint.update(self._hint("stream"))

    def _show_stream_states_in(self, widget: Select, current: str | None = None, tries: int = 5) -> None:
        """the live marks in one stream dropdown. The Server's answer can come
        while the dropdown is mounted but not yet drawn (a card just opened, or
        another one picked), when it has no label to set: it gets them on the
        next refresh instead, keeping the stream it showed."""
        if current is None:
            value = widget.value
            current = "" if value in (None, Select.BLANK, getattr(Select, "NULL", None)) else str(value)
        options, cur = self._stream_options(current)
        try:
            widget.set_options(options)
            if cur:
                widget.value = cur
        except NoMatches:
            if tries > 0 and widget.is_attached:
                self.call_after_refresh(self._show_stream_states_in, widget, current, tries - 1)

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

    def _entry_children(self, entry: dict, idx: int) -> list[Widget]:
        """the widgets of one entry: a row per field of the schema, with a
        hint under source_index, port and channel. A field the entry's source
        does not use is built hidden, and shown once a source that uses it is
        picked (_show_fields_for)."""
        src_val = str(entry.get("source", "")) if "source" in self._schema else ""
        children: list[Widget] = [Static(f"Entry {idx + 1}", classes="dict-entry-header")]
        for key in self._schema:
            val = entry.get(key, "")
            if val in ("", None) and key in self._RENAMED_FIELDS:
                val = entry.get(self._RENAMED_FIELDS[key], "")
            if val in ("", None) and key in self._FIELD_DEFAULTS and "source" in self._schema:
                # a Bases entry's; another list's host (an upstream) has no default
                val = self._FIELD_DEFAULTS[key]
            widget_id = _safe_id(f"dle__{self.field_def.path}__{idx}__{key}")
            if key == "source_index":
                widget = self._source_index_widget(src_val, val, widget_id)
            else:
                widget = self._entry_widget(key, val, widget_id)
            row_children: list[Widget] = [Static(f"{key}:", classes="dict-entry-label"), widget]
            if key == "source_index":
                browse = Button(
                    "Browse…",
                    id=_safe_id(f"dle-browse__{self.field_def.path}__{idx}"),
                    classes="dle-browse-btn",
                )
                browse.disabled = normalize_source(src_val) != "file" or not self._files_here()
                row_children.append(browse)
            shown = self._applies(key, src_val)
            row = Horizontal(*row_children, id=_safe_id(f"{widget_id}__row"), classes="dict-entry-row")
            row.display = shown
            children.append(row)
            hint = self._field_hint(key, src_val)
            if hint is not None:
                hint_widget = Static(hint, id=_safe_id(f"{widget_id}__hint"), classes="dict-entry-hint")
                hint_widget.display = shown
                children.append(hint_widget)
        rm_id = _safe_id(f"dle-rm__{self.field_def.path}__{idx}")
        children.append(Button("Remove", variant="error", id=rm_id, classes="dle-rm-btn"))
        return children

    def _build_entry(self, entry: dict) -> ComposeResult:
        idx = self._next_idx
        self._next_idx += 1
        if "source" in self._schema and str(entry.get("source", "")).strip():
            # 'rtmp' is the old name of 'stream': shown, and saved, as stream
            entry = dict(entry, source=normalize_source(entry.get("source")))
        yield Vertical(*self._entry_children(entry, idx), classes="dict-entry",
                       id=_safe_id(f"dle__{self.field_def.path}__{idx}"))

    def add_entry(self) -> None:
        """mount a new empty entry before the add button."""
        defaults = {key: ("" if typ == "str" else 0) for key, typ in self._schema.items()}
        idx = self._next_idx
        self._next_idx += 1
        container = Vertical(*self._entry_children(defaults, idx), classes="dict-entry",
                             id=_safe_id(f"dle__{self.field_def.path}__{idx}"))
        add_btn = self.query_one(f"#{_safe_id(f'dle-add__{self.field_def.path}')}", Button)
        self.mount(container, before=add_btn)

    def _show_fields_for(self, container, source) -> None:
        """show the rows (with their hints) of the fields `source` uses in
        one entry, and hide the rest."""
        for key in self._FIELDS_BY_SOURCE:
            if key not in self._schema:
                continue
            shown = self._applies(key, source)
            for suffix in (f"__{key}__row", f"__{key}__hint"):
                widget = self._find_in_container(container, suffix)
                if widget is not None:
                    widget.display = shown

    @property
    def current_value(self) -> list[dict]:
        """collect all remaining entry dicts from the widget tree."""
        result = []
        prefix = _safe_id(f"dle__{self.field_def.path}__")
        for container in self.query(".dict-entry"):
            if not container.id or not container.id.startswith(prefix):
                continue
            read = {}
            for key in self._schema:
                try:
                    w = container.query_one(f"#{_safe_id(f'{container.id}__{key}')}")
                    if isinstance(w, Switch):
                        read[key] = bool(w.value)
                    else:
                        raw = w.value
                        if _no_choice(raw):
                            raw = ""
                        read[key] = _auto_parse(str(raw).strip())
                except Exception:
                    read[key] = ""
            # a field the entry's source does not use stays out of the file
            source = read.get("source", "")
            result.append({key: value for key, value in read.items() if self._applies(key, source)})
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
        new_source = "" if _no_choice(event.value) else str(event.value)
        self._show_fields_for(container, new_source)

        # preserve the current source_index value where it still makes sense
        cur = ""
        for w in container.query():
            if w.id and w.id.endswith("__source_index"):
                try:
                    v = w.value
                    cur = "" if _no_choice(v) else str(v)
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
        # hint line
        for w in container.query(Static):
            if w.id and w.id.endswith("source_index__hint"):
                w.update(self._hint(new_source))

        # Browse button is only meaningful for file sources, on this machine
        for w in container.query(Button):
            if w.id and "dle-browse__" in w.id:
                w.disabled = str(new_source).strip().lower() != "file" or not self._files_here()

        old = self._find_in_container(container, "__source_index")
        if old is None:
            return
        widget_id = old.id
        row = old.parent
        new_widget = self._source_index_widget(new_source, value, widget_id)
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
        """open the local file browser; on pick, the entry's source_index is
        the file's full path, and the dropdown lists the other files of its
        folder."""
        start = None
        # open where the entry's file is, else where the last pick was
        current = self._find_in_container(container, "__source_index")
        value = str(getattr(current, "value", "") or "")
        if os.path.isabs(value) and os.path.isdir(os.path.dirname(value)):
            start = os.path.dirname(value)
        elif isinstance(self._last_file_dir, str) and os.path.isdir(self._last_file_dir):
            start = self._last_file_dir

        def _done(path: str | None) -> None:
            if not path:
                return
            self._last_file_dir = os.path.dirname(path)
            self.run_worker(self._rebuild_source_index(container, "file", path))

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


class MappingField(Widget):
    """widget for editing a mapping of name -> text (Base.angle_config): a row
    per pair, each removable, and a button that adds another. The whole mapping
    is written back, so a name can be added, renamed and removed here."""

    DEFAULT_CSS = """
    MappingField {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    MappingField .map-row {
        height: auto;
        margin-bottom: 0;
    }
    MappingField .map-key {
        width: 24;
    }
    MappingField .map-rm-btn {
        min-width: 12;
        margin-left: 1;
    }
    """

    def __init__(self, field_def: FieldDef, initial_value=None) -> None:
        super().__init__()
        self.field_def = field_def
        pairs = initial_value if isinstance(initial_value, dict) else field_def.default
        self._pairs: dict = dict(pairs or {})
        self._next_idx = 0
        self._prefix = _safe_id(f"map__{field_def.path}")

    def compose(self) -> ComposeResult:
        yield Static(f"[b]{self.field_def.path.split('.')[-1]}[/b]", classes="field-label")
        if self.field_def.description:
            yield Static(self.field_def.description, classes="field-desc")
        for key, val in self._pairs.items():
            yield self._row(key, val)
        yield Button("+ Add Entry", variant="success", id=f"{self._prefix}-add", classes="dle-add-btn")

    def _row(self, key, val) -> Widget:
        idx = self._next_idx
        self._next_idx += 1
        row_id = f"{self._prefix}__{idx}"
        return Horizontal(
            Input(value=str(key or ""), placeholder="name", id=f"{row_id}__k", classes="map-key"),
            Input(value="" if val is None else str(val), placeholder="what it means",
                  id=f"{row_id}__v", classes="dict-entry-input"),
            Button("Remove", variant="error", id=f"{row_id}__rm", classes="map-rm-btn"),
            id=row_id, classes="map-row",
        )

    def add_entry(self) -> None:
        """mount an empty pair before the add button."""
        self.mount(self._row("", ""), before=self.query_one(f"#{self._prefix}-add", Button))

    @property
    def current_value(self) -> dict:
        """the pairs still on screen, in their order; a row left without a name
        is not one, and stays out of the file."""
        result: dict = {}
        for row in self.query(".map-row"):
            if not row.id or not row.id.startswith(f"{self._prefix}__"):
                continue
            try:
                key = row.query_one(f"#{row.id}__k", Input).value.strip()
                val = row.query_one(f"#{row.id}__v", Input).value.strip()
            except Exception:
                continue
            if key:
                result[key] = val
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == f"{self._prefix}-add":
            event.stop()
            self.add_entry()
        elif btn_id.startswith(f"{self._prefix}__") and btn_id.endswith("__rm"):
            event.stop()
            try:
                self.query_one(f"#{btn_id[:-len('__rm')]}").remove()
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

    class FieldEdited(Message):
        """a text field was typed into; nothing is saved yet."""

        def __init__(self, path: str, value: str) -> None:
            super().__init__()
            self.path = path
            self.value = value

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
    .section-note {
        height: auto;
        color: $text-muted;
        padding: 0 2 1 2;
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
        section_titles: dict[str, str] | None = None,
        section_notes: dict[str, str] | None = None,
        entry_factories: dict[str, Callable[[str], list[FieldDef]]] | None = None,
    ) -> None:
        super().__init__()
        # what a top-level section is called on screen (its key stays the
        # config key) and a line or two on what it is for, shown inside it
        self._section_titles = section_titles or {}
        self._section_notes = section_notes or {}
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
        # groups whose add button the form handles itself: a name is asked for
        # in place, and the factory makes the fields of the new entry
        self._entry_factories = entry_factories or {}
        # dynamic sections removed here, to be removed from the file on Save
        self._removed_sections: set[str] = set()

    def _section_note(self, name: str) -> ComposeResult:
        note = self._section_notes.get(name)
        if note:
            yield Static(note, classes="section-note", id=_safe_id(f"section-note-{name}"))

    def set_section_note(self, name: str, note: str) -> bool:
        """replace the note of a section that was composed with one."""
        self._section_notes[name] = note
        try:
            self.query_one(f"#{_safe_id(f'section-note-{name}')}", Static).update(note)
        except Exception:
            return False
        return True

    def set_field_description(self, path: str, text: str) -> bool:
        """replace the line under a field's label."""
        for row in self.query(FieldRow):
            if row.field_def.path == path:
                row.set_description(text)
                return True
        return False

    def on_input_changed(self, event: Input.Changed) -> None:
        row = event.input.parent
        if isinstance(row, FieldRow):
            self.post_message(self.FieldEdited(row.field_def.path, event.value))

    def _yield_field(self, f: FieldDef) -> ComposeResult:
        """yield the appropriate widget for a single field."""
        initial = self._values.get(f.path, f.default)
        if f.field_type == "list_of_dicts":
            yield DictListField(f, initial_value=initial)
        elif f.field_type == "mapping":
            yield MappingField(f, initial_value=initial)
        else:
            yield FieldRow(f, initial_value=initial, source=self._sources.get(f.path),
                           read_only=f.path in self._readonly_paths)

    def _render_dyn_group(self, group_name: str, dyn_groups: dict[str, list[str]]) -> ComposeResult:
        """render one dynamic section group (e.g. ASR 'Base', 'Streams').
        Collapsed by default so the form doesn't open fully expanded."""
        group_coll_id = _safe_id(f"grp-{group_name}")
        inner_id = _safe_id(f"grp-inner-{group_name}")
        with Collapsible(title=self._section_titles.get(group_name, group_name), collapsed=True, id=group_coll_id):
            yield from self._section_note(group_name)
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
                    with Collapsible(title=self._section_titles.get(top, top), collapsed=True):
                        yield from self._section_note(top)
                        yield UpstreamsEditor(top, services, schema)
                else:
                    nested_subs = self._nest_subs(subs)
                    with Collapsible(title=self._section_titles.get(top, top), collapsed=True):
                        yield from self._section_note(top)
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
        if f.field_type == "mapping":
            return MappingField(f, initial_value=initial)
        return FieldRow(f, initial_value=initial, source=self._sources.get(f.path),
                        read_only=f.path in self._readonly_paths)

    def add_section(self, section_name: str, fields: list[FieldDef], values: dict) -> None:
        """dynamically add a new collapsible section with fields and a remove button."""
        self._removed_sections.discard(section_name)
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

    def set_field_value(self, path: str, value) -> bool:
        """show another value in a text field: what Save made of what was typed."""
        try:
            widget = self.query_one(f"#{_safe_id(f'field__{path}')}")
        except Exception:
            return False
        text = "" if value is None else str(value)
        if isinstance(widget, Input):
            widget.value = text
        elif isinstance(widget, TextArea):
            widget.text = text
        else:
            return False
        self._values[path] = value
        return True

    def mark_removed(self, section_name: str) -> None:
        """a section of the file that the form does not show and that Save is
        to remove (a leftover of the template, say)."""
        self._removed_sections.add(section_name)

    def remove_section(self, section_name: str) -> None:
        """remove a dynamic section by name; Save removes it from the file."""
        self._removed_sections.add(section_name)
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
        for mf in self.query(MappingField):
            values[mf.field_def.path] = mf.current_value
        for ue in self.query(UpstreamsEditor):
            for svc_name, entries in ue.current_value.items():
                values[f"{ue.section_key}.{svc_name}"] = entries
        for section_name in self._removed_sections:
            values[section_name] = REMOVED_SECTION
        return values

    def _entry_group_of(self, button_id: str) -> str | None:
        for group, (_label, bid) in self._group_add_buttons.items():
            if bid == button_id and group in self._entry_factories:
                return group
        return None

    def _show_entry_name_bar(self, group: str, button: Button) -> None:
        button.display = False
        bar = Horizontal(
            Input(placeholder=f"name of the new {group[:-1].lower() if group.endswith('s') else group} "
                              f"(letters, digits, - _ .)",
                  id=_safe_id(f"entry-name-{group}")),
            Button("Add", variant="success", id=_safe_id(f"btn-entry-add-{group}")),
            Button("Cancel", id=_safe_id(f"btn-entry-cancel-{group}")),
            Static("", id=_safe_id(f"entry-name-note-{group}"), classes="section-note"),
            classes="entry-name-bar",
        )
        button.parent.mount(bar, after=button)
        self.call_after_refresh(lambda: self.query_one(f"#{_safe_id(f'entry-name-{group}')}", Input).focus())

    def _close_entry_name_bar(self, group: str) -> None:
        for bar in self.query(".entry-name-bar"):
            if bar.query(f"#{_safe_id(f'entry-name-{group}')}"):
                bar.remove()
        _label, bid = self._group_add_buttons[group]
        try:
            self.query_one(f"#{bid}", Button).display = True
        except Exception:
            pass

    def _add_named_entry(self, group: str) -> None:
        try:
            name = self.query_one(f"#{_safe_id(f'entry-name-{group}')}", Input).value.strip()
        except Exception:
            return
        note = self.query_one(f"#{_safe_id(f'entry-name-note-{group}')}", Static)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            note.update("A name of letters, digits, - _ and . is needed.")
            return
        section_name = f"{group}.{name}"
        if section_name in self._dynamic_sections:
            note.update(f"'{name}' is there already.")
            return
        self._close_entry_name_bar(group)
        self.add_section(section_name, self._entry_factories[group](name), {})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        for group in self._entry_factories:
            if event.input.id == _safe_id(f"entry-name-{group}"):
                event.stop()
                self._add_named_entry(group)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        group = self._entry_group_of(event.button.id or "")
        if group is not None:
            event.stop()
            self._show_entry_name_bar(group, event.button)
            return
        for group in self._entry_factories:
            if event.button.id == _safe_id(f"btn-entry-add-{group}"):
                event.stop()
                self._add_named_entry(group)
                return
            if event.button.id == _safe_id(f"btn-entry-cancel-{group}"):
                event.stop()
                self._close_entry_name_bar(group)
                return
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
