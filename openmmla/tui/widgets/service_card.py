from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Rule, Select, TabbedContent, TabPane


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
    # what the card shows as "type:"; falls back to launch_type (e.g. docker
    # stacks keep launch_type "tmux" internally but display "docker")
    display_type: str = ""
    # extra (label, action id) buttons on the card; the launcher handles them
    # through ActionRequested, e.g. ("Fetch Token", "fetch-token")
    extra_actions: list = field(default_factory=list)
    # what the tree and the card title show; name stays the internal key
    label: str = ""

    @property
    def display_name(self) -> str:
        return self.label or self.name

    @property
    def shown_type(self) -> str:
        return self.display_type or self.launch_type


@dataclass
class ParamDef:
    flag: str
    label: str
    # "int" (a counter), "bool" (a toggle), "str" (a Select when it has
    # choices, else a text box) or "choice" (always a Select, even while its
    # choices are empty: they come from the card's host and may be none yet)
    param_type: str
    default: Any
    # a Select's options: plain values, or (label, value) pairs when what the
    # Select shows is not what it passes
    choices: list = field(default_factory=list)
    # the count flag of a component (e.g. "-nb"): the card shows one Select per
    # instance ("Base 1", "Base 2", ...) and follows that counter's + and -;
    # collect_params gives a list with one value per instance, "" passing
    # nothing for that instance. A list default seeds the instances in turn,
    # the rest take the options with a value in order.
    per_instance: str = ""
    # a per-instance flag whose first instance this Select follows: picking
    # Base 1 sets it to follow_values[<Base 1's value>], when that is an option.
    # On an "int" counter, the counter it goes along with while it has no
    # default of its own (None): the synchronizer's Sync Waits For is Num Bases
    # then. Either way, a counter nobody has set moves to a fresh default.
    follows: str = ""
    follow_values: dict = field(default_factory=dict)
    # a per-instance Select whose options are not the whole world: it gets a
    # "type another…" option, and picking it opens a text box in the same row
    # with this as its placeholder (a Collection recorder's Device Label, for a
    # device no pipeline config names). Empty: the options are all there is.
    free_text: str = ""
    # the per-instance flag (same counter) whose rows this param's rows sit
    # under, row by row, in that param's container (a recorder's Participant
    # under its Device Label, a base's Participant under its Base)
    under: str = ""
    # with `under`: whether row i shows, given the value of row i of `under`
    # (its typed text on "type another…"), or of `shown_by` when that names
    # another row of the same instance; None shows every row. A hidden row
    # passes ""
    shown_when: Callable[[str], bool] | None = None
    shown_by: str = ""
    # with `under`: its rows sit above the rows of `under` rather than below
    # (a recorder's Host above its Device Label)
    above: bool = False
    # what an instance with no default of its own starts on, when that is one
    # of the options (a recorder's Host: this machine); else `fill` decides
    instance_default: str = ""
    # whether an instance with no default of its own takes the next option
    # that has a value (True), or none (False)
    fill: bool = True


# the last option of a Select that takes a typed value too (ParamDef.free_text)
TYPE_ANOTHER = "\x00type"

# flags whose Select lists artifact/collection sessions; these get an inline
# "↻" button so the list can be re-queried on demand (bypassing the cache)
_SESSION_PARAM_FLAGS = {"-sid", "--session-id", "--artifact-session-id"}


def _choice_options(choices: list) -> list[tuple[str, str]]:
    """a param's choices as Select options: (label, value) pairs, a plain
    choice being its own label."""
    options = []
    for choice in choices:
        if isinstance(choice, (tuple, list)) and len(choice) == 2:
            options.append((str(choice[0]), str(choice[1])))
        else:
            options.append((str(choice), str(choice)))
    return options


def _is_select_param(param: ParamDef) -> bool:
    return bool(param.choices) or param.param_type == "choice"


def host_params(params: list[ParamDef]) -> list[ParamDef]:
    """the params whose options or default come from the card's host (its
    config, its files), which a Refresh or a config Save renews: what
    ServiceCard.update_param_choices takes."""
    return [
        param for param in params
        if param.param_type == "choice" or param.per_instance or (param.param_type == "int" and param.follows)
    ]


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

    class StopAllRequested(Message):
        """stop a collection session on every host it may be recording on."""

        def __init__(self, service_name: str, params: dict | None = None) -> None:
            super().__init__()
            self.service_name = service_name
            self.params = params or {}

    class ActionRequested(Message):
        """one of the service's extra_actions buttons was pressed."""

        def __init__(self, service_name: str, action: str, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.action = action
            self.params = params

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

    class SpeakersRequested(Message):
        """Manage on the Speakers row of base `index`: the speaker profiles of
        the card's host, and which of them that base recognizes."""

        def __init__(self, service_name: str, index: int, params: dict) -> None:
            super().__init__()
            self.service_name = service_name
            self.index = index
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
    /* the box beside a "type another…" Select: it shares the row with it */
    ServiceCard .param-typed {
        width: 30;
        min-width: 14;
        height: 3;
        margin-left: 1;
    }
    ServiceCard .param-refresh {
        width: 5;
        min-width: 5;
        height: 3;
        margin-left: 1;
    }
    ServiceCard .param-summary {
        width: 1fr;
        height: auto;
        min-height: 3;
        padding: 1 1 0 1;
    }
    ServiceCard .param-manage {
        width: 12;
        min-width: 12;
        height: 3;
    }
    /* the line that sets off the rows of one instance from the next (a
       recorder's Host, Device Label and Participant; a base's Base,
       Participant and Speakers), as wide as a label and its dropdown */
    ServiceCard Rule.instance-sep {
        margin: 0;
        width: 66;
        max-width: 100%;
        color: $text-muted;
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
        initial_collection_role: str = "audio",
    ) -> None:
        super().__init__()
        self.service_def = service_def
        self._is_running = is_running
        # which of the collection card's Audio/Video tabs opens first; the
        # launcher passes back whatever tab was showing before a rebuild
        self._initial_collection_role = (
            initial_collection_role if initial_collection_role in ("audio", "video") else "audio"
        )
        self._param_values = {
            param.flag: self._initial_param_value(param)
            for param in self.service_def.params
        }
        # what the card itself put on each per-instance Select (by flag, then
        # instance index), and the instances whose Select the user picked:
        # only a row the card chose moves when fresh choices come
        self._card_shown: dict[str, dict[int, str]] = {}
        self._picked_instances: dict[str, set[int]] = {}
        # the counters the user has set with - or +: the others follow their default
        self._counts_set: set[str] = set()
        # a per-instance param starts with one value per instance of its counter
        for param in self.service_def.params:
            if param.per_instance:
                self._param_values[param.flag] = [
                    self._instance_default(param, index)
                    for index in range(self._instance_count(param))
                ]
            elif param.param_type == "int" and param.follows:
                self._param_values[param.flag] = self._count_default(param)
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
            yield Static(f"[b]{self.service_def.display_name}[/b]", classes="card-title")
            yield Static(
                f"  env: {self.service_def.conda_env or 'none needed'}  |  type: {self.service_def.shown_type}",
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
                for label, action in self.service_def.extra_actions:
                    yield Button(
                        label,
                        variant="warning",
                        compact=True,
                        id=self._action_id(action),
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
                            if p.under:
                                continue  # its rows sit with the rows of the param it names
                            if p.per_instance:
                                # one row per instance; + and - of its counter
                                # show, hide or add rows (_sync_instances)
                                yield from self._compose_instances(p)
                                continue
                            with Horizontal(classes="param-row"):
                                for widget in self._param_widgets(p):
                                    yield widget

    def _compose_collection(self) -> ComposeResult:
        status_text = "[green]Running[/green]" if self._is_running else "[red]Stopped[/red]"
        with Vertical():
            yield Static(f"[b]{self.service_def.display_name}[/b]", classes="card-title")
            yield Static(
                f"  env: {self.service_def.conda_env or 'none needed'}  |  type: {self.service_def.shown_type}",
                classes="card-meta",
            )
            if self.service_def.description:
                yield Static(f"  {self.service_def.description}", classes="card-meta")
            yield Static(f"  Status: {status_text}", classes="card-status")

            with TabbedContent(
                initial=self._collection_tab_id(self._initial_collection_role),
                id=self._collection_tabs_id(),
            ):
                for label, role in (("Audio", "audio"), ("Video", "video")):
                    with TabPane(label, id=self._collection_tab_id(role)):
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
            # recordings for one session usually run on several machines at
            # once; this stops them all without visiting each host in turn
            yield Button(
                "Stop All Hosts",
                variant="error",
                compact=True,
                id=_safe_id(f"stop_all_collection__{self.service_def.name}__{role}"),
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
                if param.under:
                    continue  # its rows sit under the rows of the param it names
                if param.per_instance:
                    # one row per recorder of this role; its + and - show, hide
                    # and add rows (_sync_instances). The flag belongs to this
                    # tab alone, so the rows take the plain ids _sync_instances
                    # and collect_params look for, not the role-scoped ones.
                    yield from self._compose_instances(param)
                    continue
                with Horizontal(classes="param-row"):
                    for widget in self._param_widgets(
                        param,
                        param_id=lambda kind, flag, role=role: self._collection_param_id(role, kind, flag),
                    ):
                        yield widget

    def _instance_sep(self, flag: str, key: str) -> Rule:
        """the line above the first instance of `flag` ("top") or below
        instance `key`, for instances of several rows."""
        return Rule(id=self._param_id(f"sep{key}", flag), classes="instance-sep")

    def _compose_instances(self, param: ParamDef) -> ComposeResult:
        """the rows of a per-instance param, instance by instance, each with
        the rows of the params that sit with it (above or under). Instances of
        several rows are set off from each other, and from the rows around
        them, by a line."""
        paired = self._paired_params(param.flag)
        above = [other for other in paired if other.above]
        below = [other for other in paired if not other.above]
        count = len(self._param_values[param.flag])
        with Vertical(id=self._param_id("instances", param.flag), classes="param-instances"):
            if paired:
                top = self._instance_sep(param.flag, "top")
                top.display = count > 0
                yield top
            for index in range(count):
                for other in above:
                    if index < len(self._param_values.get(other.flag) or []):
                        row = self._instance_row(other, index)
                        row.display = self._instance_shown(other, index)
                        yield row
                yield self._instance_row(param, index)
                for other in below:
                    if index < len(self._param_values.get(other.flag) or []):
                        row = self._instance_row(other, index)
                        row.display = self._instance_shown(other, index)
                        yield row
                if paired:
                    yield self._instance_sep(param.flag, str(index))

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
        elif _is_select_param(param):
            options = _choice_options(param.choices)
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
        """gather current launch parameter values; a per-instance param gives
        a list with one value per instance its counter now asks for."""
        values: dict[str, Any] = {}
        for param in self.service_def.params:
            if param.per_instance:
                values[param.flag] = self._collect_instances(param)
                continue
            if param.param_type in ("bool", "int"):
                continue
            try:
                if _is_select_param(param):
                    sel = self.query_one(f"#{self._param_id('select', param.flag)}", Select)
                    self._param_values[param.flag] = "" if sel.value is Select.NULL else str(sel.value)
                else:
                    inp = self.query_one(f"#{self._param_id('input', param.flag)}", Input)
                    self._param_values[param.flag] = inp.value
            except Exception:
                pass
        return {**self._param_values, **values}

    # ── per-instance Selects ─────────────────────────────────────

    def _instance_count(self, param: ParamDef) -> int:
        try:
            return max(0, int(self._param_values.get(param.per_instance, 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _instance_default(param: ParamDef, index: int) -> str:
        """what instance `index` starts on: the list default's entry for it,
        else the index-th option that has a value, else "" (none). A param that
        takes a typed value keeps one of those too, so a device no config names
        survives the rebuild that a host switch or a Refresh makes."""
        options = _choice_options(param.choices)
        legal = {value for _, value in options}
        if isinstance(param.default, (list, tuple)) and index < len(param.default):
            wanted = str(param.default[index] if param.default[index] is not None else "")
            if wanted in legal or (param.free_text and wanted):
                return wanted
        if param.instance_default and param.instance_default in legal:
            return param.instance_default
        if not param.fill:
            return ""
        valued = [value for _, value in options if value]
        return valued[index] if index < len(valued) else ""

    def _instance_select_id(self, flag: str, index: int) -> str:
        return self._param_id(f"select{index}", flag)

    def _instance_input_id(self, flag: str, index: int) -> str:
        return self._param_id(f"typed{index}", flag)

    def _instance_options(self, param: ParamDef) -> list[tuple[str, str]]:
        """a per-instance Select's options, with "type another…" last when the
        param takes a typed value too."""
        options = _choice_options(param.choices)
        if param.free_text:
            options.append(("type another…", TYPE_ANOTHER))
        return options

    def _instance_row_id(self, flag: str, index: int) -> str:
        return self._param_id(f"instance{index}", flag)

    def _instance_row(self, param: ParamDef, index: int) -> Horizontal:
        if param.param_type == "speakers":
            # who base `index` recognizes: the launcher writes the line
            # (show_speakers) and keeps the pick, which Manage changes
            return Horizontal(
                Static(f"{param.label} {index + 1}:", classes="param-label"),
                Button("Manage", variant="primary", compact=True,
                       id=self._param_id(f"speakers{index}", param.flag), classes="param-manage"),
                Static("[dim]asking the host...[/dim]", id=self._param_id(f"summary{index}", param.flag),
                       classes="param-summary"),
                id=self._instance_row_id(param.flag, index),
                classes="param-row",
            )
        options = self._instance_options(param)
        values = self._param_values.get(param.flag) or []
        value = values[index] if index < len(values) else self._instance_default(param, index)
        typed = ""
        if not any(option_value == value for _, option_value in options):
            if param.free_text and str(value or "").strip():
                # a value of its own: the Select sits on "type another…" and the
                # box beside it holds it
                typed, value = str(value), TYPE_ANOTHER
            else:
                value = Select.NULL
        self._card_shown.setdefault(param.flag, {})[index] = "" if value is Select.NULL else str(value)
        widgets = [
            Static(f"{param.label} {index + 1}:", classes="param-label"),
            Select(
                options,
                value=value,
                prompt=f"Select {param.label.lower()} {index + 1}...",
                id=self._instance_select_id(param.flag, index),
                classes="param-select",
            ),
        ]
        if param.free_text:
            box = Input(
                value=typed,
                placeholder=param.free_text,
                id=self._instance_input_id(param.flag, index),
                classes="param-typed",
            )
            box.display = value == TYPE_ANOTHER
            widgets.append(box)
        return Horizontal(
            *widgets,
            id=self._instance_row_id(param.flag, index),
            classes="param-row",
        )

    def _instance_value(self, param: ParamDef, index: int) -> str:
        """what row `index` of `param` shows now: its Select's value, the typed
        text when that is on "type another…", else (not mounted) what was
        noted; "" for none."""
        try:
            sel = self.query_one(f"#{self._instance_select_id(param.flag, index)}", Select)
        except Exception:
            values = self._param_values.get(param.flag)
            if not isinstance(values, list) or index >= len(values):
                return ""
            return str(values[index] if values[index] is not None else "")
        value = "" if sel.value is Select.NULL else str(sel.value)
        if value == TYPE_ANOTHER:
            try:
                return self.query_one(f"#{self._instance_input_id(param.flag, index)}", Input).value.strip()
            except Exception:
                return ""
        return value

    def _instance_shown(self, param: ParamDef, index: int) -> bool:
        """whether row `index` of a param that sits under another shows, going
        by the value of that other row (or of the row `shown_by` names);
        always for any other param."""
        if not param.under or param.shown_when is None:
            return True
        source = param.shown_by or param.under
        host = next((other for other in self.service_def.params if other.flag == source), None)
        if host is None:
            return True
        return bool(param.shown_when(self._instance_value(host, index)))

    def _paired_params(self, flag: str) -> list[ParamDef]:
        """the params whose rows sit under the rows of `flag`."""
        return [param for param in self.service_def.params if param.under == flag]

    def _show_paired_rows(self, flag: str, index: int) -> None:
        """row `index` of `flag` changed: the rows under it, and those whose
        showing it decides, show or hide along."""
        for param in self.service_def.params:
            if not param.under or flag not in (param.under, param.shown_by):
                continue
            try:
                row = self.query_one(f"#{self._instance_row_id(param.flag, index)}")
            except Exception:
                continue
            row.display = index < self._instance_count(param) and self._instance_shown(param, index)

    def instance_values(self, flag: str) -> list[str]:
        """every noted instance of a per-instance param, shown or not, as its
        row shows it now (not cut to the count)."""
        param = next((other for other in self.service_def.params if other.flag == flag), None)
        if param is None:
            return []
        return [self._instance_value(param, index) for index in range(len(self._param_values.get(flag) or []))]

    def _collect_instances(self, param: ParamDef) -> list[str]:
        """the values of the instances the counter asks for; what a Select
        on screen says wins over what was noted. A hidden row passes "", while
        its pick stays noted for when it shows again."""
        count = self._instance_count(param)
        values = list(self._param_values.get(param.flag) or [])
        while len(values) < count:
            values.append(self._instance_default(param, len(values)))
        for index in range(count):
            try:
                sel = self.query_one(f"#{self._instance_select_id(param.flag, index)}", Select)
            except Exception:
                continue
            value = "" if sel.value is Select.NULL else str(sel.value)
            if value == TYPE_ANOTHER:
                # the row is on "type another…": what was typed is the value
                try:
                    value = self.query_one(f"#{self._instance_input_id(param.flag, index)}", Input).value.strip()
                except Exception:
                    value = ""
            values[index] = value
        self._param_values[param.flag] = values
        return [value if self._instance_shown(param, index) else "" for index, value in enumerate(values[:count])]

    def _sync_instances(self, count_flag: str) -> None:
        """the counter `count_flag` changed: one row per instance it asks for.
        Rows above the count are hidden, not removed, so a choice survives a
        - followed by a +; a new instance gets a row of its own."""
        params = [param for param in self.service_def.params if param.per_instance == count_flag]
        for param in params:
            # notes a default for every new instance, which its row starts on
            self._collect_instances(param)
        for param in params:
            if param.under:
                continue  # its rows go along with the rows it sits under
            count = self._instance_count(param)
            try:
                container = self.query_one(f"#{self._param_id('instances', param.flag)}", Vertical)
            except Exception:
                continue
            paired = [other for other in self._paired_params(param.flag) if other.per_instance == count_flag]
            above = [other for other in paired if other.above]
            below = [other for other in paired if not other.above]
            values = self._param_values.get(param.flag) or []
            if paired:
                try:
                    self.query_one(f"#{self._param_id('septop', param.flag)}").display = count > 0
                except Exception:
                    pass
            for index in range(max(count, len(values))):
                try:
                    row = self.query_one(f"#{self._instance_row_id(param.flag, index)}")
                except Exception:
                    row = None
                if row is None:
                    if index >= count:
                        continue
                    for other in above:
                        other_row = self._instance_row(other, index)
                        other_row.display = self._instance_shown(other, index)
                        container.mount(other_row)
                    row = self._instance_row(param, index)
                    container.mount(row)
                    for other in below:
                        other_row = self._instance_row(other, index)
                        other_row.display = self._instance_shown(other, index)
                        container.mount(other_row)
                    if paired:
                        container.mount(self._instance_sep(param.flag, str(index)))
                    continue
                row.display = index < count
                for other in above:
                    try:
                        other_row = self.query_one(f"#{self._instance_row_id(other.flag, index)}")
                    except Exception:
                        if index >= count:
                            continue
                        other_row = self._instance_row(other, index)
                        container.mount(other_row, before=row)
                    other_row.display = index < count and self._instance_shown(other, index)
                after = row
                for other in below:
                    try:
                        other_row = self.query_one(f"#{self._instance_row_id(other.flag, index)}")
                    except Exception:
                        if index >= count:
                            continue
                        other_row = self._instance_row(other, index)
                        container.mount(other_row, after=after)
                    other_row.display = index < count and self._instance_shown(other, index)
                    after = other_row
                if paired:
                    try:
                        sep = self.query_one(f"#{self._param_id(f'sep{index}', param.flag)}")
                    except Exception:
                        if index >= count:
                            continue
                        sep = self._instance_sep(param.flag, str(index))
                        container.mount(sep, after=after)
                    sep.display = index < count

    def _initial_param_value(self, param: ParamDef) -> Any:
        if param.param_type == "bool":
            return param.default if isinstance(param.default, bool) else str(param.default).lower() == "true"
        if param.param_type == "int":
            try:
                return max(0, int(param.default))
            except (TypeError, ValueError):
                return 0
        return param.default

    def _count_default(self, param: ParamDef) -> int:
        """what a counter that follows another starts on, and goes back to
        while nobody has set it: its default, else (None) the value of the
        counter it follows."""
        default = param.default
        if (default is None or str(default).strip() == "") and param.follows:
            default = self._param_values.get(param.follows, 0)
        try:
            return max(0, int(default))
        except (TypeError, ValueError):
            return 0

    def _show_count(self, flag: str, value: int) -> None:
        """put a counter on `value`, on screen too, with the rows that follow it."""
        self._param_values[flag] = max(0, int(value))
        try:
            self.query_one(f"#{self._param_id('value', flag)}", Static).update(str(self._param_values[flag]))
        except Exception:
            pass
        self._sync_instances(flag)

    def _speakers_button(self, btn_id: str) -> int | None:
        """the base whose Speakers → Manage button `btn_id` is; None for another."""
        for param in self.service_def.params:
            if param.param_type != "speakers":
                continue
            for index in range(len(self._param_values.get(param.flag) or [])):
                if btn_id == self._param_id(f"speakers{index}", param.flag):
                    return index
        return None

    def show_speakers(self, index: int, markup: str) -> None:
        """put the Speakers line of base `index`: who it recognizes, and why."""
        for param in self.service_def.params:
            if param.param_type != "speakers":
                continue
            try:
                self.query_one(f"#{self._param_id(f'summary{index}', param.flag)}", Static).update(markup)
            except Exception:
                pass

    def _action_id(self, action: str) -> str:
        return _safe_id(f"action__{action}__{self.service_def.name}")

    def _param_id(self, kind: str, flag: str) -> str:
        return _safe_id(f"param_{kind}__{self.service_def.name}__{flag}")

    def _collection_param_id(self, role: str, kind: str, flag: str) -> str:
        return _safe_id(f"param_{kind}__{self.service_def.name}__{role}__{flag}")

    def _collection_tabs_id(self) -> str:
        return _safe_id(f"collection_tabs__{self.service_def.name}")

    def _collection_tab_id(self, role: str) -> str:
        return _safe_id(f"collection_tab__{self.service_def.name}__{role}")

    def active_collection_role(self) -> str:
        """the Audio/Video tab currently showing on a collection card."""
        if self.service_def.launch_type != "collection":
            return ""
        try:
            active = self.query_one(f"#{self._collection_tabs_id()}", TabbedContent).active
        except Exception:
            return self._initial_collection_role
        for role in ("audio", "video"):
            if active == self._collection_tab_id(role):
                return role
        return self._initial_collection_role

    def collection_snapshot(self) -> dict:
        """every collection control's current value plus the active tab.

        The launcher stores this before it rebuilds the card (switching host,
        refreshing) so the user's choices survive the rebuild. Audio and Video
        render their own widgets for the shared flags, so the active tab wins
        for anything both panes carry."""
        if self.service_def.launch_type != "collection":
            return {}
        active = self.active_collection_role()
        ordered = [role for role in ("audio", "video") if role != active] + [active]
        values: dict[str, Any] = {}
        for role in ordered:
            for flag, value in self._collect_collection_params(role).items():
                if (
                    flag not in values
                    or isinstance(value, (bool, int))
                    or str(value or "").strip()
                ):
                    values[flag] = value
        return {"role": active, "values": values}

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
            *[
                flag for flag in component.flags
                if flag not in {"--session-id", "--experiment-group", "--output-root"}
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
        """- or + on a counter: the user set it, so it stays as set; a counter
        that follows it and was never set goes along."""
        self._counts_set.add(flag)
        self._show_count(flag, int(self._param_values.get(flag, 0)) + delta)
        for follower in self.service_def.params:
            if follower.param_type == "int" and follower.follows == flag and follower.flag not in self._counts_set:
                self._show_count(follower.flag, self._count_default(follower))

    def _change_collection_int_param(self, role: str, flag: str, delta: int) -> None:
        self._param_values[flag] = max(0, int(self._param_values.get(flag, 0)) + delta)
        try:
            self.query_one(f"#{self._collection_param_id(role, 'value', flag)}", Static).update(
                str(self._param_values[flag])
            )
        except Exception:
            pass
        # a recorder more or less is a Device Label row more or less
        self._sync_instances(flag)

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
            if param.per_instance:
                # one value per instance, read off the rows themselves: a
                # Device Label picked but not yet launched is still the pick
                values[param.flag] = self._collect_instances(param)
                continue
            try:
                if _is_select_param(param):
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
            # a "choice" param's options come from the card's host: none now
            # means none there, not that they were left out
            if existing and existing.choices and not param.choices and param.param_type != "choice":
                params.append(replace(param, choices=existing.choices, default=existing.default))
            else:
                params.append(param)
        self.service_def = replace(service_def, params=params)
        self.update_param_choices(host_params(params))
        try:
            metas = list(self.query(".card-meta"))
            if metas:
                metas[0].update(
                    f"  env: {service_def.conda_env or 'none needed'}  |  type: {service_def.shown_type}"
                )
            if len(metas) > 1:
                metas[1].update(f"  {service_def.description}")
        except Exception:
            pass

    def update_param_choices(self, params: list[ParamDef]) -> None:
        """take fresh options for Selects whose choices come from the card's
        host (the Bases of its config, its matrix files) without rebuilding
        the card: each Select keeps what it shows while that is still an
        option, else it moves to the new default. A Base row on "ask in its
        window" keeps it only when the user picked that: one that fell to it
        because the config had no entry for it takes its entry once there is
        one. A counter nobody has set takes its fresh default."""
        if self.service_def.launch_type == "collection":
            return
        fresh = {param.flag: param for param in params}
        updated = []
        for param in self.service_def.params:
            new = fresh.get(param.flag)
            if new is None:
                updated.append(param)
                continue
            updated.append(replace(
                param, choices=list(new.choices), default=new.default, follow_values=dict(new.follow_values)))
        self.service_def = replace(self.service_def, params=updated)
        moved_first: list[tuple[str, str]] = []
        for param in updated:
            if param.flag not in fresh:
                continue
            if param.param_type == "int":
                if param.flag not in self._counts_set:
                    self._show_count(param.flag, self._count_default(param))
                continue
            options = _choice_options(param.choices)
            legal = {value for _, value in options}
            if param.per_instance:
                values = list(self._param_values.get(param.flag) or [])
                picked = self._picked_instances.setdefault(param.flag, set())
                card_shown = self._card_shown.setdefault(param.flag, {})
                selects: list[Select | None] = []
                shown: list[str | None] = []
                for index in range(len(values)):
                    try:
                        sel = self.query_one(f"#{self._instance_select_id(param.flag, index)}", Select)
                    except Exception:
                        sel = None
                    selects.append(sel)
                    # what the row shows now (not what the last collect noted)
                    value = values[index] if sel is None else (None if sel.value is Select.NULL else str(sel.value))
                    if value == "" and index not in picked:
                        value = None  # on "ask" for want of an entry, not by the user's pick
                    shown.append(value if value is not None and value in legal else None)
                # a row whose pick is gone (or that shows nothing) takes its
                # default, else the first entry no other instance holds
                taken = {value for value in shown[:self._instance_count(param)] if value}
                valued = [value for _, value in options if value]
                for index, sel in enumerate(selects):
                    keep = shown[index]
                    if keep is None:
                        # the card chooses for this row again (a pick that is
                        # gone is no longer the user's)
                        picked.discard(index)
                        keep = self._instance_default(param, index)
                        if keep and keep in taken:
                            keep = next((value for value in valued if value not in taken), "")
                        if keep and index < self._instance_count(param):
                            taken.add(keep)
                        if index == 0 and keep:
                            # Base 1 moved: what follows it moves along, once
                            # every Select has its new options (the Select
                            # itself says nothing while they change)
                            moved_first.append((param.flag, keep))
                    if keep not in legal:
                        keep = ""
                    values[index] = keep
                    if shown[index] is None:
                        card_shown[index] = keep
                    if sel is not None:
                        with sel.prevent(Select.Changed):
                            sel.set_options(options)
                            sel.value = keep if keep in legal else Select.NULL
                self._param_values[param.flag] = values
                continue
            try:
                sel = self.query_one(f"#{self._param_id('select', param.flag)}", Select)
            except Exception:
                sel = None
            if sel is None:
                current = str(self._param_values.get(param.flag) or "")
            else:
                current = None if sel.value is Select.NULL else str(sel.value)
            default = str(param.default or "")
            keep = current if current is not None and current in legal else (default if default in legal else "")
            self._param_values[param.flag] = keep
            if sel is not None:
                with sel.prevent(Select.Changed):
                    sel.set_options(options)
                    sel.value = keep if keep in legal else Select.NULL
        for flag, value in moved_first:
            self._follow_first_instance(flag, value)

    def on_select_changed(self, event: Select.Changed) -> None:
        """a Base row was picked: the card notes that the user chose it (so
        fresh choices leave it alone, "ask in its window" included), and for
        Base 1 a Select that follows it (the ASR synchronizer's base type)
        moves along. The event goes on up to the launcher."""
        value = "" if event.value is Select.NULL else str(event.value)
        now = "" if event.select.value is Select.NULL else str(event.select.value)
        if value != now:
            return  # the Select has moved on since (fresh choices): not a pick
        for param in self.service_def.params:
            if not param.per_instance:
                continue
            for index in range(len(self._param_values.get(param.flag) or [])):
                if event.select.id != self._instance_select_id(param.flag, index):
                    continue
                card_shown = self._card_shown.get(param.flag, {})
                picked = self._picked_instances.setdefault(param.flag, set())
                # a Select says it changed when it is mounted too, on what the card gave it
                if index in card_shown and value != card_shown[index]:
                    picked.add(index)
                else:
                    picked.discard(index)
                if param.free_text:
                    self._show_typed_box(param, index, value == TYPE_ANOTHER)
                if index == 0:
                    self._follow_first_instance(param.flag, value)
                self._show_paired_rows(param.flag, index)
                return

    def on_input_changed(self, event: Input.Changed) -> None:
        """text typed on a row on "type another…": the rows under it show or
        hide along. The event goes on up."""
        for param in self.service_def.params:
            if not param.per_instance or not param.free_text:
                continue
            for index in range(len(self._param_values.get(param.flag) or [])):
                if event.input.id == self._instance_input_id(param.flag, index):
                    self._show_paired_rows(param.flag, index)
                    return

    def set_instance_choices(self, flag: str, choices: list, default: list) -> None:
        """fresh options and defaults for one per-instance param (a
        recorder's or a base's Participant when the session or the experiment
        group changes): a row the user picked keeps its pick while it is still
        an option; every other row takes its new default."""
        param = next((other for other in self.service_def.params if other.flag == flag), None)
        if param is None:
            return
        new_param = replace(param, choices=list(choices), default=list(default or []))
        self.service_def = replace(
            self.service_def,
            params=[new_param if other.flag == flag else other for other in self.service_def.params],
        )
        options = self._instance_options(new_param)
        legal = {value for _, value in options}
        # the Selects are given options only when they change: the ASR Base
        # card's Participant rows are given theirs on every redraw of its
        # Speakers lines
        fresh_options = options != self._instance_options(param)
        values = list(self._param_values.get(flag) or [])
        picked = self._picked_instances.setdefault(flag, set())
        card_shown = self._card_shown.setdefault(flag, {})
        for index in range(len(values)):
            try:
                sel = self.query_one(f"#{self._instance_select_id(flag, index)}", Select)
            except Exception:
                sel = None
            current = values[index] if sel is None else ("" if sel.value is Select.NULL else str(sel.value))
            if index in picked and current in legal:
                keep = current
            else:
                picked.discard(index)
                keep = self._instance_default(new_param, index)
            values[index] = keep
            card_shown[index] = keep
            if sel is not None:
                with sel.prevent(Select.Changed):
                    if fresh_options:
                        sel.set_options(options)
                    wanted = keep if keep in legal else Select.NULL
                    if fresh_options or sel.value != wanted:
                        sel.value = wanted
        self._param_values[flag] = values
        count = self._instance_count(new_param)
        for index in range(len(values)):
            try:
                row = self.query_one(f"#{self._instance_row_id(flag, index)}")
            except Exception:
                continue
            row.display = index < count and self._instance_shown(new_param, index)
            # the rows whose showing this one decides (a base's Speakers)
            self._show_paired_rows(flag, index)

    def _show_typed_box(self, param: ParamDef, index: int, wanted: bool) -> None:
        """the text box of a row on "type another…": shown and focused while
        that is the pick, hidden (and emptied) once an option is picked again."""
        try:
            box = self.query_one(f"#{self._instance_input_id(param.flag, index)}", Input)
        except Exception:
            return
        if box.display == wanted:
            return
        box.display = wanted
        if wanted:
            box.focus()
        else:
            box.value = ""

    def _follow_first_instance(self, flag: str, picked: str) -> None:
        """set every Select that follows the first instance of `flag` to what
        goes with `picked`, when that is one of its options."""
        for follower in self.service_def.params:
            if follower.follows != flag:
                continue
            wanted = str(follower.follow_values.get(picked) or "")
            if not wanted or wanted not in {value for _, value in _choice_options(follower.choices)}:
                continue
            self._param_values[follower.flag] = wanted
            try:
                self.query_one(f"#{self._param_id('select', follower.flag)}", Select).value = wanted
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

    def select_collection_session(self, session_id: str) -> None:
        """point both role tabs at a session, adding it to the options if new.

        Called when a launch resolves (or creates) the session id, so the card
        on screen agrees with it without waiting for a rebuild."""
        if self.service_def.launch_type != "collection" or not session_id:
            return
        choices: list[str] = []
        params = []
        for param in self.service_def.params:
            if param.flag != "--session-id":
                params.append(param)
                continue
            choices = [str(choice) for choice in param.choices]
            if session_id not in choices:
                choices.append(session_id)
            params.append(replace(param, choices=choices, default=session_id))
        self.service_def = replace(self.service_def, params=params)
        self._param_values["--session-id"] = session_id
        for role in ("audio", "video"):
            try:
                sel = self.query_one(
                    f"#{self._collection_param_id(role, 'select', '--session-id')}", Select
                )
            except Exception:
                continue
            sel.set_options((choice, choice) for choice in choices)
            sel.value = session_id

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
                if btn_id == _safe_id(f"stop_all_collection__{self.service_def.name}__{role}"):
                    params = self._collect_collection_params(role)
                    self.post_message(self.StopAllRequested(self.service_def.name, params))
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
        elif btn_id.startswith("action__"):
            for _label, action in self.service_def.extra_actions:
                if btn_id == self._action_id(action):
                    self.post_message(
                        self.ActionRequested(self.service_def.name, action, self.collect_params())
                    )
                    return
        elif btn_id.startswith("delete_files__"):
            params = self.collect_params()
            self.post_message(self.DeleteFilesRequested(self.service_def.name, params))
        elif self._speakers_button(btn_id) is not None:
            index = self._speakers_button(btn_id)
            self.post_message(self.SpeakersRequested(self.service_def.name, index, self.collect_params()))
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
