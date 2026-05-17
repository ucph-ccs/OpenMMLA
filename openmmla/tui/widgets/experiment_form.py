from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, Select

from openmmla.utils.experiments import (
    load_experiments, save_experiments, list_tasks,
)


def _normalize_participant_info(info: object) -> dict[str, str]:
    """normalize participant info from experiments.yaml for the TUI."""
    if not isinstance(info, dict):
        return {"group_id": "", "tag_id": "", "description": ""}
    normalized = {}
    for key in ("group_id", "tag_id", "description"):
        value = info.get(key, "")
        normalized[key] = str(value).strip() if value is not None else ""
    return normalized


def _format_participant_summary(person: str, info: object) -> str:
    """build a readable participant summary for the detail view."""
    normalized = _normalize_participant_info(info)
    group_id = normalized["group_id"] or "-"
    tag_id = normalized["tag_id"] or "-"
    description = normalized["description"] or "-"
    return f"{person}  ->  {group_id}  / tag: {tag_id}  / desc: {description}"


class ExperimentForm(Widget):
    """experiment management widget with list view and detail view."""

    class DataChanged(Message):
        """posted when experiment data is modified."""

    DEFAULT_CSS = """
    ExperimentForm {
        height: 1fr;
        padding: 1 2;
    }
    ExperimentForm .ef-title { text-style: bold; margin-bottom: 1; }
    ExperimentForm .ef-status { margin-bottom: 1; height: auto; }
    ExperimentForm .ef-entry { layout: horizontal; height: auto; margin-bottom: 1; }
    ExperimentForm .ef-entry-name { width: 1fr; padding-top: 1; }
    ExperimentForm .ef-entry Button { margin: 0 1; min-width: 10; }
    ExperimentForm .ef-field { layout: horizontal; height: auto; margin-bottom: 1; }
    ExperimentForm .ef-field-label { width: 22; padding-top: 1; }
    ExperimentForm .ef-field-input { width: 1fr; }
    ExperimentForm .ef-actions { height: auto; margin-top: 1; }
    ExperimentForm .ef-actions Button { margin: 0 1; min-width: 16; }
    ExperimentForm .ef-section { margin-top: 1; height: auto; }
    ExperimentForm .ef-participant { layout: horizontal; height: auto; margin-bottom: 1; }
    ExperimentForm .ef-participant-name { width: 1fr; padding-top: 1; }
    ExperimentForm .ef-participant Button { margin: 0 1; min-width: 10; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._data = load_experiments()
        self._editing_exp: str | None = None
        self._editing_participant: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="ef-root"):
            yield from self._compose_list_view()

    # ── list view ────────────────────────────────────────────────

    def _compose_list_view(self):
        yield Static("[b]Experiments[/b]", classes="ef-title")
        yield Static("", id="ef-status", classes="ef-status")

        for exp in self._data.get("active_experiments", []):
            eid = exp.get("experiment_id", "?")
            title = exp.get("title", "")
            task = exp.get("task_type", "")
            status = exp.get("status", "")
            groups = self._groups_summary(eid)
            yield Horizontal(
                Static(
                    f"{eid}  —  {title}  [{task}]  ({status})  {groups}",
                    classes="ef-entry-name",
                ),
                Button("Edit", id=f"ef-open-{eid}"),
                Button("Delete", variant="error", id=f"ef-del-{eid}"),
                classes="ef-entry",
            )

        yield Horizontal(
            Label("New experiment ID:", classes="ef-field-label"),
            Input(placeholder="e.g. exp_03", id="ef-new-id", classes="ef-field-input"),
            classes="ef-field",
        )
        yield Horizontal(
            Button("Create Experiment", variant="primary", id="ef-create"),
            classes="ef-actions",
        )

    def _groups_summary(self, eid: str) -> str:
        assignments = self._data.get("assignments", {}).get(eid, {})
        if not assignments:
            return ""
        groups: dict[str, int] = {}
        for info in assignments.values():
            gid = info.get("group_id", "?")
            groups[gid] = groups.get(gid, 0) + 1
        parts = [f"{g}:{n}" for g, n in sorted(groups.items())]
        return f"[dim]({', '.join(parts)})[/dim]"

    # ── detail / edit view ───────────────────────────────────────

    def _compose_detail_view(self, eid: str):
        exp = next(
            (e for e in self._data.get("active_experiments", []) if e.get("experiment_id") == eid),
            {},
        )

        yield Static(f"[b]Experiment: {eid}[/b]", classes="ef-title")
        yield Static("", id="ef-status", classes="ef-status")

        yield Horizontal(
            Label("Experiment ID:", classes="ef-field-label"),
            Input(value=exp.get("experiment_id", ""), id="ef-id", classes="ef-field-input"),
            classes="ef-field",
        )
        yield Horizontal(
            Label("Title:", classes="ef-field-label"),
            Input(value=exp.get("title", ""), id="ef-title", classes="ef-field-input"),
            classes="ef-field",
        )
        task_names = list_tasks()
        options = [(t, t) for t in task_names]
        task_val = exp.get("task_type", "")
        yield Horizontal(
            Label("Task Type:", classes="ef-field-label"),
            Select(
                options,
                value=task_val if task_val else Select.BLANK,
                allow_blank=True,
                prompt="select task type",
                id="ef-task-type",
                classes="ef-field-input",
            ),
            classes="ef-field",
        )
        status_val = exp.get("status", "active")
        yield Horizontal(
            Label("Status:", classes="ef-field-label"),
            Select(
                [("active", "active"), ("inactive", "inactive")],
                value=status_val if status_val else Select.BLANK,
                allow_blank=True,
                id="ef-status-select",
                classes="ef-field-input",
            ),
            classes="ef-field",
        )

        # current participants
        yield Static("[b]Participants[/b]", classes="ef-title")
        assignments = self._data.get("assignments", {}).get(eid, {})
        participant_rows = []
        for person, info in sorted(assignments.items()):
            participant_rows.append(
                Horizontal(
                    Static(_format_participant_summary(person, info), classes="ef-participant-name"),
                    Button("Edit", id=f"ef-edp-{person}"),
                    Button("Remove", variant="error", id=f"ef-rmp-{person}"),
                    classes="ef-participant",
                )
            )
        yield Vertical(*participant_rows, id="ef-participants", classes="ef-section")

        # add participant
        yield Static("[b]Add Participant[/b]", classes="ef-title")
        yield Horizontal(
            Label("Name:", classes="ef-field-label"),
            Input(placeholder="e.g. Alice", id="ef-p-name", classes="ef-field-input"),
            classes="ef-field",
        )
        yield Horizontal(
            Label("Group ID:", classes="ef-field-label"),
            Input(placeholder="e.g. group_01", id="ef-p-group", classes="ef-field-input"),
            classes="ef-field",
        )
        yield Horizontal(
            Label("Tag ID:", classes="ef-field-label"),
            Input(placeholder="e.g. 12", id="ef-p-tag", classes="ef-field-input"),
            classes="ef-field",
        )
        yield Horizontal(
            Label("Description:", classes="ef-field-label"),
            Input(
                placeholder="e.g. learner wearing a grey hoodie",
                id="ef-p-description",
                classes="ef-field-input",
            ),
            classes="ef-field",
        )
        yield Horizontal(
            Button("Add", variant="success", id="ef-p-add"),
            Button("Cancel Edit", id="ef-p-cancel-edit", disabled=True),
            classes="ef-actions",
        )

        # bottom actions
        yield Horizontal(
            Button("Save & Back", variant="primary", id="ef-save"),
            Button("Cancel", variant="default", id="ef-cancel"),
            classes="ef-actions",
        )

    # ── view switching ───────────────────────────────────────────

    async def _show_list(self) -> None:
        self._editing_exp = None
        self._editing_participant = None
        root = self.query_one("#ef-root")
        await root.remove_children()
        await root.mount(*list(self._compose_list_view()))

    async def _show_detail(self, eid: str) -> None:
        self._editing_exp = eid
        self._editing_participant = None
        root = self.query_one("#ef-root")
        await root.remove_children()
        await root.mount(*list(self._compose_detail_view(eid)))

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#ef-status", Static).update(text)
        except Exception:
            pass

    async def _refresh_participants(self) -> None:
        """rebuild only the participant list in detail view."""
        eid = self._editing_exp
        if not eid:
            return
        container = self.query_one("#ef-participants")
        await container.remove_children()
        assignments = self._data.get("assignments", {}).get(eid, {})
        for person, info in sorted(assignments.items()):
            h = Horizontal(
                Static(_format_participant_summary(person, info), classes="ef-participant-name"),
                Button("Edit", id=f"ef-edp-{person}"),
                Button("Remove", variant="error", id=f"ef-rmp-{person}"),
                classes="ef-participant",
            )
            await container.mount(h)

    def _set_participant_form_mode(self, editing_person: str | None) -> None:
        """update participant form buttons for add vs edit mode."""
        self._editing_participant = editing_person
        add_button = self.query_one("#ef-p-add", Button)
        cancel_button = self.query_one("#ef-p-cancel-edit", Button)
        if editing_person:
            add_button.label = "Save Participant"
            add_button.variant = "primary"
            cancel_button.disabled = False
        else:
            add_button.label = "Add"
            add_button.variant = "success"
            cancel_button.disabled = True

    def _clear_participant_form(self) -> None:
        """reset participant form inputs and leave add mode active."""
        self.query_one("#ef-p-name", Input).value = ""
        self.query_one("#ef-p-group", Input).value = ""
        self.query_one("#ef-p-tag", Input).value = ""
        self.query_one("#ef-p-description", Input).value = ""
        self._set_participant_form_mode(None)

    def _load_participant_form(self, person: str) -> None:
        """fill participant form with existing data for editing."""
        eid = self._editing_exp
        if not eid:
            return
        info = self._data.get("assignments", {}).get(eid, {}).get(person)
        if info is None:
            self._set_status(f"[red]Participant '{person}' was not found.[/red]")
            return
        normalized = _normalize_participant_info(info)
        self.query_one("#ef-p-name", Input).value = person
        self.query_one("#ef-p-group", Input).value = normalized["group_id"]
        self.query_one("#ef-p-tag", Input).value = normalized["tag_id"]
        self.query_one("#ef-p-description", Input).value = normalized["description"]
        self._set_participant_form_mode(person)

    # ── event handling ───────────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""

        # list view actions
        if btn_id.startswith("ef-open-"):
            eid = btn_id[len("ef-open-"):]
            await self._show_detail(eid)

        elif btn_id.startswith("ef-del-"):
            eid = btn_id[len("ef-del-"):]
            exps = self._data.get("active_experiments", [])
            self._data["active_experiments"] = [e for e in exps if e.get("experiment_id") != eid]
            self._data.get("assignments", {}).pop(eid, None)
            save_experiments(self._data)
            await self._show_list()
            self.call_after_refresh(lambda: self._set_status(f"[red]Experiment '{eid}' deleted.[/red]"))
            self.post_message(self.DataChanged())

        elif btn_id == "ef-create":
            new_id = self.query_one("#ef-new-id", Input).value.strip()
            if not new_id:
                self._set_status("[red]Experiment ID is required.[/red]")
                return
            existing = [e["experiment_id"] for e in self._data.get("active_experiments", [])]
            if new_id in existing:
                self._set_status(f"[red]'{new_id}' already exists. Click Edit instead.[/red]")
                return
            self._data.setdefault("active_experiments", []).append({
                "experiment_id": new_id,
                "title": "",
                "status": "active",
                "task_type": "",
            })
            self._data.setdefault("assignments", {})[new_id] = {}
            save_experiments(self._data)
            await self._show_detail(new_id)
            self.post_message(self.DataChanged())

        # detail view actions
        elif btn_id == "ef-save":
            await self._save_detail()

        elif btn_id == "ef-cancel":
            await self._show_list()

        elif btn_id == "ef-p-add":
            await self._add_participant()

        elif btn_id == "ef-p-cancel-edit":
            self._clear_participant_form()
            self._set_status("[yellow]Participant edit cancelled.[/yellow]")

        elif btn_id.startswith("ef-edp-"):
            person = btn_id[len("ef-edp-"):]
            self._load_participant_form(person)
            self._set_status(f"[yellow]Editing participant '{person}'.[/yellow]")

        elif btn_id.startswith("ef-rmp-"):
            person = btn_id[len("ef-rmp-"):]
            await self._remove_participant(person)

    async def _save_detail(self) -> None:
        eid = self._editing_exp
        if not eid:
            return

        new_id = self.query_one("#ef-id", Input).value.strip()
        title = self.query_one("#ef-title", Input).value.strip()
        task_sel = self.query_one("#ef-task-type", Select).value
        task_type = str(task_sel) if task_sel is not Select.BLANK else ""
        status_sel = self.query_one("#ef-status-select", Select).value
        status = str(status_sel) if status_sel is not Select.BLANK else "active"

        if not new_id:
            self._set_status("[red]Experiment ID is required.[/red]")
            return

        exp = next(
            (e for e in self._data.get("active_experiments", []) if e.get("experiment_id") == eid),
            None,
        )
        if exp:
            if new_id != eid:
                exp["experiment_id"] = new_id
                assigns = self._data.get("assignments", {})
                assigns[new_id] = assigns.pop(eid, {})
            exp["title"] = title
            exp["task_type"] = task_type
            exp["status"] = status

        save_experiments(self._data)
        await self._show_list()
        self.call_after_refresh(lambda: self._set_status(f"[green]Experiment '{new_id}' saved.[/green]"))
        self.post_message(self.DataChanged())

    async def _add_participant(self) -> None:
        eid = self._editing_exp
        if not eid:
            return
        original_name = self._editing_participant
        name = self.query_one("#ef-p-name", Input).value.strip()
        group = self.query_one("#ef-p-group", Input).value.strip()
        tag_id = self.query_one("#ef-p-tag", Input).value.strip()
        description = self.query_one("#ef-p-description", Input).value.strip()
        if not name:
            self._set_status("[red]Participant name is required.[/red]")
            return
        if not group:
            self._set_status("[red]Group ID is required.[/red]")
            return
        if not tag_id:
            self._set_status("[red]Tag ID is required.[/red]")
            return
        if not description:
            self._set_status("[red]Description is required.[/red]")
            return

        assignments = self._data.setdefault("assignments", {}).setdefault(eid, {})
        if name in assignments and name != original_name:
            self._set_status(f"[red]Participant '{name}' already exists.[/red]")
            return
        for person, info in assignments.items():
            if person == original_name:
                continue
            if _normalize_participant_info(info).get("tag_id") == tag_id:
                self._set_status(
                    f"[red]Tag ID '{tag_id}' is already assigned to '{person}'.[/red]"
                )
                return

        participant_data = {
            "group_id": group,
            "tag_id": tag_id,
            "description": description,
        }
        if original_name and original_name != name:
            assignments.pop(original_name, None)
        assignments[name] = participant_data
        save_experiments(self._data)
        self._clear_participant_form()
        if original_name:
            self._set_status(f"[green]{name} updated.[/green]")
        else:
            self._set_status(f"[green]{name} -> {group} (tag {tag_id})[/green]")
        await self._refresh_participants()
        self.post_message(self.DataChanged())

    async def _remove_participant(self, person: str) -> None:
        eid = self._editing_exp
        if not eid:
            return
        self._data.get("assignments", {}).get(eid, {}).pop(person, None)
        save_experiments(self._data)
        if self._editing_participant == person:
            self._clear_participant_form()
        self._set_status(f"[red]{person} removed.[/red]")
        await self._refresh_participants()
        self.post_message(self.DataChanged())
