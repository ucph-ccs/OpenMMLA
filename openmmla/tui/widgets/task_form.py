from __future__ import annotations

import yaml

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Button, Input, Label, TextArea

from openmmla.utils.experiments import list_tasks, load_task, save_task, delete_task


class TaskForm(Widget):
    """task management widget with list view and editor view."""

    class DataChanged(Message):
        """posted when task data is modified."""

    DEFAULT_CSS = """
    TaskForm {
        height: 1fr;
        padding: 1 2;
    }
    TaskForm .tf-title { text-style: bold; margin-bottom: 1; }
    TaskForm .tf-status { margin-bottom: 1; height: auto; }
    TaskForm .tf-entry { layout: horizontal; height: auto; margin-bottom: 1; }
    TaskForm .tf-entry-name { width: 1fr; padding-top: 1; }
    TaskForm .tf-entry Button { margin: 0 1; min-width: 10; }
    TaskForm .tf-field { layout: horizontal; height: auto; margin-bottom: 1; }
    TaskForm .tf-field-label { width: 18; padding-top: 1; }
    TaskForm .tf-field-input { width: 1fr; }
    TaskForm .tf-editor { height: 1fr; min-height: 16; }
    TaskForm .tf-actions { height: auto; margin-top: 1; }
    TaskForm .tf-actions Button { margin: 0 1; min-width: 16; }
    """

    def __init__(self, notify=None) -> None:
        super().__init__()
        self._task_names = list_tasks()
        self._editing: str | None = None
        # where a Delete asks and answers: the status line the launcher keeps
        # below the form, else this form's own
        self._notify = notify
        # the task whose Delete was pressed once: the next press deletes it
        self._pending_delete: str | None = None

    def _say(self, text: str) -> None:
        if self._notify is not None:
            self._notify(text)
            return
        self._set_status(text)
        try:
            # this line sits at the top, where a row further down may have scrolled it away
            self.query_one("#tf-status").scroll_visible(animate=False)
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="tf-root"):
            yield from self._compose_list_view()

    # ── list view ────────────────────────────────────────────────

    def _compose_list_view(self):
        yield Static("[b]Task Definitions[/b]", classes="tf-title")
        yield Static("", id="tf-status", classes="tf-status")

        for name in self._task_names:
            data = load_task(name)
            domain = data.get("domain", "")
            yield Horizontal(
                Static(f"{name}  —  domain: {domain}", classes="tf-entry-name"),
                # the task name rides in `name`: a file called "my.task.yaml"
                # makes no valid widget id
                Button("Edit", name=name, classes="tf-row-edit"),
                Button("Delete", variant="error", name=name, classes="tf-row-del"),
                classes="tf-entry",
            )

        yield Horizontal(
            Label("New task name:", classes="tf-field-label"),
            Input(placeholder="e.g. programming", id="tf-new-name", classes="tf-field-input"),
            classes="tf-field",
        )
        yield Horizontal(
            Button("Create Task", variant="primary", id="tf-create"),
            classes="tf-actions",
        )

    # ── editor view ──────────────────────────────────────────────

    def _compose_editor_view(self, name: str):
        data = load_task(name)
        yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

        yield Static(f"[b]Editing: {name}[/b]", classes="tf-title")
        yield TextArea(yaml_text, language="yaml", id="tf-yaml", classes="tf-editor")
        yield Horizontal(
            Button("Save", variant="primary", id="tf-save"),
            Button("Cancel", variant="default", id="tf-cancel"),
            classes="tf-actions",
        )

    # ── view switching ───────────────────────────────────────────

    async def _show_list(self) -> None:
        self._editing = None
        self._task_names = list_tasks()
        root = self.query_one("#tf-root")
        await root.remove_children()
        await root.mount(*list(self._compose_list_view()))

    async def _show_editor(self, name: str) -> None:
        self._editing = name
        root = self.query_one("#tf-root")
        await root.remove_children()
        await root.mount(*list(self._compose_editor_view(name)))

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#tf-status", Static).update(text)
        except Exception:
            pass

    # ── event handling ───────────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        # a Delete waits for its second press only until another button is pressed
        armed, self._pending_delete = self._pending_delete, None

        if event.button.has_class("tf-row-edit"):
            await self._show_editor(event.button.name or "")

        elif event.button.has_class("tf-row-del"):
            name = event.button.name or ""
            if armed != name:
                self._pending_delete = name
                self._say(
                    f"[yellow]Press Delete again to delete task '{escape(name)}'. "
                    f"Experiments whose Task Type it is keep the name.[/yellow]")
                return
            delete_task(name)
            done = f"[red]Task '{escape(name)}' deleted.[/red]"
            if self._notify is not None:
                # said before the list is drawn again: a write to another host
                # reports on the same line once it is done
                self._notify(done)
            self._task_names = list_tasks()
            await self._show_list()
            if self._notify is None:
                self.call_after_refresh(lambda: self._set_status(done))
            self.post_message(self.DataChanged())

        elif btn_id == "tf-create":
            name = self.query_one("#tf-new-name", Input).value.strip()
            if not name:
                self._set_status("[red]Task name is required.[/red]")
                return
            save_task(name, {"domain": ""})
            self._task_names = list_tasks()
            await self._show_editor(name)
            self.post_message(self.DataChanged())

        elif btn_id == "tf-save":
            await self._do_save()

        elif btn_id == "tf-cancel":
            await self._show_list()

    async def _do_save(self) -> None:
        if not self._editing:
            return
        yaml_text = self.query_one("#tf-yaml", TextArea).text.strip()
        if not yaml_text:
            return
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            return
        if not isinstance(data, dict):
            return
        save_task(self._editing, data)
        name = self._editing
        await self._show_list()
        self.call_after_refresh(lambda: self._set_status(f"[green]Task '{name}' saved.[/green]"))
        self.post_message(self.DataChanged())
