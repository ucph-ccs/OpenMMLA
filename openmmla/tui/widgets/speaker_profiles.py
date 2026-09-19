"""Speakers → Manage on the ASR Base card: the speaker profiles registered on
the card's host, which of them the card's bases recognize, and registering or
deleting them, without the base's own Edit Speaker Profiles menu.

A tick is a speaker the bases of the next Start recognize. Nobody ticked
anything yet: they follow the session's experiment group (its participants
that have a profile), else every profile. Use Group goes back to that.

Record registers the name from the source of the base picked under Record
from, on the card's host: the speaker reads the sentences shown while it
records. Register Files registers it from reference audio on this machine,
copied to the host first when that is another one. A name that has a profile
already adds to it. The screen dismisses with {"picked": list or None,
"answer": the last speakers.Answer, "start_dir": the folder Add File… was in}."""

from __future__ import annotations

import os
import time
from datetime import datetime

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Input, Label, Select, SelectionList, Static

from openmmla.bases.asr.speaker_profiles import REGISTRATION_SENTENCES, name_problem
from openmmla.tui import speakers as spk
from openmmla.tui.widgets.config_form import FileBrowserModal

HELP = (
    "A tick is a speaker the base recognizes at the next Start. Until you tick one yourself it follows the "
    "session's experiment group: its participants that have a profile here (named as the participant or as their "
    "tag id), else every profile. Use Group goes back to that. Registering adds a profile to the host, for every "
    "base there to tick; a base already running keeps the ones it started with."
)


class SpeakerProfilesScreen(ModalScreen):

    DEFAULT_CSS = """
    SpeakerProfilesScreen {
        align: center middle;
    }
    SpeakerProfilesScreen > #spk-dialog {
        width: 94%;
        height: 92%;
        border: thick $primary;
        background: $surface;
        padding: 0 2;
    }
    SpeakerProfilesScreen #spk-title {
        text-style: bold;
        margin-bottom: 1;
    }
    SpeakerProfilesScreen .spk-muted {
        color: $text-muted;
    }
    SpeakerProfilesScreen #spk-body {
        height: 1fr;
    }
    SpeakerProfilesScreen #spk-list {
        height: auto;
        min-height: 5;
        max-height: 14;
        margin-top: 1;
    }
    SpeakerProfilesScreen .spk-row {
        height: auto;
        margin-top: 1;
    }
    SpeakerProfilesScreen .spk-row Label {
        padding-top: 1;
        width: 14;
    }
    SpeakerProfilesScreen .spk-row Button {
        margin-right: 1;
    }
    SpeakerProfilesScreen .spk-heading {
        text-style: bold;
        margin-top: 1;
    }
    SpeakerProfilesScreen #spk-name {
        width: 40;
    }
    SpeakerProfilesScreen #spk-name-note {
        width: 1fr;
        padding: 1 0 0 1;
    }
    SpeakerProfilesScreen #spk-base {
        width: 44;
        margin-right: 1;
    }
    SpeakerProfilesScreen #spk-seconds {
        width: 12;
        margin-right: 1;
    }
    SpeakerProfilesScreen #spk-seconds-label {
        width: 10;
    }
    SpeakerProfilesScreen #spk-files {
        width: 1fr;
        padding-top: 1;
    }
    SpeakerProfilesScreen #spk-read {
        height: auto;
        margin-top: 1;
        padding: 0 1;
        display: none;
        border: round $warning;
    }
    SpeakerProfilesScreen #spk-read.active {
        display: block;
    }
    SpeakerProfilesScreen #spk-log {
        height: auto;
        margin-top: 1;
    }
    SpeakerProfilesScreen #spk-actions {
        height: auto;
        margin: 1 0;
    }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(
        self,
        host: spk.SpeakerHost,
        *,
        answer: spk.Answer | None,
        picked: list[str] | None,
        participants: list[dict],
        group: str,
        bases: list[tuple[str, str]],
        base: str = "",
        base_label: str = "",
        vad: bool = True,
        nr: bool = True,
        store: bool = True,
        start_dir: str | None = None,
    ) -> None:
        """`answer` is what the host said last (None: not asked yet), `picked`
        the speakers ticked on the card (None: nobody ticked any), `participants`
        and `group` the session's experiment group, `bases` the (label, id) of
        the host's Bases entries to record from, `base` the one picked first,
        `base_label` which base of the card the ticks are for ("Base 1 ·
        voice_badge_0"). vad/nr/store are the card's switches, which a
        registration keeps to."""
        super().__init__()
        self._host = host
        self._base_label = base_label
        self._answer = answer
        self._picked = list(picked) if picked is not None else None
        self._participants = list(participants)
        self._group = group
        self._bases = [(label, value) for label, value in bases if value]
        self._base = base if any(value == base for _, value in self._bases) else (
            self._bases[0][1] if self._bases else "")
        self._vad, self._nr, self._store = vad, nr, store
        self._start_dir = start_dir
        self._files: list[str] = []
        # a deletion waiting for its second press: the speaker it named
        self._pending_delete: str | None = None
        self._busy = ""
        self._recording_until = 0.0
        self._countdown = None
        self._log_lines: list[str] = []

    # ---- building it ----

    def compose(self) -> ComposeResult:
        with Vertical(id="spk-dialog"):
            yield Static(f"Speaker profiles on {escape(self._host.where)}", id="spk-title")
            with VerticalScroll(id="spk-body"):
                if self._base_label:
                    yield Static(f"[b]{escape(self._base_label)}[/b]: the ticks are what this base recognizes; the "
                                 "other bases of the card have ticks of their own. The profiles are the host's, "
                                 "shared by all of them.", classes="spk-muted")
                yield Static(HELP, classes="spk-muted")
                yield Static("", id="spk-group", classes="spk-muted")
                yield SelectionList[str](id="spk-list")
                yield Static("", id="spk-pick", classes="spk-muted")
                with Horizontal(classes="spk-row"):
                    yield Button("Use Group", variant="primary", id="btn-spk-group")
                    yield Button("Use All", id="btn-spk-all")
                    yield Button("Use None", id="btn-spk-none")
                    yield Button("↻", variant="primary", id="btn-spk-refresh")
                    yield Button("Delete", variant="error", id="btn-spk-delete")
                yield Static("Register a speaker", classes="spk-heading")
                with Horizontal(classes="spk-row"):
                    yield Label("Name:")
                    yield Input(placeholder="the participant's name", id="spk-name",
                                suggester=SuggestFromList(self._participant_names(), case_sensitive=False))
                    yield Static("", id="spk-name-note", classes="spk-muted")
                with Horizontal(classes="spk-row"):
                    yield Label("Record from:")
                    yield Select(
                        [(label, value) for label, value in self._bases] or [("no Bases entry", "")],
                        value=self._base if self._bases else "", allow_blank=False, id="spk-base")
                    yield Label("Seconds:", id="spk-seconds-label")
                    yield Input(placeholder="as set", id="spk-seconds", type="number")
                    yield Button("Record", variant="warning", id="btn-spk-record")
                with Horizontal(classes="spk-row"):
                    yield Label("Files:")
                    yield Button("Add File…", id="btn-spk-add")
                    yield Button("Clear", id="btn-spk-clear")
                    yield Button("Register Files", variant="warning", id="btn-spk-files")
                    yield Static("", id="spk-files", classes="spk-muted")
                yield Static("", id="spk-read")
                yield Static("", id="spk-log", classes="spk-muted")
            with Horizontal(id="spk-actions"):
                yield Button("Done", variant="success", id="btn-spk-done")

    def on_mount(self) -> None:
        self._show_group()
        self._show_files()
        if self._answer is None or self._answer.listing is None:
            self._refresh()
        else:
            self._show_list()
            if not self._answer.ok and self._answer.problem:
                self._say(f"[yellow]{escape(self._answer.problem)}[/yellow]")

    def action_close(self) -> None:
        if self._busy:
            self._say(f"[yellow]{escape(self._busy)} is running on {escape(self._host.where)}: wait for it to "
                      "end before closing.[/yellow]")
            return
        self.dismiss({"picked": self._picked, "answer": self._answer, "start_dir": self._start_dir})

    # ---- what is there ----

    @property
    def _names(self) -> list[str]:
        listing = self._answer.listing if self._answer else None
        return listing.names if listing else []

    def _participant_names(self) -> list[str]:
        return [str(p.get("participant_id") or p.get("tag_id") or "") for p in self._participants
                if str(p.get("participant_id") or p.get("tag_id") or "")]

    def _choice(self) -> spk.Choice:
        listing = self._answer.listing if self._answer else None
        return spk.choose(self._picked, listing, self._participants)

    def _show_group(self) -> None:
        if not self._group:
            text = "No experiment group on the card's Session: nothing to follow but every profile."
        elif not self._participants:
            text = f"{escape(self._group)} lists no participants (Study → Experiments)."
        else:
            absent = spk.unregistered(self._names, self._participants)
            text = f"{escape(self._group)}: {escape(', '.join(self._participant_names()))}"
            if absent and self._answer and self._answer.listing is not None:
                text += f"  [yellow](no profile yet: {escape(', '.join(absent))})[/yellow]"
        self.query_one("#spk-group", Static).update(text)

    def _label(self, speaker: spk.Speaker) -> Text:
        text = Text(speaker.name, style="bold")
        if spk.is_participant(speaker.name, self._participants):
            text.append(f"  participant of {self._group}", style="green")
        if speaker.embeddings:
            detail = f"  {speaker.embeddings} embedding(s)"
            if speaker.audio:
                detail += f", {speaker.audio} audio"
            text.append(detail, style="dim")
        else:
            text.append("  no embedding: register again", style="yellow")
        if speaker.updated:
            text.append(f"  {datetime.fromtimestamp(speaker.updated):%Y-%m-%d %H:%M}", style="dim")
        return text

    def _show_list(self) -> None:
        listing = self._answer.listing if self._answer else None
        chosen = self._choice().names
        ticked = set(chosen if chosen is not None else (listing.names if listing else []))
        widget = self.query_one("#spk-list", SelectionList)
        highlighted = widget.highlighted
        with widget.prevent(SelectionList.SelectedChanged):
            widget.clear_options()
            widget.add_options([(self._label(s), s.name, s.name in ticked) for s in (listing.speakers if listing else [])])
        if highlighted is not None and widget.option_count:
            widget.highlighted = min(highlighted, widget.option_count - 1)
        self._show_pick()
        self._show_group()

    def _show_pick(self) -> None:
        choice = self._choice()
        listing = self._answer.listing if self._answer else None
        if listing is None:
            text = f"[yellow]{escape(self._host.where)} has not said which profiles it has.[/yellow]"
        elif not listing.speakers:
            text = f"[yellow]No speaker profile on {escape(self._host.where)} yet: register them below.[/yellow]"
        elif choice.how == "picked":
            text = (f"Ticked by you for this base: {len(choice.names or [])} of {len(listing.speakers)}. "
                    "Use Group lets the group decide again.")
        elif choice.how == "group":
            text = f"Following {escape(self._group)}: its participants that have a profile here."
        else:
            text = ("Following every profile: " + (f"none is a participant of {escape(self._group)}."
                                                   if self._group else "the card's Session has no group."))
        self.query_one("#spk-pick", Static).update(text)

    def _show_files(self) -> None:
        if self._files:
            text = ", ".join(escape(os.path.basename(path)) for path in self._files)
            if not self._host.local:
                text += f"  [dim](copied to {escape(self._host.where)} to register)[/dim]"
        else:
            text = "reference audio on this machine"
        self.query_one("#spk-files", Static).update(text)

    def _say(self, *lines: str) -> None:
        self._log_lines = list(lines)
        self._show_log()

    def _add_line(self, line: str) -> None:
        self._log_lines = (self._log_lines + [line])[-8:]
        self._show_log()

    def _show_log(self) -> None:
        try:
            self.query_one("#spk-log", Static).update("\n".join(self._log_lines))
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "spk-name":
            return
        event.stop()
        self._show_name_note(event.value.strip())

    def _show_name_note(self, name: str) -> None:
        if not name:
            note = ""
        elif name_problem(name):
            note = f"[yellow]{escape(name_problem(name))}[/yellow]"
        elif name in self._names:
            note = "has a profile: registering adds to it"
        elif spk.is_participant(name, self._participants):
            note = f"[green]a participant of {escape(self._group)}[/green]"
        else:
            note = "a new profile" + (f", not a participant of {escape(self._group)}" if self._group else "")
        try:
            self.query_one("#spk-name-note", Static).update(note)
        except Exception:
            pass

    # ---- ticking ----

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        event.stop()
        widget = self.query_one("#spk-list", SelectionList)
        ticked = set(widget.selected)
        self._picked = [name for name in self._names if name in ticked]
        self._pending_delete = None
        self._show_pick()

    # ---- the buttons ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button.id or ""
        if button != "btn-spk-delete":
            self._pending_delete = None
        if button == "btn-spk-done":
            self.action_close()
        elif button == "btn-spk-group":
            self._picked = None
            self._show_list()
        elif button == "btn-spk-all":
            self._picked = list(self._names)
            self._show_list()
        elif button == "btn-spk-none":
            self._picked = []
            self._show_list()
        elif button == "btn-spk-refresh":
            self._refresh()
        elif button == "btn-spk-delete":
            self._delete()
        elif button == "btn-spk-record":
            self._register(files=False)
        elif button == "btn-spk-files":
            self._register(files=True)
        elif button == "btn-spk-add":
            self.app.push_screen(FileBrowserModal(self._start_dir), self._file_chosen)
        elif button == "btn-spk-clear":
            self._files = []
            self._show_files()

    def _file_chosen(self, path: str | None) -> None:
        if not path:
            return
        self._start_dir = os.path.dirname(path)
        if not path.lower().endswith(spk.AUDIO_EXTENSIONS):
            self._say(f"[yellow]{escape(os.path.basename(path))} is not audio "
                      f"({', '.join(spk.AUDIO_EXTENSIONS)}).[/yellow]")
            return
        if path not in self._files:
            self._files.append(path)
        self._show_files()

    def _set_busy(self, what: str) -> None:
        self._busy = what
        for button in ("btn-spk-record", "btn-spk-files", "btn-spk-delete", "btn-spk-refresh", "btn-spk-done"):
            try:
                self.query_one(f"#{button}", Button).disabled = bool(what)
            except Exception:
                pass

    # ---- asking the host ----

    def _refresh(self) -> None:
        if self._busy:
            return
        self._say(f"Asking {escape(self._host.where)} for its speaker profiles...")
        self._set_busy("Listing the profiles")
        self.run_worker(self._list(), exclusive=True, group="spk-work")

    async def _list(self) -> None:
        import asyncio
        answer = await asyncio.to_thread(spk.list_profiles, self._host)
        self._set_busy("")
        self._take(answer)
        if answer.ok:
            self._say(f"{len(answer.listing.speakers) if answer.listing else 0} profile(s) on "
                      f"{escape(self._host.where)}.")
        else:
            self._say(f"[red]{escape(answer.problem)}[/red]")

    def _take(self, answer: spk.Answer) -> None:
        """keep what the host said, when it said which profiles it has."""
        if answer.listing is not None:
            self._answer = answer
            if self._picked is not None:
                # a profile deleted meanwhile is ticked no more
                self._picked = [name for name in self._picked if name in answer.listing.names]
        elif self._answer is None:
            self._answer = answer
        self._show_list()

    def _delete(self) -> None:
        widget = self.query_one("#spk-list", SelectionList)
        if widget.highlighted is None or not widget.option_count:
            self._say("Highlight the profile to delete in the list first.")
            return
        name = str(widget.get_option_at_index(widget.highlighted).value)
        if self._pending_delete != name:
            self._pending_delete = name
            self._say(f"[yellow]Press Delete again to delete the profile of {escape(name)} on "
                      f"{escape(self._host.where)}: its embeddings and audio go for good.[/yellow]")
            return
        self._pending_delete = None
        self._set_busy(f"Deleting {name}")
        self._say(f"Deleting the profile of {escape(name)} on {escape(self._host.where)}...")
        self.run_worker(self._run_delete(name), exclusive=True, group="spk-work")

    async def _run_delete(self, name: str) -> None:
        import asyncio
        answer = await asyncio.to_thread(spk.delete_profiles, self._host, [name])
        self._set_busy("")
        self._take(answer)
        if answer.ok and name in (answer.fields.get("deleted") or []):
            self._say(f"[green]Deleted the profile of {escape(name)}.[/green]")
        elif answer.ok:
            self._say(f"[yellow]{escape(self._host.where)} has no profile of {escape(name)}.[/yellow]")
        else:
            self._say(f"[red]{escape(answer.problem)}[/red]")

    def _register(self, files: bool) -> None:
        if self._busy:
            return
        name = self.query_one("#spk-name", Input).value.strip()
        problem = name_problem(name)
        if problem:
            self._say(f"[yellow]Name: {escape(problem)}.[/yellow]")
            return
        base = self.query_one("#spk-base", Select).value
        base = "" if base is Select.NULL else str(base or "")
        if not base:
            self._say("[yellow]The host's config has no Bases entry to take the settings from: add one on the "
                      "Config tab.[/yellow]")
            return
        if files and not self._files:
            self._say("[yellow]Add the reference audio with Add File… first.[/yellow]")
            return
        seconds = None
        text = self.query_one("#spk-seconds", Input).value.strip()
        if text and not files:
            try:
                seconds = float(text)
            except ValueError:
                seconds = None
            if not seconds or seconds <= 0:
                self._say("[yellow]Seconds: a number above 0, or empty for the base type's "
                          "register_duration.[/yellow]")
                return
        what = f"{'Adding to the profile of' if name in self._names else 'Registering'} {escape(name)}"
        if files:
            self._say(f"{what} from {len(self._files)} file(s) on {escape(self._host.where)}...")
        else:
            self._say(f"{what}: get ready to read the sentences below...")
            self._show_sentences(seconds, waiting=True)
        self._set_busy(f"Registering {name}")
        self.run_worker(self._run_register(name, base, seconds, list(self._files) if files else []),
                        exclusive=True, group="spk-work")

    def _show_sentences(self, seconds: float | None, waiting: bool) -> None:
        read = self.query_one("#spk-read", Static)
        sentences = "\n".join(f"{i}. {s}" for i, s in enumerate(REGISTRATION_SENTENCES, 1))
        if waiting:
            head = "[b]Opening the base's source…[/b] Read these aloud once the recording starts:"
        else:
            left = max(0, int(self._recording_until - time.monotonic() + 0.99))
            head = f"[b][red]● Recording[/red] — read aloud: {left} s left[/b]"
        read.update(f"{head}\n{sentences}")
        read.add_class("active")

    def _hide_sentences(self) -> None:
        if self._countdown is not None:
            self._countdown.stop()
            self._countdown = None
        self.query_one("#spk-read", Static).remove_class("active")

    def _on_register_line(self, line: str) -> None:
        # "Recording 15 s from base 0 (pyaudio): ..." — the recording starts now
        if line.startswith("Recording ") and " s from base " in line:
            try:
                seconds = float(line.split()[1])
            except (IndexError, ValueError):
                seconds = 0.0
            self._recording_until = time.monotonic() + seconds
            self._show_sentences(seconds, waiting=False)
            if self._countdown is None:
                self._countdown = self.set_interval(1.0, lambda: self._show_sentences(None, waiting=False))
        elif line.startswith("Recorded;"):
            self._hide_sentences()
        self._add_line(escape(line))

    async def _run_register(self, name: str, base: str, seconds: float | None, files: list[str]) -> None:
        answer = await spk.register(
            self._host, name=name, base=base, duration=seconds, files=files,
            vad=self._vad, nr=self._nr, store=self._store, on_line=self._on_register_line)
        self._hide_sentences()
        self._set_busy("")
        was_ticked_by_hand = self._picked is not None
        self._take(answer)
        if answer.ok:
            if was_ticked_by_hand and name not in self._picked:
                # a speaker just registered is one the bases should know
                self._picked.append(name)
                self._show_list()
            if files:
                self._files = []
                self._show_files()
            self._add_line(f"[green]Registered {escape(name)} on {escape(self._host.where)}.[/green]")
            self._show_name_note(name)
        else:
            self._add_line(f"[red]{escape(answer.problem)}[/red]")
