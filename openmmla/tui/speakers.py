"""Speakers on the ASR Base card: the speaker profiles registered on the
card's host, which of them its bases recognize, and registering or deleting
them without the base's own menu.

The profiles are the host's, one folder per speaker under
artifacts/runtime/pipelines/asr-base/<host>/profiles/ (bases.asr.speaker_profiles),
shared by every base there. This machine's are listed and deleted here;
another host's with `mmla asr-speakers` over SSH, in its asr-base env and its
own checkout. A registration always runs `mmla asr-speakers --register` on
the card's host: it records from a base's source there (or reads files copied
there first) and needs the ASR server.

What a Start passes its bases (-spk) when nobody picked speakers on the card:
the profiles that are participants of the session's experiment group (a
profile named as the participant, or as their tag id), else nothing, and each
base recognizes every profile its host has."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Callable

from rich.markup import escape

from openmmla.bases.asr import speaker_profiles as sp
from openmmla.commands.asr.speakers import MARKER
from openmmla.tui.ssh import get_profile_by_name, scp_file_async, ssh_run_async, ssh_run_sync, wrap_local, wrap_remote
from openmmla.tui.stream_cuts import _quote_root
from openmmla.utils.artifact_paths import short_hostname
from openmmla.utils.asr_scope import normalize_asr_scope, resolve_speaker_verification
from openmmla.utils.config import get_base_by_id, get_bases

CONDA_ENV = "asr-base"
PIPELINE_PARTS = ("pipelines", "asr-base")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".wma")

# a card's summary names this many speakers, then counts the rest
SHOWN_NAMES = 6


@dataclass(frozen=True)
class Speaker:
    name: str
    embeddings: int = 0
    audio: int = 0
    updated: float = 0.0


@dataclass
class Listing:
    """the speaker profiles of one host, as it answered."""
    speakers: list[Speaker]
    host: str = ""        # the host's own name, which names its profiles folder
    directory: str = ""

    @property
    def names(self) -> list[str]:
        return [speaker.name for speaker in self.speakers]


@dataclass
class Answer:
    """what one run of mmla asr-speakers (or its local counterpart) came to."""
    ok: bool
    listing: Listing | None = None
    fields: dict = field(default_factory=dict)
    problem: str = ""


@dataclass(frozen=True)
class SpeakerHost:
    """the host of an ASR Base card: "local" or an SSH profile name. `root` is
    this machine's project root."""
    target: str
    root: str

    @property
    def local(self) -> bool:
        return self.target == "local"

    @property
    def where(self) -> str:
        return "this machine" if self.local else self.target


def _speakers(rows) -> list[Speaker]:
    speakers = []
    for row in rows or []:
        if not isinstance(row, dict) or not str(row.get("name") or ""):
            continue
        try:
            speakers.append(Speaker(str(row["name"]), int(row.get("embeddings") or 0), int(row.get("audio") or 0),
                                    float(row.get("updated") or 0.0)))
        except (TypeError, ValueError):
            speakers.append(Speaker(str(row["name"])))
    return speakers


def local_listing(root: str) -> Listing:
    directory = sp.profiles_dir(os.path.join(root, *PIPELINE_PARTS))
    return Listing(_speakers(sp.list_speakers(directory)), short_hostname(), directory)


def parse_answer(text: str, returncode: int, where: str) -> Answer:
    """what the output of mmla asr-speakers on `where` says: its last @speakers
    line, or why there was none."""
    fields: dict = {}
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith(MARKER):
            try:
                fields = json.loads(line[len(MARKER):])
            except ValueError:
                fields = {}
            break
    listing = None
    if isinstance(fields, dict) and "speakers" in fields:
        listing = Listing(_speakers(fields.get("speakers")), str(fields.get("host") or ""),
                          str(fields.get("dir") or ""))
    if fields and returncode == 0 and not fields.get("error"):
        return Answer(True, listing, fields)
    return Answer(False, listing, fields if isinstance(fields, dict) else {},
                  str((fields or {}).get("error") or "") or explain_failure(text, returncode, where))


def explain_failure(text: str, returncode: int, where: str) -> str:
    """a failed run of mmla asr-speakers, in words."""
    output = text or ""
    if "Unknown command: asr-speakers" in output:
        return (f"The OpenMMLA checkout on {where} has no `mmla asr-speakers` yet: pull the latest OpenMMLA "
                "there (Environment → Git Pull), then press ↻.")
    if ("EnvironmentNameNotFound" in output or "Could not find conda environment" in output
            or "mmla: command not found" in output):
        return f"{where} has no conda env '{CONDA_ENV}' with OpenMMLA in it: set it up on the Environment screen."
    tail = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith(MARKER)][-3:]
    return " / ".join(tail) or f"mmla asr-speakers exited with code {returncode} on {where}."


def _join(root: str, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *[part.strip("/") for part in parts if part]])


def _remote_config_dir(profile) -> str:
    return _join(profile.remote_project_path, *PIPELINE_PARTS)


def speakers_command(config_dir: str, python_path: str, args: list[str], quote=shlex.quote,
                     with_config: bool = False) -> str:
    """the shell command that runs mmla asr-speakers for the checkout of
    `config_dir` (pipelines/asr-base); `quote` quotes a path there. Every item of
    `args` is quoted with shlex, except one given as ("path", value)."""
    parts = []
    for arg in args:
        if isinstance(arg, tuple):
            parts.append(quote(arg[1]))
        else:
            parts.append(shlex.quote(str(arg)))
    config = f" -c {quote(_join(config_dir, 'config.yml'))}" if with_config else ""
    return (f"cd {quote(config_dir)} && export PYTHONPATH={quote(python_path)}:$PYTHONPATH && "
            f"mmla asr-speakers -p {quote(config_dir)}{config} {' '.join(parts)}")


def _remote_run(host: SpeakerHost, args: list, timeout: float, with_config: bool = False) -> Answer:
    profile = get_profile_by_name(host.target)
    if profile is None:
        return Answer(False, problem=f"SSH profile '{host.target}' not found.")
    command = speakers_command(_remote_config_dir(profile), profile.remote_project_path, args, _quote_root,
                               with_config)
    try:
        result = ssh_run_sync(profile, wrap_remote(command, CONDA_ENV), timeout=timeout)
    except subprocess.TimeoutExpired:
        return Answer(False, problem=f"{host.where} did not answer within {timeout:g} s.")
    except Exception as e:
        # the ssh argument list holds the password: say what failed, not how it was called
        return Answer(False, problem=f"Could not reach {host.where} ({type(e).__name__}).")
    return parse_answer((result.stdout or "") + "\n" + (result.stderr or ""), result.returncode, host.where)


def list_profiles(host: SpeakerHost, timeout: float = 45.0) -> Answer:
    """the speaker profiles registered on `host` (blocking: run it off the UI thread)."""
    if host.local:
        try:
            return Answer(True, local_listing(host.root))
        except OSError as e:
            return Answer(False, problem=f"Could not read the speaker profiles here: {e}")
    return _remote_run(host, ["--list"], timeout)


def delete_profiles(host: SpeakerHost, names: list[str], timeout: float = 45.0) -> Answer:
    """delete the profiles of `names` on `host` (blocking)."""
    if host.local:
        try:
            directory = sp.profiles_dir(os.path.join(host.root, *PIPELINE_PARTS))
            deleted, missing = sp.delete_speakers(directory, names)
            return Answer(True, local_listing(host.root), {"deleted": deleted, "missing": missing})
        except OSError as e:
            return Answer(False, problem=f"Could not delete here: {e}")
    return _remote_run(host, ["--delete", *names], timeout)


def _flag(value: bool) -> str:
    return "true" if value else "false"


async def register(host: SpeakerHost, *, name: str, base: str, duration: float | None = None,
                   files: list[str] | None = None, vad: bool = True, nr: bool = True, store: bool = True,
                   on_line: Callable[[str], None] = lambda line: None) -> Answer:
    """register `name` on `host`: from `files` on this machine (copied there
    first when the host is another), else recorded for `duration` seconds from
    the source of base `base` there. Every line it prints goes to `on_line`."""
    files = list(files or [])
    args: list = ["--register", name, "-b", str(base), "-vad", _flag(vad), "-nr", _flag(nr), "-s", _flag(store)]
    if duration and not files:
        args += ["-t", f"{duration:g}"]
    # the recording, and the ASR server preparing it and making the voice features
    timeout = (duration or 60.0) + 60.0 * max(1, len(files)) + 120.0
    upload_dir = ""
    if host.local:
        if files:
            args += ["--files", *[("path", os.path.abspath(path)) for path in files]]
        config_dir = os.path.join(host.root, *PIPELINE_PARTS)
        command = wrap_local(speakers_command(config_dir, host.root, args, shlex.quote, with_config=True), CONDA_ENV)
        try:
            proc = await asyncio.create_subprocess_shell(
                command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
        except OSError as e:
            return Answer(False, problem=f"Could not run mmla asr-speakers here: {e}")
    else:
        profile = get_profile_by_name(host.target)
        if profile is None:
            return Answer(False, problem=f"SSH profile '{host.target}' not found.")
        if files:
            upload_dir = _join(profile.remote_project_path, "artifacts", "runtime", "uploads",
                               f"asr-speakers-{uuid.uuid4().hex[:12]}")
            copied = await _upload(profile, files, upload_dir, host.where, on_line)
            if isinstance(copied, str):
                return Answer(False, problem=copied)
            args += ["--files", *[("path", path) for path in copied]]
        command = speakers_command(_remote_config_dir(profile), profile.remote_project_path, args, _quote_root,
                                   with_config=True)
        if upload_dir:
            command = f"{command}; rc=$?; rm -rf {_quote_root(upload_dir)}; exit $rc"
        proc = await ssh_run_async(profile, wrap_remote(command, CONDA_ENV))
    lines: list[str] = []

    async def read() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            lines.append(line)
            if line and not line.startswith(MARKER):
                on_line(line)

    try:
        await asyncio.wait_for(read(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return Answer(False, problem=f"The registration on {host.where} took longer than {timeout:g} s and was "
                                     "stopped.")
    returncode = await proc.wait()
    return parse_answer("\n".join(lines), returncode, host.where)


async def _upload(profile, files: list[str], upload_dir: str, where: str, on_line) -> list[str] | str:
    """copy `files` into `upload_dir` on the host of `profile`: the paths there,
    or why they could not be copied."""
    mkdir = await ssh_run_async(profile, f"mkdir -p {_quote_root(upload_dir)}")
    if await mkdir.wait() != 0:
        return f"Could not make a folder for the files on {where}."
    copied = []
    for index, path in enumerate(files):
        # the name the file has there: short and plain, the extension kept for the decoder
        remote_path = _join(upload_dir, f"{index + 1:02d}{os.path.splitext(path)[1].lower()}")
        on_line(f"Copying {os.path.basename(path)} to {where}...")
        proc = await scp_file_async(profile, path, remote_path)
        output = (await proc.communicate())[0].decode(errors="replace").strip()
        if proc.returncode != 0:
            return f"Could not copy {os.path.basename(path)} to {where}: {output or f'exit code {proc.returncode}'}"
        copied.append(remote_path)
    return copied


# ---- which speakers a Start passes ----

def session_group(new_session: bool, session: object, experiment_group: object,
                  group_choices: list[str]) -> tuple[str, str] | None:
    """the (experiment, group) a card's Start runs: its Experiment Group for a
    new session, else the group whose session ids the picked one's starts with
    (<experiment>_<group>_<time>)."""
    if new_session:
        text = str(experiment_group or "").strip()
        if "/" not in text:
            return None
        experiment, group = (part.strip() for part in text.split("/", 1))
        return (experiment, group) if experiment and group else None
    session_id = str(session or "").strip()
    for choice in sorted(group_choices, key=len, reverse=True):
        if "/" not in choice:
            continue
        experiment, group = choice.split("/", 1)
        if session_id.startswith(f"{experiment}_{group}_"):
            return experiment, group
    return None


def _matches(name: str, participant: dict) -> bool:
    wanted = name.strip().casefold()
    return any(wanted == str(participant.get(key) or "").strip().casefold()
               for key in ("participant_id", "tag_id") if str(participant.get(key) or "").strip())


def is_participant(name: str, participants: list[dict]) -> bool:
    return any(_matches(name, participant) for participant in participants)


def group_pick(names: list[str], participants: list[dict]) -> list[str]:
    """the registered profiles that are participants of the group (named as the
    participant or as their tag id), in the group's order."""
    picked: list[str] = []
    for participant in participants:
        for name in names:
            if name not in picked and _matches(name, participant):
                picked.append(name)
    return picked


def unregistered(names: list[str], participants: list[dict]) -> list[str]:
    """the participants of the group no profile stands for."""
    return [str(participant.get("participant_id") or participant.get("tag_id") or "")
            for participant in participants
            if not any(_matches(name, participant) for name in names)]


@dataclass
class Choice:
    """what a Start passes its bases: `names` for -spk, or None for no -spk at
    all (each base recognizes every profile its host has)."""
    names: list[str] | None
    how: str   # "picked" on the card, the "group"'s participants, "all" registered, or "unknown"


def choose(picked: list[str] | None, listing: Listing | None, participants: list[dict]) -> Choice:
    if picked is not None:
        return Choice(list(picked), "picked")
    if listing is None:
        return Choice(None, "unknown")
    matched = group_pick(listing.names, participants)
    if matched:
        return Choice(matched, "group")
    return Choice(None, "all")


def _names(names: list[str]) -> str:
    shown = ", ".join(escape(name) for name in names[:SHOWN_NAMES])
    rest = len(names) - SHOWN_NAMES
    return shown + (f" +{rest} more" if rest > 0 else "")


@dataclass
class Context:
    """what the Speakers of an ASR Base card come to on its host."""
    group: str                  # "<experiment>/<group>" of the card's Session, "" for none
    participants: list[dict]    # that group's, from Study → Experiments
    listing: Listing | None     # the profiles on the host; None: not known
    problem: str                # why they are not known
    choice: Choice


def summary(context: Context, where: str, unused: str = "") -> str:
    """the card's Speakers line: who its bases recognize, and why those.
    `unused` says why no base of the card recognizes anyone, if none does."""
    choice, listing, participants, group, problem = (
        context.choice, context.listing, context.participants, context.group, context.problem)
    note = f"  [dim](not used: {escape(unused)})[/dim]" if unused else ""
    if listing is None:
        why = escape(problem) if problem else f"{escape(where)} has not said yet"
        start = (f"Start names {_names(choice.names)}" if choice.names
                 else "Start names none: each base recognizes every profile it has")
        return f"[yellow]Profiles not known ({why}).[/yellow] {start}{note}"
    names = listing.names
    if choice.how == "picked":
        if not choice.names:
            return f"[yellow]None picked[/yellow] ({len(names)} registered on {escape(where)}){note}"
        missing = [name for name in choice.names if name not in names]
        text = f"{_names(choice.names)} [dim](picked)[/dim]"
        if missing:
            text += f"  [yellow]not registered on {escape(where)}: {_names(missing)}[/yellow]"
        return text + note
    if not names:
        return f"[yellow]None registered on {escape(where)}[/yellow]: Manage registers them{note}"
    if choice.how == "group":
        text = f"{_names(choice.names or [])} [dim](the participants of {escape(group)})[/dim]"
        absent = unregistered(names, participants)
        if absent:
            text += f"  [yellow]not registered: {_names(absent)}[/yellow]"
        return text + note
    text = f"All {len(names)} registered on {escape(where)}: {_names(names)}"
    if group:
        text += f"  [dim](none is a participant of {escape(group)})[/dim]"
    return text + note


# ---- the bases that use them ----

def verifies_speakers(config: dict, base_id: str) -> bool | None:
    """whether base `base_id` of `config` recognizes speakers (its base type's
    asr_scope is individual, or speaker_verification is on); None when the
    config does not say."""
    entry = get_base_by_id(config or {}, base_id) if base_id else None
    if entry is None:
        bases = get_bases(config or {})
        if base_id or len(bases) != 1:
            return None
        entry = bases[0]   # a base given no -b takes the only entry
    block = ((config or {}).get("Base") or {}).get(str(entry.get("base_type")))
    if not isinstance(block, dict):
        return None
    try:
        scope = normalize_asr_scope(block.get("asr_scope"))
    except ValueError:
        return None
    return resolve_speaker_verification(block.get("speaker_verification", "auto"), scope)
