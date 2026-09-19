"""the speaker profiles ASR bases recognize: one folder per speaker under
artifacts/runtime/pipelines/asr-base/<host>/profiles/, holding the embeddings
(.pkl) its registrations made and, when audio is stored, the audio they came
from (.wav). Every base on a host shares them; which of them a base loads is
its selection (-spk/--speakers, or Edit Speaker Profiles in its menu).

Nothing here is heavy: the Launcher lists and deletes the profiles of its own
machine with it, and `mmla asr-speakers` those of another host."""

from __future__ import annotations

import os
import shutil

# speakers travel as one comma-separated flag value (-spk a,b)
SEPARATOR = ","

# what a speaker reads aloud while a registration records: phonetically
# balanced, so the profile covers much of the voice
REGISTRATION_SENTENCES = (
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
)


def profiles_dir(project_dir: str | os.PathLike[str] | None) -> str:
    """where the speaker profiles of this host are, for a base run with -p project_dir."""
    from openmmla.utils.artifact_paths import runtime_pipeline_artifact_dir
    return os.fspath(runtime_pipeline_artifact_dir(project_dir, 'asr-base', 'profiles'))


def name_problem(name: object) -> str:
    """why `name` cannot name a speaker profile (a folder of its own, and one
    entry of a comma-separated list); "" when it can."""
    text = str(name if name is not None else "")
    if not text.strip():
        return "a speaker needs a name"
    if text != text.strip():
        return "a name cannot start or end with a space"
    if text.startswith("."):
        return "a name cannot start with a dot"
    if any(ch in text for ch in ("/", "\\", SEPARATOR)):
        return "a name cannot hold / \\ or ,"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return "a name cannot hold control characters"
    if len(text) > 64:
        return "a name is at most 64 characters"
    return ""


def parse_speakers(value: object) -> list[str] | None:
    """the speakers a -spk value names, in order and once each; None for no
    value at all (every registered profile), [] for an empty one."""
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(SEPARATOR)
    names: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    return names


def join_speakers(names) -> str:
    return SEPARATOR.join(str(name) for name in names)


def registered(directory: str) -> list[str]:
    """the speakers with a profile folder in `directory`, sorted."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    return sorted(entry for entry in entries
                  if not entry.startswith('.') and os.path.isdir(os.path.join(directory, entry)))


def list_speakers(directory: str) -> list[dict]:
    """each profile in `directory`: its name, how many embeddings and audio
    files it holds, and when it last changed (epoch seconds). A profile with no
    embedding is recognized only once one is made from its audio."""
    speakers = []
    for name in registered(directory):
        folder = os.path.join(directory, name)
        embeddings = audio = 0
        updated = os.path.getmtime(folder)
        try:
            files = os.listdir(folder)
        except OSError:
            files = []
        for file_name in files:
            if file_name.endswith('.pkl'):
                embeddings += 1
            elif file_name.endswith('.wav'):
                audio += 1
            else:
                continue
            try:
                updated = max(updated, os.path.getmtime(os.path.join(folder, file_name)))
            except OSError:
                pass
        speakers.append({"name": name, "embeddings": embeddings, "audio": audio, "updated": updated})
    return speakers


def delete_speakers(directory: str, names) -> tuple[list[str], list[str]]:
    """delete the profiles of `names` (a list: a profile made by hand may hold
    a comma) in `directory`; (deleted, not there). A name that could reach
    outside the folder is never there."""
    deleted: list[str] = []
    missing: list[str] = []
    root = os.path.realpath(directory)
    for name in dict.fromkeys(str(name) for name in names):
        folder = os.path.join(directory, name)
        outside = name in ("", ".", "..") or "/" in name or "\\" in name
        if outside or not os.path.isdir(folder) or os.path.dirname(os.path.realpath(folder)) != root:
            missing.append(name)
            continue
        shutil.rmtree(folder)
        deleted.append(name)
    return deleted, missing
