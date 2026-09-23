from __future__ import annotations

# the scope a transcript is attributed at: each speaker on their own
# (speaker verification), the one person who wears the microphone (the wearer
# its Bases entry or the session's Collection Start names), or the session's
# group as one
ASR_SCOPES = ("individual", "wearer", "group")

# names a scope went by before, and other spellings of one; a config still
# holding one keeps working
_LEGACY_ASR_SCOPES = {"participant": "individual", "wear": "wearer", "worn": "wearer"}

# the scopes whose chunks never end at a change of speaker, as no speaker is
# told apart on them: a room microphone's, and a worn one's
_UNVERIFIED_SCOPES = ("wearer", "group")


def normalize_asr_scope(asr_scope: str | None = None) -> str:
    """normalize the ASR attribution scope of a Base block (asr_scope)."""
    resolved_scope = str(asr_scope or "individual").strip().lower()
    resolved_scope = _LEGACY_ASR_SCOPES.get(resolved_scope, resolved_scope)
    if resolved_scope not in ASR_SCOPES:
        raise ValueError(f"asr_scope must be 'individual', 'wearer' or 'group', not '{asr_scope}'.")
    return resolved_scope


def resolve_speaker_verification(value, asr_scope: str) -> bool:
    """resolve the speaker_verification setting of a Base block.

    auto follows the ASR attribution scope: individual-level ASR verifies speakers,
    wearer- and group-level ASR skip speaker profile verification by default.
    """
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return asr_scope == "individual"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


GROUP_CHUNK_SECONDS = 30.0


def chunk_cap(value, asr_scope: str) -> float | None:
    """the longest a chunk of one speaker may grow before it is transcribed on its own, in
    seconds. A chunk ends at a change of speaker, which a group- or wearer-scope base never hears:
    its chunk would end only at silence, minutes later in a classroom, so it is cut at 30 s unless
    told otherwise; an individual base's chunks end at speaker changes, so it has no cap unless
    told. Nothing or an unfilled placeholder keeps that default, a number is taken as given, 0 (or
    less) means no cap; a non-number keeps the default too."""
    text = str(value if value is not None else "").strip()
    default = GROUP_CHUNK_SECONDS if asr_scope in _UNVERIFIED_SCOPES else None
    if not text or (text.startswith("<") and text.endswith(">")):
        return default
    try:
        seconds = float(text)
    except ValueError:
        return default
    return seconds if seconds > 0 else None


# what the Launch tab can pick for a base besides a participant's tag (-pt): its
# speech is the group's, or its speakers are told apart by verification
LAUNCH_GROUP = "group"
LAUNCH_SPEAKERS = "speakers"


def launch_attribution(value) -> str | None:
    """whom a base launched with --participant attributes its speech to: 'group', 'speakers'
    (speaker verification) or a participant's tag; None when the flag names nothing, and the
    config decides (asr_scope, the Bases entry's participant, the session's Collection pick)."""
    text = participant_of(value)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if lowered in (LAUNCH_GROUP, LAUNCH_SPEAKERS) else text


def participant_of(value) -> str | None:
    """the tag id a Bases entry's participant names, as text: 0 and '0' are '0', a whole float
    is its integer (2.0 is '2'); nothing, a blank, an unfilled placeholder or a yes/no is None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            return None
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    if not text or (text.startswith("<") and text.endswith(">")):
        return None
    return text
