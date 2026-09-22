from __future__ import annotations

# the scope a transcript is attributed at: each speaker on their own
# (speaker verification) or the session's group as one
ASR_SCOPES = ("individual", "group")

# names a scope went by before; a config still holding one keeps working
_LEGACY_ASR_SCOPES = {"participant": "individual"}


def normalize_asr_scope(asr_scope: str | None = None) -> str:
    """normalize the ASR attribution scope of a Base block (asr_scope)."""
    resolved_scope = str(asr_scope or "individual").strip().lower()
    resolved_scope = _LEGACY_ASR_SCOPES.get(resolved_scope, resolved_scope)
    if resolved_scope not in ASR_SCOPES:
        raise ValueError(f"asr_scope must be 'individual' or 'group', not '{asr_scope}'.")
    return resolved_scope


def resolve_speaker_verification(value, asr_scope: str) -> bool:
    """resolve the speaker_verification setting of a Base block.

    auto follows the ASR attribution scope: individual-level ASR verifies speakers,
    group-level ASR skips speaker profile verification by default.
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
    seconds. A chunk ends at a change of speaker, which a group-scope base never hears: its chunk
    would end only at silence, minutes later in a classroom, so it is cut at 30 s unless told
    otherwise; an individual base's chunks end at speaker changes, so it has no cap unless told.
    Nothing or an unfilled placeholder keeps that default, a number is taken as given, 0 (or
    less) means no cap; a non-number keeps the default too."""
    text = str(value if value is not None else "").strip()
    default = GROUP_CHUNK_SECONDS if asr_scope == "group" else None
    if not text or (text.startswith("<") and text.endswith(">")):
        return default
    try:
        seconds = float(text)
    except ValueError:
        return default
    return seconds if seconds > 0 else None
