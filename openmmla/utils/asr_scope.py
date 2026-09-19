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
