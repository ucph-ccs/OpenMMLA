from __future__ import annotations


def normalize_asr_scope(asr_scope: str | None = None) -> str:
    """normalize ASR attribution scope used by collection and analysis code."""
    resolved_scope = str(asr_scope or "participant").strip().lower()
    if resolved_scope not in {"participant", "group"}:
        raise ValueError("Analytics.asr_scope must be 'participant' or 'group'.")
    return resolved_scope
