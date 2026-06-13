"""Prompt-profile definitions for the multi-angle VFA analyzer.

Kept dependency-free so the TUI can import it without pulling in torch/cv2.

A profile selects which end-to-end prompt template files are loaded. The
two-step (VLM observation + LLM classification) templates are fixed and shared
across profiles. Profiles correspond to the ICALT26 paper conditions:

- cot: zero-shot Chain-of-Thought prompting (main pipeline)
- baseline: direct classification without CoT reasoning
- baseline_no_pre: baseline without the pre-context block
"""

PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "cot": {
        "multi_angle_end_system_prompt.txt": "multi_angle_end_system_prompt_template",
        "multi_angle_end_user_prompt.txt": "multi_angle_end_user_prompt_template",
    },
    "baseline": {
        "multi_angle_end_system_prompt_baseline.txt": "multi_angle_end_system_prompt_template",
        "multi_angle_end_user_prompt_baseline.txt": "multi_angle_end_user_prompt_template",
    },
    "baseline_no_pre": {
        "multi_angle_end_system_prompt_baseline_no_pre.txt": "multi_angle_end_system_prompt_template",
        "multi_angle_end_user_prompt_baseline_no_pre.txt": "multi_angle_end_user_prompt_template",
    },
}

TWO_STEP_TEMPLATES: dict[str, str] = {
    "multi_angle_vlm_system_prompt.txt": "multi_angle_vlm_system_prompt_template",
    "multi_angle_vlm_user_prompt.txt": "multi_angle_vlm_user_prompt_template",
    "multi_angle_llm_system_prompt.txt": "multi_angle_llm_system_prompt_template",
    "multi_angle_llm_user_prompt.txt": "multi_angle_llm_user_prompt_template",
}

DEFAULT_PROMPT_PROFILE = "cot"


def profile_template_files(profile: str) -> dict[str, str]:
    """Return the filename -> attribute mapping for a profile (plus two-step files)."""
    if profile not in PROMPT_PROFILES:
        raise ValueError(
            f"Unknown prompt_profile '{profile}'. "
            f"Expected one of: {', '.join(sorted(PROMPT_PROFILES))}"
        )
    return {**PROMPT_PROFILES[profile], **TWO_STEP_TEMPLATES}


def active_prompt_files(profile: str, end_to_end: bool) -> list[str]:
    """Return the prompt filenames actually used for a given configuration."""
    if end_to_end:
        return sorted(PROMPT_PROFILES.get(profile, {}))
    return sorted(TWO_STEP_TEMPLATES)
