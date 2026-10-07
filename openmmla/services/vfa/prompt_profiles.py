"""Prompt-profile definitions for the multi-angle VFA analyzer.

Kept dependency-free so the TUI can import it without pulling in torch/cv2.

A profile selects which end-to-end prompt template files are loaded. The
two-step (VLM observation + LLM classification) templates are fixed and shared
across profiles. Profiles correspond to the ICALT26 paper conditions:

- cot: zero-shot Chain-of-Thought prompting (main pipeline)
- baseline: direct classification without CoT reasoning
- baseline_no_pre: baseline on frames without the drawn marks (no AprilTags, face
  boxes or gaze lines); the prompt identifies people by their clothing descriptions

Which marks the server draws on the action labels' frames is action_overlays (see
action_overlays() below); april_tag and gaze_detect only load the detectors.
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

# whether a profile's prompt explains the marks the server draws (AprilTag IDs, face boxes,
# gaze lines, 'in:' values); the two-step prompts explain them too. baseline_no_pre does not:
# its frames were meant to go without them
PROFILE_EXPLAINS_MARKS: dict[str, bool] = {
    "cot": True,
    "baseline": True,
    "baseline_no_pre": False,
}

# which marks the action labels' frames get (VLLMFrameAnalyzer.action_overlays): auto follows
# the prompt, all draws the AprilTags and the gaze, tags the AprilTags alone, gaze the face
# boxes and gaze lines alone, none a clean frame. Words, not on/off, which YAML reads as booleans
ACTION_OVERLAY_SETTINGS = ("auto", "all", "tags", "gaze", "none")
DEFAULT_ACTION_OVERLAYS = "auto"

_OVERLAY_MARKS = {
    "all": (True, True),
    "tags": (True, False),
    "gaze": (False, True),
    "none": (False, False),
}


def action_overlays(setting, profile: str, end_to_end: bool) -> dict[str, bool]:
    """the marks drawn on the action labels' frames, {'april_tag': bool, 'gaze': bool}, for an
    action_overlays `setting` under a prompt `profile` (end-to-end) or the two-step prompts.
    Unset, an unfilled <...> placeholder or auto: drawn when the prompt explains them. true and
    false (what YAML makes of on/off) mean all and none."""
    if isinstance(setting, bool):
        setting = "all" if setting else "none"
    text = str(setting).strip().lower() if setting is not None else ""
    if not text or (text.startswith("<") and text.endswith(">")) or text == "auto":
        drawn = PROFILE_EXPLAINS_MARKS.get(profile, True) if end_to_end else True
        return {"april_tag": drawn, "gaze": drawn}
    if text not in _OVERLAY_MARKS:
        raise ValueError(
            f"Unknown action_overlays '{setting}'. "
            f"Expected one of: {', '.join(ACTION_OVERLAY_SETTINGS)}"
        )
    tags, gaze = _OVERLAY_MARKS[text]
    return {"april_tag": tags, "gaze": gaze}


def overlay_marks_label(marks: dict[str, bool]) -> str:
    """the drawn marks in a few words, for logs and the console"""
    if marks.get("april_tag") and marks.get("gaze"):
        return "AprilTags and gaze lines"
    if marks.get("april_tag"):
        return "AprilTags only"
    if marks.get("gaze"):
        return "gaze lines only"
    return "none (clean frames)"


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
