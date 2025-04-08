# openmmla/cli_config.py

# Optional dependency group for each command
OPTIONAL_DEP_MAP = {
    "asr-base": "asr-base",
    "asr-sync": "asr-base",
    "asr-post": "asr-base",
    "asr-infer": "asr-server",
    "asr-resample": "asr-server",
    "asr-enhance": "asr-server",
    "asr-separate": "asr-server",
    "asr-transcribe": "asr-server",
    "asr-vad": "asr-server",
    "ips-ccal": "ips-base",
    "ips-csync": "ips-base",
    "ips-ctag": "ips-base",
    "ips-base": "ips-base",
    "ips-sync": "ips-base",
    "ips-vis": "ips-base",
    "vfa-vlm": "vfa-server",
}

# List of all available commands
COMMANDS = {
    "asr-base": (
        "openmmla.commands.asr.asr_base:main",
        "Run ASR audio base for speaker recognition and transcription."
    ),
    "asr-sync": (
        "openmmla.commands.asr.asr_sync:main",
        "Run ASR audio synchronizer for synchronizing results from audio bases."
    ),
    "asr-post": (
        "openmmla.commands.asr.asr_post:main",
        "Run ASR post-time audio analyser."
    ),
    "asr-infer": (
        "openmmla.commands.asr.asr_infer:main",
        "Start audio inference server."
    ),
    "asr-resample": (
        "openmmla.commands.asr.asr_resample:main",
        "Start audio resampling server."
    ),
    "asr-enhance": (
        "openmmla.commands.asr.asr_enhance:main",
        "Start speech enhancement server."
    ),
    "asr-separate": (
        "openmmla.commands.asr.asr_separate:main",
        "Start speech separation server."
    ),
    "asr-transcribe": (
        "openmmla.commands.asr.asr_transcribe:main",
        "Start speech transcription server."
    ),
    "asr-vad": (
        "openmmla.commands.asr.asr_vad:main",
        "Start voice activity detection server."
    ),
    "ips-ccal": (
        "openmmla.commands.ips.ips_ccal:main",
        "Run camera calibrator for calibrating camera's intrinsic parameters."
    ),
    "ips-csync": (
        "openmmla.commands.ips.ips_csync:main",
        "Run camera sync manager for synchronizing multi-camera's transformation matrices."
    ),
    "ips-ctag": (
        "openmmla.commands.ips.ips_ctag:main",
        "Run camera tag detector for detecting AprilTags."
    ),
    "ips-base": (
        "openmmla.commands.ips.ips_base:main",
        "Run IPS video base for AprilTags detection."
    ),
    "ips-sync": (
        "openmmla.commands.ips.ips_sync:main",
        "Run IPS video synchronizer for synchronizing results from video bases."
    ),
    "ips-vis": (
        "openmmla.commands.ips.ips_vis:main",
        "Run IPS video visualizer for visualizing the bases results."
    ),
    "vfa-vlm": (
        "openmmla.commands.vfa.vfa_vlm:main",
        "Start vision language model server."
    ),
    "ses-ctl": (
        "openmmla.commands.session.ses_ctl:main",
        "Start/stop bucket sessions."
    ),
    "ses-ana": (
        "openmmla.commands.session.ses_ana:main",
        "Analyze and summarize collected multimodal measurements by session."
    ),
}
