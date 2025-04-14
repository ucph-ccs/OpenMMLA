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
    "vfa-base": "vfa-base",
    "vfa-vllm": "vfa-server",
}

# List of all available commands
COMMANDS = {
    "asr-base": (
        "openmmla.commands.asr.base:main",
        "Run ASR base of real-time audio analyzer."
    ),
    "asr-sync": (
        "openmmla.commands.asr.sync:main",
        "Run ASR synchronizer of real-time audio analyzer."
    ),
    "asr-post": (
        "openmmla.commands.asr.post:main",
        "Run ASR post-time audio analyser."
    ),
    "asr-infer": (
        "openmmla.commands.asr.infer:main",
        "Start audio inference server."
    ),
    "asr-resample": (
        "openmmla.commands.asr.resample:main",
        "Start audio resampling server."
    ),
    "asr-enhance": (
        "openmmla.commands.asr.enhance:main",
        "Start speech enhancement server."
    ),
    "asr-separate": (
        "openmmla.commands.asr.separate:main",
        "Start speech separation server."
    ),
    "asr-transcribe": (
        "openmmla.commands.asr.transcribe:main",
        "Start speech transcription server."
    ),
    "asr-vad": (
        "openmmla.commands.asr.vad:main",
        "Start voice activity detection server."
    ),
    "ips-ccal": (
        "openmmla.commands.ips.ccal:main",
        "Run camera calibrator for camera intrinsic calibration."
    ),
    "ips-csync": (
        "openmmla.commands.ips.csync:main",
        "Run camera sync manager for multi-cameras coordinate synchronization."
    ),
    "ips-ctag": (
        "openmmla.commands.ips.ctag:main",
        "Run camera tag detector for multi-cameras coordinate synchronization."
    ),
    "ips-base": (
        "openmmla.commands.ips.base:main",
        "Run IPS base of real-time indoor positioning system."
    ),
    "ips-sync": (
        "openmmla.commands.ips.sync:main",
        "Run IPS synchronizer of real-time indoor positioning system."
    ),
    "ips-vis": (
        "openmmla.commands.ips.vis:main",
        "Run IPS visualizer of real-time indoor positioning system."
    ),
    "vfa-base": (
        "openmmla.commands.vfa.base:main",
        "Run VFA base of real-time video frame analyzer."
    ),
    "vfa-vllm": (
        "openmmla.commands.vfa.vllm:main",
        "Start multimodal large language model server."
    ),
    "ses-ctl": (
        "openmmla.commands.ses.ctl:main",
        "Start/stop bucket session."
    ),
    "ses-ana": (
        "openmmla.commands.ses.ana:main",
        "Start session analysis."
    ),
}
