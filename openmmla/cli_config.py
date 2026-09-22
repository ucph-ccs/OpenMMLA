# openmmla/cli_config.py

# Commands hidden from the main `mmla --help` listing. They remain fully
# functional (the TUI and power users still invoke them); they are just not
# part of the user-facing surface. Show them with `mmla --help --all`.
#
# - asr-infer/resample/enhance/separate/vad/transcribe: single-worker dev
#   runners for ASR services; production stacks are launched from the TUI
#   (one gunicorn tmux session per service from pipelines/asr-server/config.yml)
# - vfa-vllm: single-worker dev runner for the VFA frame analyzer
# - crypto: secrets are now encrypted automatically (TUI config save and
#   Base/Server config load); only needed for manual key init/rotation
HIDDEN_COMMANDS = {
    "asr-infer",
    "asr-resample",
    "asr-enhance",
    "asr-separate",
    "asr-transcribe",
    "asr-vad",
    "vfa-vllm",
    "crypto",
}

# Optional dependency group for each command
OPTIONAL_DEP_MAP = {
    "asr-base": "asr-base",
    "asr-sync": "asr-base",
    "asr-speakers": "asr-base",
    "asr-infer": "asr-server-nemo",
    "asr-resample": "asr-server-nemo",
    "asr-enhance": "asr-server-nemo",
    "asr-separate": "asr-server-nemo",
    "asr-transcribe": "asr-server-nemo",
    "asr-vad": "asr-server-nemo",
    "ips-ccal": "ips-base",
    "ips-csync": "ips-base",
    "ips-ctag": "ips-base",
    "ips-base": "ips-base",
    "ips-sync": "ips-base",
    "ips-vis": "ips-base",
    "vfa-base": "vfa-base",
    "vfa-sync": "vfa-base",
    "vfa-vllm": "vfa-server",
    "collect-audio": "tui",
    "collect-video": "tui",
    "ses-ctl": "uber-base",
    "ses-ana": "uber-base",
    "ses-fuse": "uber-base",
    "ses-man": "uber-base",
    "ses-import": "uber-base",
    "ses-tidy": "uber-base",
    "ses-align": "uber-base",
    "ses-code": "uber-base",
    "ses-calibrate": "ips-base",
    "tui": "tui",
    "crypto": "tui",
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
    "asr-speakers": (
        "openmmla.commands.asr.speakers:main",
        "List, register and delete the speaker profiles ASR bases recognize."
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
    "vfa-sync": (
        "openmmla.commands.vfa.sync:main",
        "Run VFA synchronizer of real-time video frame analyzer."
    ),
    "vfa-vllm": (
        "openmmla.commands.vfa.vllm:main",
        "Start multimodal large language model server."
    ),
    "collect-audio": (
        "openmmla.commands.collect.audio:main",
        "Record raw audio files for post-time processing."
    ),
    "collect-video": (
        "openmmla.commands.collect.video:main",
        "Record raw video files for post-time processing."
    ),
    "ses-ctl": (
        "openmmla.commands.ses.ctl:main",
        "Control bucket session."
    ),
    "ses-ana": (
        "openmmla.commands.ses.ana:main",
        "Analyze bucket data."
    ),
    "ses-fuse": (
        "openmmla.commands.ses.fuse:main",
        "Build a session's fusion table: speech, space, body, gaze and action features per time window."
    ),
    "ses-calibrate": (
        "openmmla.commands.ses.calibrate:main",
        "Compute a session's transformation matrices between its cameras from its recordings, and check given ones."
    ),
    "ses-man": (
        "openmmla.commands.ses.man:main",
        "Manage bucket data and local data."
    ),
    "ses-import": (
        "openmmla.commands.ses.imp:main",
        "Bring a session recorded in an earlier layout into artifacts/<session>/collection/ for file replay."
    ),
    "ses-tidy": (
        "openmmla.commands.ses.tidy:main",
        "Rename a session under artifacts/, relabel its hosts, reduce legacy/ to what is raw."
    ),
    "ses-align": (
        "openmmla.commands.ses.align:main",
        "Measure a session's recordings against one clock by their audio, move them, cut them to one start."
    ),
    "ses-code": (
        "openmmla.commands.ses.code:main",
        "Code a session's ten-second windows by hand in the browser (ground truth for the interaction classes)."
    ),
    "tui": (
        "openmmla.commands.tui:main",
        "Launch TUI management console for config, services, and monitoring."
    ),
    "crypto": (
        "openmmla.commands.crypto:main",
        "Manage encryption keys for sensitive config values."
    ),
}
