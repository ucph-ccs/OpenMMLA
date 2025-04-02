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
        "openmmla.examples.asr.run_audio_base:main",
        "Run ASR audio base for speaker recognition and transcription."
    ),
    "asr-sync": (
        "openmmla.examples.asr.run_audio_synchronizer:main",
        "Run ASR audio synchronizer for synchronizing results from audio bases."
    ),
    "asr-post": (
        "openmmla.examples.asr.run_post_audio_analyzer:main",
        "Run ASR post-time audio analyser."
    ),
    "asr-infer": (
        "openmmla.examples.asr.serve_audio_inferer:main",
        "Start audio inference server."
    ),
    "asr-resample": (
        "openmmla.examples.asr.serve_audio_resampler:main",
        "Start audio resampling server."
    ),
    "asr-enhance": (
        "openmmla.examples.asr.serve_speech_enhancer:main",
        "Start speech enhancement server."
    ),
    "asr-separate": (
        "openmmla.examples.asr.serve_speech_separator:main",
        "Start speech separation server."
    ),
    "asr-transcribe": (
        "openmmla.examples.asr.serve_speech_transcriber:main",
        "Start speech transcription server."
    ),
    "asr-vad": (
        "openmmla.examples.asr.serve_voice_activity_detector:main",
        "Start voice activity detection server."
    ),
    "ips-ccal": (
        "openmmla.examples.ips.run_camera_calibrator:main",
        "Run camera calibrator for calibrating camera's intrinsic parameters."
    ),
    "ips-csync": (
        "openmmla.examples.ips.run_camera_sync_manager:main",
        "Run camera sync manager for synchronizing multi-camera's transformation matrices."
    ),
    "ips-ctag": (
        "openmmla.examples.ips.run_camera_tag_detector:main",
        "Run camera tag detector for detecting AprilTags."
    ),
    "ips-base": (
        "openmmla.examples.ips.run_video_base:main",
        "Run IPS video base for AprilTags detection."
    ),
    "ips-sync": (
        "openmmla.examples.ips.run_video_synchronizer:main",
        "Run IPS video synchronizer for synchronizing results from video bases."
    ),
    "ips-vis": (
        "openmmla.examples.ips.run_video_visualizer:main",
        "Run IPS video visualizer for visualizing the bases results."
    ),
    "vfa-vlm": (
        "openmmla.examples.vfa.serve_video_frame_analyzer:main",
        "Start vision language model server."
    ),
    "ses-ctl": (
        "openmmla.examples.session.run_session_control:main",
        "Start/stop bucket sessions."
    ),
    "ses-ana": (
        "openmmla.examples.session.run_session_analysis:main",
        "Analyze and summarize collected multimodal measurements by session."
    ),
}
