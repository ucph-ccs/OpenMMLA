# VoiceActivityDetector (silero-vad) — VAD, port 5006, CPU only (silero is tiny)
# Build from the repository root:
#   docker build -f docker/asr-voice-activity-detector/Dockerfile -t openmmla/asr-voice-activity-detector .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CPU torch wheel keeps this image small; pinned 2.4.1 because silero-vad
# 5.1.2 uses torchaudio.list_audio_backends, removed in newer torchaudio
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.1" "torchaudio==2.4.1" && \
    pip install \
        "silero-vad==5.1.2" \
        "soundfile" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5006 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_voice_activity_detector:app
