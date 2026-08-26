# SpeechSeparator (ModelScope MossFormer2) — speech separation, port 5004, GPU
# Build from the repository root:
#   docker build -f docker/asr-speech-separator/Dockerfile -t openmmla/asr-speech-separator .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        "torch==2.4.1" "torchaudio==2.4.1" \
        "numpy<2" \
        "modelscope[framework]==1.16.1" \
        "rotary-embedding-torch==0.6.4" \
        "speechbrain==1.0.0" \
        "Pillow" \
        "librosa" "soundfile" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5004 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_speech_separator:app
