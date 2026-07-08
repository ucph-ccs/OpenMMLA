# AudioResampler — sample-rate conversion, port 5002, CPU only
# Build from the repository root:
#   docker build -f docker/asr-audio-resampler/Dockerfile -t openmmla/asr-audio-resampler .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        "numpy" "librosa" "soundfile" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5002 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_audio_resampler:app
