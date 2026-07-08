# AudioInferer (WeSpeaker backend) — speaker embedding inference, port 5001, GPU
# Build from the repository root:
#   docker build -f docker/asr-audio-inferer-wespeaker/Dockerfile -t openmmla/asr-audio-inferer-wespeaker .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# torch 2.4.1 default wheels are CUDA 12.1 and bundle the CUDA runtime via
# nvidia-* pip packages: the host only needs the NVIDIA driver + container toolkit
RUN pip install \
        "torch==2.4.1" "torchaudio==2.4.1" \
        "numpy==1.22.4" \
        "librosa" "soundfile" "tqdm" \
        "silero-vad" \
        "transformers>=4.43,<5" \
        "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@9ce7995648a281ba6ce6f4e7a33941672ef22779" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5001 \
    WORKERS=1

# --timeout 0: model load at boot can far exceed gunicorn's 30s default
CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_audio_inferer:app
