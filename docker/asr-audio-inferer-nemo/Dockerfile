# AudioInferer (NeMo backend, EXPERIMENTAL) — speaker embedding inference, port 5001, GPU
# Alternative to the wespeaker image; enable with the compose profile "nemo".
# Build from the repository root:
#   docker build -f docker/asr-audio-inferer-nemo/Dockerfile -t openmmla/asr-audio-inferer-nemo .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pins mirror the asr-server-nemo extra in pyproject.toml
RUN pip install "Cython==3.0.11" && \
    pip install \
        "nemo-toolkit[asr]<=1.23.0" \
        "onnxruntime==1.18.1" \
        "huggingface_hub==0.23.5" \
        "librosa" "soundfile" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5001 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_audio_inferer:app
