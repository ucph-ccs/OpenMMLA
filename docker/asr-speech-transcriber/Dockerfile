# SpeechTranscriber (whisperx 3.8.x) — transcription, port 5005, GPU
# Isolated from the wespeaker stack: modern whisperx + torch 2.8 + ctranslate2>=4.5
# (cuDNN 9 comes from torch's bundled nvidia pip packages — no host cuDNN needed,
# which fixes the "libcudnn_ops_infer.so.8 not found" worker crashes).
# Build from the repository root:
#   docker build -f docker/asr-speech-transcriber/Dockerfile -t openmmla/asr-speech-transcriber .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        "whisperx==3.8.6" \
        "openai-whisper" \
        "denoiser-compat==0.1.5.dev0" \
        "librosa" "soundfile" \
        "azure-cognitiveservices-speech" \
        "dashscope" \
        "flask==3.0.3" "gunicorn==23.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

# ctranslate2 dlopens cuDNN/cuBLAS at runtime; point it at the libraries that
# ship with torch's nvidia pip packages
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5005 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.asr.apps.serve_speech_transcriber:app
