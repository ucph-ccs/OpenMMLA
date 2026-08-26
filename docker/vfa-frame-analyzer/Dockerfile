# VLLMFrameAnalyzer — multi-angle frame analysis, port 5007, GPU
# (torch for gaze detection, tensorflow/retina-face for face detection,
#  OpenAI-compatible client for the VLM/LLM backends)
# Build from the repository root:
#   docker build -f docker/vfa-frame-analyzer/Dockerfile -t openmmla/vfa-frame-analyzer .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# libgl1/libglib2.0-0 for opencv, fonts for annotated overlays
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pins mirror the vfa-server extra in pyproject.toml
RUN pip install \
        "torch==2.7.0" "torchvision==0.22.0" \
        "numpy==1.26.4" \
        "openai==1.73.0" \
        "opencv-python==4.10.0.84" \
        "Pillow==10.4.0" \
        "pupil_apriltags==1.0.4.post11" \
        "retina-face==0.0.17" \
        "tf-keras==2.19.0" \
        "timm==1.0.15" \
        "scikit-learn==1.6.1" \
        "matplotlib==3.9.2" \
        "huggingface-hub==0.24.6" \
        "flask==3.0.3" "gunicorn==22.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5007 \
    WORKERS=1

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.vfa.apps.serve_multi_angle_vllm_frame_analyzer:app
