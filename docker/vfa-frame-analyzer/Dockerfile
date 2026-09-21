# VLLMFrameAnalyzer — multi-angle frame analysis, port 5007, GPU
# (torch for gaze detection, tensorflow/retina-face for face detection,
#  ultralytics (8.4, the first line that ships the YOLO26 weights) for the pose model behind /vllm/features,
#  OpenAI-compatible client for the VLM/LLM backends)
# The pose weights are fetched once, at the first start, into /project/weights
# (pipelines/vfa-server/weights on the host), so a rebuild does not fetch them again.
# Licence note: ultralytics (and ultralytics-thop) are AGPL-3.0. openmmla's own code stays
# MIT, but whoever publishes or serves this image over a network carries the AGPL
# source-offer obligation for the combined work; leave ultralytics out of the image
# (and set features.enabled: false) to keep a deployment AGPL-free.
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
        "ultralytics==8.4.157" \
        "flask==3.0.3" "gunicorn==22.0.0" "gevent==24.2.1" \
        "pyyaml" "cryptography>=42.0.0"

COPY pyproject.toml README.md /opt/openmmla/
COPY openmmla /opt/openmmla/openmmla
RUN pip install --no-deps /opt/openmmla

ENV PROJECT_DIR=/project \
    CONFIG_PATH=/project/config.yml \
    PORT=5007 \
    WORKERS=1
# relative paths (a pose_model written as a path) resolve inside the bind mount
WORKDIR /project

CMD gunicorn -k gevent --timeout 0 -w ${WORKERS} -b 0.0.0.0:${PORT} \
    openmmla.services.vfa.apps.serve_multi_angle_vllm_frame_analyzer:app
