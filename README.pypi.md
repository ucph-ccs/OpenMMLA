# OpenMMLA

OpenMMLA is an IoT-based multimodal data collection toolkit for learning analytics. It provides three pipelines, real-time speech recognition with speaker diarization (ASR), AprilTag-based indoor positioning (IPS) and vision-language video frame analysis (VFA), synchronized across base stations, stored in InfluxDB and MongoDB, and shown on a dashboard live and after the session.

- Documentation: https://openmmla.readthedocs.io
- Source code and issues: https://github.com/ucph-ccs/OpenMMLA
- Paper: https://doi.org/10.1145/3706468.3706525

## Installation

This package provides the `openmmla` library and the `mmla` command line. Each pipeline has its own extra, so a machine only installs what it runs:

```bash
pip install "openmmla[asr-base]"      # ASR base station
pip install "openmmla[ips-base]"      # IPS base station
pip install "openmmla[vfa-base]"      # VFA base station
pip install "openmmla[uber-base]"     # analytics and session exports
pip install "openmmla[uber-server]"   # dashboard backend and worker
pip install "openmmla[tui]"           # management console
```

The pipelines and the management console run against a checkout of the repository, which holds the pipeline configuration bundles (`pipelines/`), the shared settings (`config/`) and the Docker compose files of the AI services (`docker/`). The recommended setup is therefore an editable install from the clone:

```bash
git clone https://github.com/ucph-ccs/OpenMMLA.git
cd OpenMMLA
conda create -n tui python=3.10 -y && conda activate tui
pip install -e ".[tui]"
mmla tui
```

See the documentation for the system prerequisites (conda, git, tmux, PortAudio, FFmpeg), the system services (InfluxDB, MongoDB, Redis, Mosquitto, Nginx, dashboard) and the pipeline guides.

## Citation

```bibtex
@inproceedings{10.1145/3706468.3706525,
author = {Li, Zaibei and Yamaguchi, Shunpei and Spikol, Daniel},
title = {OpenMMLA: an IoT-based Multimodal Data Collection Toolkit for Learning Analytics},
year = {2025},
doi = {10.1145/3706468.3706525},
booktitle = {Proceedings of the 15th International Learning Analytics and Knowledge Conference},
pages = {872–879},
}
```
