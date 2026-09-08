# OpenMMLA

OpenMMLA is an IoT-based multimodal data collection toolkit for learning analytics. The repository has two parts:

1. The `openmmla` Python package: the core classes, utilities and pipeline implementations, the `mmla` command line, and the management console.
2. The runtime platform: the pipeline configuration bundles under `pipelines/`, the shared settings under `config/`, the Docker images of the AI services and databases under `docker/`, and the wearable firmware under `pipelines/wearables/`.

The full documentation lives in the `docs/` directory and is published at https://openmmla.readthedocs.io.

## Platform design

### High-level system design

A pipeline built with the toolkit follows a three-stage data flow:

![OpenMMLA high-level system design](docs/img/high_level_system_design.png)

<details markdown="1">
<summary><strong>Data flow stages</strong></summary>

+ **Data input stage (purple)**: multimodal raw data from sensors and wearable badges is streamed to base stations or a central media server, where it is turned into structured, coded streams for efficient transmission and processing.
+ **Data processing stage (black)**: each coded stream is processed by the matching *Base* (ASR, IPS or VFA), which handles the signal processing itself and offloads the heavier tasks to the AI server. The *Synchronizer* merges the results of all bases in a session and writes segment-level measurement features to the time-series database.
+ **Data output stage (lime)**: the measurement features are visualized on the dashboard in real time and combined into indicators of group interaction. Post-processing visualizations, logs and reports are stored with the session and reachable from the same dashboard, for both real-time awareness and retrospective analysis of group dynamics.

</details>

### System architecture

![OpenMMLA system architecture](docs/img/system_architecture.png)

<details markdown="1">
<summary><strong>Hardware components</strong></summary>

- **Sensors**: wearable devices and environmental sensors.
    + Wearables: *Regular-Badge* (an [AprilTag](https://april.eecs.umich.edu/software/apriltag) only), *Voice-Badge* (a [Nicla Vision](https://docs.arduino.cc/hardware/nicla-vision/) board with power supply, streaming audio), *Vision-Badge* (Nicla Vision plus AprilTag).
    + Environmental sensors: USB microphones (Jabra Speak2 75, built-in mics, ...) and USB cameras (Logitech HD C920, ...), attached to base stations or streamed from Raspberry Pis.
- **Base stations**: microprocessors or PCs that process the streams. Each runs one or more *Base* instances and a *Synchronizer* of the same pipeline.
- **Servers**: PCs that provide centralized services.
    + *Base server*: the AI services (speaker inference, transcription, VAD, frame analysis, ...) as Docker containers on a GPU machine.
    + *Uber server*: the system services: InfluxDB and MongoDB (databases), Redis and Mosquitto (messaging), Nginx (load balancing and RTMP streaming) and the dashboard.
- **Dashboard**: web pages for phones and browsers with session selection, real-time and post-time visualizations, and measurement downloads.

</details>

## Quick start

1. **System prerequisites** on every machine that runs a component: [Conda](docs/prerequisites.md#conda), [Git](docs/prerequisites.md#git-tmux-portaudio-and-ffmpeg), [tmux](docs/prerequisites.md#git-tmux-portaudio-and-ffmpeg), [PortAudio](docs/prerequisites.md#git-tmux-portaudio-and-ffmpeg) and [FFmpeg](docs/prerequisites.md#git-tmux-portaudio-and-ffmpeg). Install commands and official links: [System Prerequisites](docs/prerequisites.md).

2. **System services** on the uber server: [InfluxDB](docs/system_services.md#influxdb), [MongoDB](docs/system_services.md#mongodb), [Redis](docs/system_services.md#redis) and [Mosquitto](docs/system_services.md#mosquitto) are required; [Nginx](docs/system_services.md#nginx-optional) and the [Dashboard](docs/system_services.md#dashboard-optional) are optional. Installation, listener configuration and start/stop: [System Services](docs/system_services.md). InfluxDB and MongoDB can run as containers instead: [Docker](docs/docker.md).

3. **Management console**: clone the repository and start the TUI, which does the rest (creates the pipeline environments, holds the service addresses, launches and monitors everything, locally or over SSH):

    ```bash
    git clone https://github.com/ucph-ccs/OpenMMLA.git
    cd OpenMMLA
    conda create -n tui python=3.10 -y
    conda activate tui
    pip install -e '.[tui]'
    mmla tui
    ```

    Guide to every tab and card: [Management Console (TUI)](docs/tui.md).

4. **Pipelines**: set up and run [ASR with diarization](docs/pipelines/asr.md), the [Indoor positioning system](docs/pipelines/ips.md) or the [Video frame analyzer](docs/pipelines/vfa.md).

## Documentation

| Page | Content |
|---|---|
| [System Prerequisites](docs/prerequisites.md) | Conda, Git, tmux, PortAudio, FFmpeg per OS |
| [System Services](docs/system_services.md) | InfluxDB, MongoDB, Redis, Mosquitto, Nginx, dashboard: install, start/stop, System Settings |
| [Management Console](docs/tui.md) | the TUI: environments, launcher tree, cards, sessions, status, remote hosts |
| [ASR](docs/pipelines/asr.md), [IPS](docs/pipelines/ips.md), [VFA](docs/pipelines/vfa.md) | the three pipelines end to end |
| [Docker](docs/docker.md) | the ASR/VFA AI service stacks and the InfluxDB/MongoDB stack |
| [Dashboard](docs/dashboard.md) | Flask backend, Celery worker and the static frontend |
| [Nginx](docs/nginx.md), [RTMP Streaming](docs/rtmp_streaming.md) | load balancing and camera/microphone streaming |
| [Raspberry Pi](docs/raspi_config.md) | a Pi as streaming device or base station |
| [Databases](docs/database.md) | what is stored in InfluxDB and MongoDB, CLI tips, backups, migration |
| [FAQ](docs/faq.md) | known problems and fixes |

## Citation

If you use this project in your research, please cite:

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
