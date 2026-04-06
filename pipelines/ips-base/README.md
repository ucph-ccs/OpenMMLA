# Indoor Positioning System (IPS)

This pipeline provides indoor positioning using camera-based tracking with AprilTags. It processes video streams 
in real-time to track the position of participants in a room, enabling spatial analysis of group interactions.

## Pipeline Overview
<img src="docs/indoor_positioning_system.png" alt="indoor_positioning_system" width="400"/>

The system uses multiple cameras to track participants wearing AprilTag markers:
- Video capture from multiple cameras (distributed or centralized)
- AprilTag detection to identify and locate markers
- Coordinate transformation between camera views
- Position tracking of participants
- Synchronization of data from multiple IPS bases
- Real-time visualization of positions

## Usage Instructions

### Install Dependencies
```bash
# Install the required dependencies for the IPS pipeline on specific machines 
# (e.g., base station: ips-base, uber server: uber-server)
conda create -n ips-base -c conda-forge python=3.10.12 -y
conda activate ips-base
pip install -e .[ips-base] # or pip install openmmla[ips-base]
# Optional: if you would like to use lab streaming layer as an input source
pip install pylsl==1.17.6 
conda install -c conda-forge liblsl=1.16.2
```


### Data Input Setup

IPS Base supports the following video input sources (configured via `Base.source` in `config.yml`):

| Source | Description | Device Setup |
|--------|-------------|-------------|
| `opencv` | USB camera directly connected to the base station or Raspberry Pi | Plug in the camera; the base will detect available video devices (indices 0–3) and prompt you to select at startup |
| `rtmp` | Video stream pulled from an NGINX RTMP server | Add RTMP entries in the `Streams` section of `config.yml` with `target: rtmp://...`. The streaming device pushes video to NGINX via ffmpeg |
| `file` | Replay from previously recorded video files | Set `file_dir` and `initial_sync_time` in config |

Additional input path:
- **Nicla Vision on-device AprilTag detection**: The badge runs `onboard_apriltag_detect.py`, performs AprilTag detection locally, and publishes position results via **MQTT** to `<session>/ips` topic. This bypasses the camera-based `VideoStream` pipeline entirely. See `pipelines/wearables/nicla-vision/ips/`

#### Stream Configuration

All stream sources (RTMP and remote devices) are configured in the unified `Streams` section of `config.yml`:

```yaml
Streams:
  # external RTMP stream (already running, Base only pulls from the URL)
  cam-external:
    target: rtmp://uber-server.local/ips/3

  # managed stream (TUI starts/stops ffmpeg on remote Raspberry Pi via SSH)
  cam-1:
    ssh_profile: rpi-living-room    # must match a TUI SSH profile name
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local/ips/1
    codec: libx264
    resolution: 1920x1080
    fps: 30
```

Streams with `ssh_profile` can be started/stopped from the TUI Launcher's **Streams** tab. Streams without `ssh_profile` are treated as external (already running).

Camera calibration is required before running the IPS pipeline:
1. Run `./calibrate.sh` (or `mmla ips-ccal`) to compute camera intrinsic parameters
2. Run `./synchronize.sh` (or `mmla ips-ctag` + `mmla ips-csync`) to synchronize coordinate systems across multiple cameras

### On Servers

> **Tip**: You can use `mmla tui` to configure and launch all services from the TUI Launcher, instead of running commands manually.

```bash
# 1. Run uber services on uber server with conda env `uber-server`
# Go to /pipelines/uber-server/ to run with scripts or run manually with brew or systemctl
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard
```

### On Base Stations
```bash
# Run ips pipelines on base station with conda env `ips-base`
# Edit your own config.yml file, see pipelines/ips-base/config_template.yml for more details
# Prefer the mmla commands below; examples/run_*.py launchers were removed.
# You can either run it via bash or python

# ===================BASH========================
# Run camera intrinsic calibration
usage: ./calibrate.sh

# Run multiple cameras coordinate synchronization
usage: ./synchronize.sh [-nc] 2 [-ns] 1 [-h]

options:
  -nc  NUM_CAMERA             : Number of camera detectors to run (default: 3)
  -ns  NUM_SYNCMANAGER       : Number of sync managers to run (default: 1)
  -h                          : Display this help message

# Run real-time indoor position system
Usage: ./run.sh [-nb NUM_BASE] [-ns NUM_SYNCHRONIZER] [-g GRAPHICS] [-s STORE] [-h]

options:
  -nb NUM_BASE         : Number of IPS bases to run (default: 3)
  -ns NUM_SYNCHRONIZER : Number of synchronizers to run (default: 1)
  -g GRAPHICS          : Enable graphics (default: true)
  -s STORE             : Enable store (default: false)
  -v VERBOSE           : Enable verbose mode (default: false)
  -h                   : Display this help message

# ==================PYTHON========================
# Activate conda env `ips-base`
conda activate ips-base

# Run camera intrinsic calibration
mmla ips-ccal -c <config_path>

# Run multiple cameras coordinate synchronization
mmla ips-ctag -c <config_path> # start a camera detector
mmla ips-csync -c <config_path> # start a camera sync manager

# Run real-time indoor position system
mmla ips-base -c <config_path> # start an ips base
mmla ips-sync -c <config_path> # start an ips base synchronizer
mmla ips-vis -c <config_path> # start an ips base visualizer
```