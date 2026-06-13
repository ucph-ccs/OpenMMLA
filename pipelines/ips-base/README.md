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

### Run with the TUI (recommended)

Start the management console and go to **Launcher → Pipelines → IPS**:

```bash
mmla tui
```

1. **Config tab**: edit `config.yml` (see `config_template.yml` for all options)
2. **IPS Camera Calibration**: compute camera intrinsic parameters (required once per camera)
3. **IPS Camera Sync**: run tag detectors + sync manager to align coordinate systems across cameras
4. **IPS Base**: launch bases, synchronizer, and visualizer for the session
5. **Transform Matrix tab**: review and sync the calibration matrices to remote hosts

Infrastructure services (InfluxDB, Redis, MQTT, ...) are launched from the same Launcher under **Infrastructure**.

### Manual CLI (alternative)

```bash
conda activate ips-base

# Camera intrinsic calibration
mmla ips-ccal -c <config_path>

# Multi-camera coordinate synchronization
mmla ips-ctag -c <config_path>   # start a camera detector
mmla ips-csync -c <config_path>  # start a camera sync manager

# Real-time indoor positioning system
mmla ips-base -c <config_path>   # start an ips base
mmla ips-sync -c <config_path>   # start an ips synchronizer
mmla ips-vis -c <config_path>    # start an ips visualizer
```
