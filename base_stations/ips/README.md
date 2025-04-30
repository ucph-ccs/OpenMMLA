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
```


### On Servers
```bash
# 1. Run uber services on uber server with conda env `uber-server`
# Go to /servers/uber/ to run with scripts or run manually with brew or systemctl
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard
```

### On Base Stations
```bash
# Run ips pipelines on base station with conda env `ips-base`
# Edit your own config.yml file, see base_stations/ips/config_template.yml for more details
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