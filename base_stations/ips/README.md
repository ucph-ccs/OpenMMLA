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
# (e.g., base stations: ips-base, uber servers: uber-server)
conda create -n ips-base -c conda-forge python=3.10.12 -y
conda activate ips-base
pip install openmmla[ips-base]
```


### On Servers
```bash
# Run uber services on your machine with uber-server env
# Go to servers/uber/
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard
```

### On Base Stations
```bash
# Run ips pipeline on your machine with ips-base env
# Edit your own config.yml file, see base_stations/ips/config_template.yml for more details
# You can either run it via bash or python

# ===================BASH========================
# For camera intrinsic calibration
usage: ./calibrate.sh

# For multiple cameras coordinate synchronization
usage: ./synchronize.sh [-nc] 2 [-ns] 1 [-h]

options:
  -nc  NUM_CAMERA             : Number of camera detectors to run (default: 3)
  -ns  NUM_SYNCMANAGER       : Number of sync managers to run (default: 1)
  -h                          : Display this help message

# For real-time indoor position system
Usage: ./run.sh [-nb NUM_BASE] [-ns NUM_SYNCHRONIZER] [-g GRAPHICS] [-s STORE] [-h]

options:
  -nb NUM_BASE         : Number of IPS bases to run (default: 3)
  -ns NUM_SYNCHRONIZER : Number of synchronizers to run (default: 1)
  -g GRAPHICS          : Enable graphics (default: true)
  -s STORE             : Enable store (default: false)
  -v VERBOSE           : Enable verbose mode (default: false)
  -h                   : Display this help message

# ==================PYTHON========================
# Activate ips-base environment
conda activate ips-base

# For camera intrinsic calibration
mmla ips-ccal [-h] [-p project_dir] -c config_path

# For multiple cameras coordinate synchronization
mmla ips-ctag [-h] [-p project_dir] -c config_path # start a camera detector
mmla ips-csync [-h] [-p project_dir] -c config_path # start a camera sync manager

# For real-time indoor position system
mmla ips-base [-h] [-p project_dir] -c config_path [-g graphics] [-s store] [-v verbose] # start an ips base
mmla ips-sync [-h] [-p project_dir] -c config_path [-v verbose] # start an ips base synchronizer
mmla ips-vis [-h] [-p project_dir] -c config_path [-s store] # start an ips base visualizer
```