# Video Frame Analyzer (VFA)

This pipeline provides video frame analysis capabilities, including vision grounding (mapping AprilTag ID to person 
in image), vision captioning (give description about the image), and text classification to categorize each person's
actions into predefined categories.

## Pipeline Overview
![video_frame_analyzer](docs/video_frame_analyzer.png)

## Usage Instructions

### Install Dependencies
```bash
# Install the required dependencies for the VFA pipeline on specific machines (e.g., base stations : vfa-base,
# vfa servers -> vfa-server, uber servers -> uber-server)
conda create -n vfa-base -c conda-forge python=3.10.12 -y
pip install openmmla[vfa-base]
```


### On Servers
```bash
# Run uber services on your machine with uber-server env
# Go to servers/uber/
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard

# Run vfa services on your machine with vfa-server env
# Edit your own config.yml file, see servers/vfa/config_template.yml for more details
# You can either run it via bash or python

# ==================BASH========================
# Start vfa services at once via servers/vfa/bash/run.sh, you can edit the weight and port of each server inside the run script file
./run.sh


# =================PYTHON========================
# Activate vfa-server environment
conda activate vfa-server

# Start vaf services one by one
# 1. VFA server with single worker via openmmla command
# e.g.,
# openmmla vfa-infer -c config.yml
openmmla <vfa-server-commands> -c <config_file_path>
 
# 2. ASR server with multiple workers via gunicorn
# e.g.,
# export CONFIG_FILE=config.yml
# gunicorn -k gevent -w 3 -b 0.0.0.0:5007 openmmla.commands.vfa.vfa_vlm:app
export CONFIG_FILE=config.yml
gunicorn -k gevent -w <number-workers> -b 0.0.0.0:<port> <path-to-vfa-servers-app>:app
```

### On Base Stations
```bash
# Edit your own config.yml file, see base_stations/vfa/config_template.yml for more details
# You can either run it via bash or python
```