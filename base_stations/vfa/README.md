# Video Frame Analyzer (VFA)

This pipeline provides video frame analysis capabilities, including vision grounding (mapping AprilTag ID to person 
in image), vision captioning (give description about the image), and text classification to categorize each person's
actions into predefined categories.

## Pipeline Overview
<img src="docs/video_frame_analyzer.png" alt="video_frame_analyzer" width="800"/>

The system performs nonverbal behavior analysis using large language models (LLMs) and vision-language models (VLMs), processing image frames every 30 seconds:
- Image frame capture from base station (vfa-base)
- AprilTag detection and in-image coordinate calculation
- Prompt construction for VLM to describe gestures, postures, and interactions
- VLM output links AprilTag IDs to natural language descriptions
- Prompt construction for LLM using VLM-generated descriptions
- LLM classification of actions into predefined behavior categories
- Output generation in structured <ID, Action> dictionary format

## Usage Instructions

### Install Dependencies
```bash
# Install the required dependencies for the VFA pipeline on specific machines 
# (e.g., base station: vfa-base, base server: vfa-server, uber server: uber-server)
conda create -n vfa-base -c conda-forge python=3.10.12 -y
conda activate vfa-base
pip install -e .[vfa-base] # or pip install openmmla[vfa-base]
```


### On Servers
```bash
# 1. Run uber services on uber server with conda env `uber-server`
# Go to servers/uber/
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard

# 2. Run vfa services on base server with conda env `vfa-server`
# Edit your own config.yml file, see servers/vfa/config_template.yml for more details
# You can either run it via bash or python

# ==================BASH========================
# Start vfa services at once
# Go to /servers/vfa/bash
./run.sh


# =================PYTHON========================
# Start vaf services one by one
# Activate conda env `vfa-server`
conda activate vfa-server

## Option 1: run with single worker via mmla command
## e.g., 
## mmla vfa-vllm -c config.yml
mmla <vfa-server-commands> -c <config_file_path>
 
## Option 2: run with multiple workers via gunicorn
## e.g., 
## export CONFIG_FILE=config.yml
## gunicorn -k gevent -w 3 -b 0.0.0.0:5007 openmmla.commands.vfa.vllm:app
export CONFIG_FILE=<config_file_path>
gunicorn -k gevent -w <number-workers> -b 0.0.0.0:<port> <path-to-vfa-servers-app>:app
```

### On Base Stations
```bash
# Run vfa pipelines on base station with conda env `vfa-base`
# Edit your own config.yml file, see base_stations/vfa/config_template.yml for more details
# You can either run it via bash or python

# ===================BASH========================
# Go to /base_stations/vfa/bash
# Run real-time video frame analyzer
Usage: ./run.sh [-nb NUM_BASE] [-ns NUM_SYNCHRONIZER] [-g GRAPHICS] [-s STORE] [-v VERBOSE] [-h]

options:
  -nb NUM_BASE         : Number of VFA bases to run (default: 1)
  -ns NUM_SYNCHRONIZER : Number of synchronizers to run (default: 1)
  -g GRAPHICS          : Enable graphics (default: true)
  -v VERBOSE           : Enable verbose mode (default: false)
  -h                   : Display this help message
  
# ==================PYTHON========================
# Activate conda env `vfa-base`
conda activate vfa-base

# Run real-time video frame analyzer
mmla vfa-base -c <config_file_path> # start a vfa base
mmla vfa-sync -c <config_file_path> # start a vfa base synchronizer
```