# ASR with Diarization

This pipeline provides automatic speech recognition with speaker diarization capabilities. It processes audio streams in real-time to transcribe speech and identify different speakers.

## Pipeline Overview

The pipeline processes audio through several stages:
- Audio capture/input from microphones
- Voice activity detection (VAD) to identify speech segments
- Speech enhancement to clean audio signals
- Speaker diarization to separate and identify speakers
- Speech recognition to transcribe the audio to text
- Synchronization of results from multiple audio bases

## Usage Instructions

### Server
```bash
# On your uber-server machine, run services (required: InfluxDB, Redis, MQTT, Nginx; optional: Nginx, NEXT.js, Flask, Celery)
# Go to servers/uber/
make all # if start all services 
make all -without=nginx, celery, flask, next # add your exlcuded services in without options

# On your asr-server machine, run asr services, make sure you have installed the asr-server environment with conda
# Go to servers/asr/ and edit the config.yml file
# Go to servers/asr/bash and run the bash script
./run.sh

# You can also run the services individually
# Activate asr-server environment
conda activate asr-server
# with openmmla command, only for single worker
openmmla asr-infer -c config.yml

# with gunicron, for multiple workers
export CONFIG_FILE=config.yml
gunicorn -k gevent -w <number-worker> -b 0.0.0.0:<port> <asr-server>:app
e.g.,
gunicorn -k gevent -w 3 -b 0.0.0.0:5001 openmmla.commands.asr.asr_infer:app
```

### Base Station
```base
# On your base-station machine
# Go to base_stations/asr/
# Edit the config.yml file
# Go to base_stations/asr/bash and run the bash script
./run.sh
```






