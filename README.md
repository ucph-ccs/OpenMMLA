# OpenMMLA

OpenMMLA a toolkit for multimodal learning analytics, providing various built-in pipelines for different tasks. The toolkit is for building up the MMLA pipeline as shown below:

<img src="docs/high_level_system_design.png" alt="OpenMMLA system design" width="100%">

<details>
<summary><strong>High-level System Design (Data flow)</strong></summary>

The platform's high-level design consists of three stages: input, processing, and output. 
+ **Data input stage (purple)**, multimodal raw data from sensors & wearable badges are streamed directly to base stations or a central media server. The raw inputs are transformed into structured, coded streams for efficient transmission and processing. 
+ **Data processing stage (black)**, these encoded streams are processed individually by the corresponding *(Audio/Video/..)Base*, which handles some signal processing, while more complex tasks are offloaded to the server. For *Bases* within the same group, results are synchronized and uploaded to the time series database, where segment-level measurement features are generated.
+ **Data output stage (lime)**, these measurement features are visualized on the dashboard in real time and combined into indicators to analyze group interactions. The platform also generates post-processing visualizations, logs, and reports, which are stored and accessible via the shared dashboard, enabling both real-time awareness and retrospective analysis of group dynamics.
</details>

## System Architecture

OpenMMLA consists of several hardware components working together:
- **Sensors**: wearable devices and distributed environmental sensors for data acquisition. As for wearable devices, we support: 
   + *Voice-Badge*: [Nicla Vision board](https://docs.arduino.cc/hardware/nicla-vision/) with power supply
   + *Vision-Badge*: Nicla Vision board with [AprilTag](https://april.eecs.umich.edu/software/apriltag) and power supply
   + *Regular-Badge*: AprilTag only
- **Base Stations**: microprocessors/PCs that processes various data streams. Each base station runs one or more instances of *Base* and *Synchronizer*, with specific types (e.g., *AudioSynchronizer*) synchronizing data from corresponding *Base* components (e.g., *AudioBase*).
- **Servers**: powerful PCs that provides centralized services for other devices within distributed environments. Based on functionality, it can be divided into:
   + *Base Servers*: REST servers for AI services (infer, transcribe, vad, vlm... via Flask/FastAPI).
   + *Uber Servers*: other services like database (InfluxDB), Messaging (Redis, MQTT), RTMP streaming & Load balancing (Nginx), dashboard application (Next.js & Flask).
- **Dashboard**: web-page interfaces accessible via phone and web browsers, featuring on session selection, real-time visualizations, post-time visualizations and measurements downloads.

<details>
<summary><strong>Detailed Architecture</strong></summary>

![mBox System Design](docs/system_architecture.png)

</details>

## Quick Setup

The setup requirements depend on which part of the system you're implementing:

### System Dependencies 

These tools are required for **Base Stations** and **Base Servers** to process audio/video data:

- **Python Environment**: [Conda](https://docs.conda.io/en/latest/miniconda.html)
- **Terminal Multiplexer**: [tmux](https://github.com/tmux/tmux/wiki/Installing)
- **Audio Processing**: [PortAudio](http://www.portaudio.com/download.html)
- **Video Processing**: [FFmpeg](https://ffmpeg.org/download.html)

<details>
<summary>Conda Installation</summary>

```bash
# macOS & Ubuntu
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname)-$(uname -m).sh"
bash Miniconda3-latest-$(uname)-$(uname -m).sh

# Debian (Raspberry Pi)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
</details>

<details>
<summary>Others Installation</summary>

```bash
# macOS
brew install ffmpeg portaudio tmux
echo 'export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"' >> ~/.zshrc
source ~/.zshrc

# Ubuntu
sudo apt update && sudo apt upgrade
sudo apt install build-essential git ffmpeg python3-pyaudio libsndfile1 libasound-dev tmux
wget https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar -zxvf pa_stable_v190700_20210406.tgz
cd portaudio
./configure && make
sudo make install

# Debian (Raspberry Pi Bullseye or later)
sudo apt update && sudo apt upgrade
sudo apt install -y build-essential git ffmpeg python3-pyaudio libsndfile1 portaudio19-dev tmux
```
</details>

### Required Services 

These services are primarily required for the **Uber Servers** which acts as the central hub:

- **InfluxDB**: Time series database for storing measurements data(Required)
- **Redis**: Message broker and cache (Required)
- **Mosquitto**: MQTT broker for publish/subscribe messaging (Required)
- **Nginx**: RTMP server and load balancer (Optional)
- **Dashbaord** NEXT.js frontend & Flask backend server (Optional)

<details>
<summary>Services Installation</summary>

#### InfluxDB Installation
```bash
# For macOS
brew install influxdb
brew services start influxdb

# For Ubuntu/Debian
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update && sudo apt install influxdb2
sudo systemctl enable influxdb
sudo systemctl start influxdb

# Go to http://localhost:8086, and follow the instructions to create admin user with operator API token, save your token in a safe place, 
# it will be used for config later.
```

#### Redis Installation
```bash
# For macOS
brew install redis
# Edit /opt/homebrew/etc/redis.conf: set "protected-mode no" and "bind 0.0.0.0"
brew services restart redis

# For Ubuntu/Debian
sudo apt install -y redis-server
# Edit /etc/redis/redis.conf: set "protected-mode no" and "bind 0.0.0.0"
sudo systemctl enable redis-server
sudo systemctl restart redis-server
```

#### Mosquitto Installation
```bash
# For macOS
brew install mosquitto
# Edit /opt/homebrew/etc/mosquitto/mosquitto.conf: add "listener 1883 0.0.0.0" and "allow_anonymous true"
brew services restart mosquitto

# For Ubuntu/Debian
sudo apt install -y mosquitto
# Edit /etc/mosquitto/mosquitto.conf: add "listener 1883 0.0.0.0" and "allow_anonymous true"
sudo systemctl enable mosquitto
sudo systemctl restart mosquitto
```

#### Nginx Installation (Optional)
For detailed Nginx installation and configuration as both a load balancer and RTMP server, please refer to the [Nginx Setup Guide](docs/nginx.md).


#### Dashboard Installation (Optional)
For detailed instructions on setting up the Next.js frontend and Flask backend for the dashboard, please refer to the [Dashboard Setup Guide](docs/dashboard.md).

</details>

### OpenMMLA Installation

You can install OpenMMLA in two ways:

#### Option 1: Via Git
```bash
# Clone the repository
git clone https://github.com/ucph-ccs/openmmla.git
cd openmmla

# Create conda environment with specific envs 
# e.g., asr-base, asr-server, ips-base, vfa-server
conda create -n <your-env-name> -c conda-forge python=3.10.12 -y
conda activate <your-env-name> 

# Install base package
pip install -e .

# Install specific dependencies for specific envs
# For uber servers with uber-server env
pip install -e .[uber-server]

# For base stations with asr-base env
pip install -e .[asr-base] 

# For base servers with asr-server env
pip install -e .[asr-server]
 
# For base stations with ips-base env
pip install -e .[ips-base]  

# For base servers with vfa-server env
pip install -e .[vfa-server] 
```

#### Option 2: Via PyPI
```bash
# Create conda environment with specific envs
# e.g., asr-base, asr-server, ips-base, vfa-server
conda create -n <your-env-name> -c conda-forge python=3.10.12 -y
conda activate <your-env-name> 

# Install base package
pip install openmmla

# Install specific dependencies for specific envs
# For uber servers with uber-server env
pip install openmmla[uber-server]

# For base stations with asr-base env
pip install openmmla[asr-base] 

# For base servers with asr-server env
pip install openmmla[asr-server]
 
# For base stations with ips-base env
pip install openmmla[ips-base]  

# For base servers with vfa-server env
pip install openmmla[vfa-server]
```

### Pipeline Setup

After installing OpenMMLA and its dependencies, you can set up specific pipelines:

1. **Automatic Speech Recognition (ASR) with Diarization**
   - See [ASR Pipeline](base_stations/asr/README.md) for detailed setup and usage instructions.

2. **Indoor Positioning System (IPS)**
   - See [IPS Pipeline](base_stations/ips/README.md) for detailed setup and usage instructions.

3. **Video Frame Analyzer (VFA)**
   - See [VFA Pipeline](base_stations/vfa/README.md) for detailed setup and usage instructions.

Each pipeline has specific usage instructions and configuration options detailed in their respective documentation.

## [FAQ](docs/faq.md)

## Citation

If you use this project in your research, please cite the following paper:

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