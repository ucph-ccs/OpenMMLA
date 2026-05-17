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
# Optional: if you would like to use lab streaming layer as an input source
pip install pylsl==1.17.6 
conda install -c conda-forge liblsl=1.16.2
```

### LLM and VLM Backend Options

VFA requires backend services to run large language models (LLMs) and vision-language models (VLMs). You can choose either to run these models locally on your own hardware or connect to cloud services:

#### Local Backend Options
- **vLLM**: Efficient inference for LLMs with optimized attention algorithms
  - Installation and setup guide: [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
  - For local VFA inference, install the OpenMMLA wrapper and local vLLM runtime extras: `pip install -e ".[vfa-server,vfa-vllm-runtime]"`
  - Example: `vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt '{"image":4}'`
  - From the TUI Launcher, use **VFA → MLLM Server** to run the local vLLM model service and **VFA → VFA Server** to run the OpenMMLA wrapper.
  
- **Ollama**: Simplified local deployment of open-source models
  - Installation: [Ollama Download](https://ollama.com/download)
  - Example: `ollama pull llava` to download a multimodal model

#### Cloud API Options
For these options, you'll need to obtain API keys from the respective providers and configure them in your `config.yml`:

- **OpenAI API**: Access to GPT-4 Vision and other OpenAI models
  - Requires an [OpenAI API key](https://platform.openai.com)
  
- **Google Gemini**: Google's multimodal LLM 
  - Requires a [Gemini API key](https://ai.google.dev/)
  
- **DeepSeek**: DeepSeek's vision-language models
  - Requires a [DeepSeek API key](https://platform.deepseek.com)
  
- **Qwen**: Alibaba's multimodal models via ModelScope
  - Requires a [Qwen API key](https://help.aliyun.com/document_detail/611472.html)

Configure your chosen backend in the `config.yml` file under the appropriate VLM/LLM section.

### Customizing Prompt Templates

VFA now supports loading custom prompt templates from external files, allowing you to modify the prompts without changing the code:

1. Add the `prompt_templates_dir` parameter to your `config.yml` file:
   ```yaml
   VLLMFrameAnalyzer:
     prompt_templates_dir: "prompts"  # Path to your prompt templates directory
     # Other configuration options...
   ```

2. Create the templates directory and add template files:
   ```bash
   mkdir -p /path/to/your/prompts
   ```

3. Create the following template files:
   - `multi_angle_end_system_prompt.txt`: System prompt for end-to-end analysis (both observation and classification)
   - `multi_angle_end_user_prompt.txt`: User prompt for end-to-end analysis
   - `multi_angle_vlm_system_prompt.txt`: System prompt for vision-only observations
   - `multi_angle_vlm_user_prompt.txt`: User prompt for vision-only observations
   - `multi_angle_llm_system_prompt.txt`: System prompt for classification-only
   - `multi_angle_llm_user_prompt.txt`: User prompt for classification-only

   All templates can use variable substitution with the syntax `{{variable_name}}`. Supported variables:
   - `{{num_perspectives}}`: Number of camera angles being analyzed
   - `{{angle_descriptions}}`: Descriptions of each camera perspective
   - `{{participant_descriptions}}`: Descriptions of known participants
   - `{{action_definitions}}`: Definitions of actions to classify
   - `{{image_description}}`: VLM observations (only for LLM prompts)

   Use the variable directly in the template file, e.g. `Here below is the angle descriptions for the images: {{angle_descriptions}}`

### Data Input Setup

VFA Base supports the following video input sources (configured via `Base.source` in `config.yml`):

| Source | Description | Device Setup |
|--------|-------------|-------------|
| `opencv` | USB camera directly connected to the base station or Raspberry Pi | Plug in the camera; the base will detect available video devices and prompt you to select at startup |
| `rtmp` | Video stream pulled from an NGINX RTMP server | Add RTMP entries in the `Streams` section of `config.yml` with `target: rtmp://...`. The streaming device pushes video to NGINX via ffmpeg |
| `file` | Replay from previously recorded video files | Set `file_dir` and `initial_sync_time` in config |

VFA captures frames at a configurable interval (`keyframe_interval`, default 30s) for VLM/LLM analysis, rather than processing every frame. Configure `angle_config` in `config.yml` to describe camera perspectives for multi-angle analysis.

#### Stream Configuration

All stream sources (RTMP and remote devices) are configured in the unified `Streams` section of `config.yml`:

```yaml
Streams:
  # external RTMP stream (already running, Base only pulls from the URL)
  cam-external:
    target: rtmp://uber-server.local/vfa/side

  # managed stream (TUI starts/stops ffmpeg on remote Raspberry Pi via SSH)
  cam-front:
    ssh_profile: rpi-front          # must match a TUI SSH profile name
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local/vfa/front
    codec: libx264
    resolution: 1920x1080
    fps: 30
```

Streams with `ssh_profile` can be started/stopped from the TUI Launcher's **Streams** tab. Streams without `ssh_profile` are treated as external (already running).

### On Servers

> **Tip**: You can use `mmla tui` to configure and launch all services from the TUI Launcher, instead of running commands manually.

```bash
# 1. Run uber services on uber server with conda env `uber-server`
# Go to /pipelines/uber-server/ to run with scripts or run manually with brew or systemctl
make all # if start all services 
make all -without=nginx,celery,flask,next # if start without nginx(load balancer, RTMP) and dashboard

# 2. Run vfa services on base server with conda env `vfa-server`
# Edit your own config.yml file, see pipelines/vfa-server/config_template.yml for more details
# You can either run it via bash or python

# ==================BASH========================
# Start vfa services at once
# Go to /pipelines/vfa-server/bash
./run.sh


# =================PYTHON========================
# Start vfa services one by one; Flask apps live under openmmla/services/vfa/apps/.
# Activate conda env `vfa-server`
conda activate vfa-server

## Option 1: run with single worker via mmla command
## e.g., 
## mmla vfa-vllm -c config.yml
mmla <vfa-server-commands> -c <config_file_path>
 
## Option 2: run with multiple workers via gunicorn
## e.g., 
## export CONFIG_FILE=config.yml
## gunicorn -k gevent -w 3 -b 0.0.0.0:5007 openmmla.services.vfa.apps.serve_multi_angle_vllm_frame_analyzer:app
export CONFIG_FILE=<config_file_path>
gunicorn -k gevent -w <number-workers> -b 0.0.0.0:<port> openmmla.services.vfa.apps.<serve_module>:app
```

### On Base Stations
```bash
# Run vfa pipelines on base station with conda env `vfa-base`
# Edit your own config.yml file, see pipelines/vfa-base/config_template.yml for more details
# Prefer the mmla commands below; examples/run_*.py launchers were removed.
# You can either run it via bash or python

# ===================BASH========================
# Go to /pipelines/vfa-base/bash
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
