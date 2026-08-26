# ASR with Diarization

This pipeline provides automatic speech recognition with speaker diarization capabilities. It processes audio streams in real-time to transcribe speech and identify different speakers.

## Pipeline Overview
<img src="docs/real_time_asr_analyzer.png" alt="real_time_asr_analyzer" width="800"/>

The real-time pipeline processes audio through several stages:
- Audio capture/input from microphones
- Voice activity detection (VAD) to identify speech segments
- Speech enhancement to clean audio signals
- Speaker diarization to separate and identify speakers
- Speech recognition to transcribe the audio to text
- Synchronization of results from multiple asr bases

For post-time processing, record raw audio with a Collection Session and replay it through
the same pipeline with `Base.<device>.source: file` (see Data Input Setup below).

## Usage Instructions

### Install Dependencies
```bash
# Install the required dependencies for the ASR pipeline on specific machines 
# (e.g., base station: asr-base, base server: asr-server-nemo or asr-server-wespeaker, uber server: uber-server)
conda create -n asr-base -c conda-forge -y python=3.10.12
conda activate asr-base
pip install -e .[asr-base] # or pip install openmmla[asr-base]
# Optional: if you would like to use lab streaming layer as an input source
pip install pylsl==1.17.6 
conda install -c conda-forge liblsl=1.16.2
```

For ASR server environments, choose one backend-specific extra.

```bash
pip install -e ".[asr-server-nemo]"       # NeMo speaker embedding backend
pip install -e ".[asr-server-wespeaker]"  # WeSpeaker backend, isolated to avoid dependency conflicts
```

### Data Input Setup

ASR Base supports the following audio input sources (configured via `Base.<device>.source` in `config.yml`):

| Source | Description | Device Setup |
|--------|-------------|-------------|
| `pyaudio` | USB microphone directly connected to the base station (e.g., Jabra Speak2 75, built-in mic) | Plug in the microphone; the base will prompt you to select the audio device and channel at startup |
| `udp` / `tcp` | Nicla Vision or Portenta H7 wearable badge streaming audio over Wi-Fi | Flash the firmware (`.ino`) onto the badge with the correct Wi-Fi credentials and ASR Base host/port in `arduino_secrets.h`. See `pipelines/wearables/nicla-vision/asr/` or `pipelines/wearables/portenta-h7/asr/` |
| `rtmp` | Audio stream pulled from an NGINX RTMP server | Add RTMP entries in the `Streams` section of `config.yml` with `target: rtmp://...`. The streaming device pushes audio to NGINX via ffmpeg |
| `lsl` | Lab Streaming Layer input | Configure `lsl_name` in `stream_kwargs`. Requires `pylsl` installed |
| `file` | Replay from previously recorded audio files | Set `file_dir` and `initial_sync_time` in config |

For wearable badges, the key configuration is in the firmware:
- **Wi-Fi**: SSID and password in `arduino_secrets.h`
- **Target**: ASR Base's IP address and port (base listens on `port_offset + base_id`)
- **Format**: Must match `stream_kwargs` in config (default: 16 kHz, mono, 16-bit PCM)

#### Stream Configuration

All stream sources (RTMP and remote devices) are configured in the unified `Streams` section of `config.yml`:

```yaml
Streams:
  # external RTMP stream (already running, Base only pulls from the URL)
  rtmp-mic-1:
    target: rtmp://uber-server.local/stream_01

  # managed stream (TUI starts/stops ffmpeg on remote Raspberry Pi via SSH)
  rpi-mic-1:
    ssh_profile: rpi-table-1        # must match a TUI SSH profile name
    device: hw:1,0                  # ALSA audio device on the remote machine
    target: udp://asr-base.local:5001
    format: s16le
    rate: 16000
    channels: 1
```

Streams with `ssh_profile` can be started/stopped from the TUI Launcher's **Streams** tab. Streams without `ssh_profile` are treated as external (already running).

### Run with the TUI (recommended)

Start the management console and use the Launcher:

```bash
mmla tui
```

1. **ASR Server** (on the AI server): edit the Config tab (`pipelines/asr-server/config.yml`), then Start — the TUI launches one tmux session per service (inference, VAD, enhancement, transcription, ...), locally or over SSH
2. **Pipelines → ASR → ASR Base** (on base stations): set session, mode, and processing options on the card, then Start — bases and synchronizer open in terminal tabs
3. **Streams tab**: start/stop remote ffmpeg streams defined in the `Streams` config section

### Manual CLI (alternative)

```bash
# Base station (conda env: asr-base)
mmla asr-base -c <config_path> -m full  # start an asr base (mode: record | recognize | full)
mmla asr-sync -c <config_path>          # start an asr synchronizer

# AI server (conda env: asr-server-nemo or asr-server-wespeaker)
# one gunicorn process per service from openmmla/services/asr/apps/
export CONFIG_FILE=<config_path>
gunicorn -k gevent -w <workers> -b 0.0.0.0:<port> openmmla.services.asr.apps.<serve_module>:app
```
