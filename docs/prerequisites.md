# System Prerequisites

Every machine that runs an OpenMMLA component needs the tools below: base stations, base servers (AI services) and the uber server (system services). A device that only pushes a camera or microphone stream, such as a Raspberry Pi, needs FFmpeg alone; see the [Raspberry Pi setup](raspi_config.md).

| Tool | Used for |
|---|---|
| Conda | one isolated Python environment per pipeline; the TUI's Environment tab creates and fills them |
| Git | cloning the repository; the TUI runs from a checkout and can clone it onto remote hosts |
| tmux | services and pipeline components run in detached tmux sessions, so they survive a closed terminal |
| PortAudio | audio input for ASR base stations and raw audio collection (PyAudio builds against it) |
| FFmpeg | RTMP streaming, raw audio/video collection and file replay |

## Conda

Miniforge is the recommended distribution: it is small, defaults to conda-forge, and ships builds for Apple Silicon and Raspberry Pi.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Open a new shell afterwards so `conda` is on your `PATH`. Any other conda distribution (Miniconda, Anaconda) works too.

## Git, tmux, PortAudio and FFmpeg

### macOS

Install [Homebrew](https://brew.sh) first if it is missing, then:

```bash
brew install git tmux portaudio ffmpeg
```

### Ubuntu, Debian and Raspberry Pi OS

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git tmux ffmpeg portaudio19-dev python3-pyaudio libsndfile1
```

`build-essential` and `portaudio19-dev` are what `pip install pyaudio` compiles against inside the pipeline environments; `libsndfile1` is needed by librosa and soundfile.

## Optional tools

- **Docker Engine** (plus the NVIDIA container toolkit on GPU hosts): runs the ASR/VFA AI services and, optionally, InfluxDB and MongoDB as containers. See the [Docker guide](docker.md).
- **sshpass**: only for TUI SSH profiles that authenticate with a password; key-based profiles do not need it. `sudo apt install sshpass` on Debian/Ubuntu. On macOS it is not in Homebrew core: `brew install hudochenkov/sshpass/sshpass`.
- **Lab Streaming Layer**: only for `lsl` audio/video sources. Inside the pipeline environment run `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.
- **NVIDIA driver and CUDA**: for GPU inference on base servers. See the [FAQ](faq.md#nvidia-driver-installation).

## Check

```bash
conda --version && git --version && tmux -V && ffmpeg -version | head -1
```

## Official links

- Conda: [Miniforge](https://github.com/conda-forge/miniforge), [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git: https://git-scm.com/
- tmux: https://github.com/tmux/tmux/wiki/Installing
- PortAudio: https://www.portaudio.com/
- FFmpeg: https://ffmpeg.org/
