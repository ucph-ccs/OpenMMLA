# System prerequisites

These are the tools every machine that runs an OpenMMLA component needs: the console, the base stations, the GPU server and the uber server. Install them before the [Quickstart](quickstart.md); a device that only pushes a camera or microphone stream needs FFmpeg alone ([Raspberry Pi](raspi_config.md)).

| Tool | Used for |
|---|---|
| [Conda](https://github.com/conda-forge/miniforge) | one Python environment per pipeline; the console's Environment tab creates and fills them |
| [Git](https://git-scm.com/) | cloning the repository; the console runs from a clone and can clone it onto other machines |
| [tmux](https://github.com/tmux/tmux/wiki/Installing) | the dashboard and its worker, the MLLM Server, a native MediaMTX and the streams run in detached tmux sessions, so they survive a closed terminal |
| [PortAudio](https://www.portaudio.com/) | audio input on ASR base stations; PyAudio builds against it |
| [FFmpeg](https://ffmpeg.org/) | streaming to MediaMTX, an ASR base pulling its stream, Collection recordings, file replay and the session commands |

!!! tip "Let the console install them"
    Once conda is on a machine, the console's Environment tab names in its `System` column what the machine lacks for each env, and **Install Tools** installs it with apt or Homebrew ([Environment tab](tui/environment.md)).

## Conda

Install Miniforge, which is small, defaults to conda-forge, and has builds for Apple Silicon and Raspberry Pi:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Open a new shell afterwards, so that `conda` is on your `PATH`. Any other conda distribution, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda, works too.

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

`build-essential` and `portaudio19-dev` are what `pip install pyaudio` compiles against inside the pipeline environments; `libsndfile1` is what librosa and soundfile load.

## Optional tools

| Tool | Needed for | Install |
|---|---|---|
| Docker Engine, and the NVIDIA container toolkit on the GPU server | the ASR and VFA servers, and InfluxDB, MongoDB and MediaMTX in containers | [Docker guide](docker.md) |
| `sshpass` | SSH profiles that log in with a password; key logins do not need it | `sudo apt install sshpass`, or `brew install sshpass` |
| Lab Streaming Layer | `lsl` audio and video sources only | inside the pipeline env: `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2` |
| NVIDIA driver and CUDA | GPU inference on the GPU server | [FAQ](faq.md#nvidia-driver-installation) |

## Check

```bash
conda --version && git --version && tmux -V && ffmpeg -version | head -1
```
