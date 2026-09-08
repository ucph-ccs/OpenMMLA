# ASR with Diarization

Automatic speech recognition with speaker diarization. Audio from microphones or wearable badges is split into speech segments, attributed to registered speakers and transcribed in real time, and the results of every base in a session are synchronized into one timeline.

## Pipeline overview

![Real-time ASR analyzer](../img/real_time_asr_analyzer.png)

1. Audio capture from a microphone, a badge stream, an RTMP stream, LSL, or a recorded file
2. Voice activity detection to isolate speech segments
3. Speech enhancement, and optionally speech separation, to clean the signal
4. Speaker recognition against the speaker profiles registered on the base
5. Transcription of each segment
6. Synchronization of the results of all bases into time buckets, written to InfluxDB as `asr_recognition` and `asr_transcription` events

| Component | Runs on | Command | Environment |
|---|---|---|---|
| ASR Base, one per microphone | base station | `mmla asr-base` | conda env `asr-base` |
| ASR Synchronizer, one per session | base station | `mmla asr-sync` | conda env `asr-base` |
| ASR Server: audio inferer, resampler, speech enhancer, speech separator, speech transcriber, voice activity detector | GPU base server | docker compose | images in `docker/` |

Create the `asr-base` environment from the TUI's Environment tab or by hand (`conda create -n asr-base python=3.10 -y && pip install -e '.[asr-base]'`). For an `lsl` source add `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.

## Configuration

`pipelines/asr-base/config.yml` is the base station config; `config_template.yml` documents every key. Open the **Config** tab of the ASR Base card and press **Save** to create it from the template, or copy the template by hand.

| Section | What it holds |
|---|---|
| `Base.<device>` | one block per kind of microphone: recognition thresholds and durations, gain, VAD thresholds, `port_offset` for badge streams, `file_dir` for file replay, and `stream_kwargs` (channels, rate, format, chunk size). Add one with `+ Add Base`. |
| `Bases` | the list of base nodes. Each entry has an `id`, the `base_type` it uses from `Base`, a `source`, a `source_index`, an optional `channel`, and a `port` for udp/tcp sources. Every base you start picks one of these entries. |
| `Synchronizer` | `bucket_duration`, `match_tolerance`, `result_expiry_time` |
| `Streams` | managed and external streams, see below |
| `Server.asr` | the six service endpoints, either through the gateway (`http://<gateway>:8080/transcribe`) or direct (`http://<server>:5005/transcribe`) |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings, see [System Services](../system_services.md) |

### Input sources

`Bases[].source` selects where a base reads audio from:

| Source | Description | Setup |
|---|---|---|
| `pyaudio` | USB or built-in microphone on the base station (Jabra Speak2 75, laptop mic, ...) | `source_index` is the PyAudio device index; `channel` picks the input channel of a multi-channel device. Leave them unset to be asked at startup. |
| `udp` / `tcp` | Nicla Vision or Portenta H7 badge streaming over Wi-Fi | flash the firmware under `pipelines/wearables/nicla-vision/asr/` or `pipelines/wearables/portenta-h7/asr/` with the Wi-Fi credentials and the base's host and `port` in `arduino_secrets.h`; the badge format must match `stream_kwargs` (16 kHz, mono, 16-bit PCM by default) |
| `rtmp` | audio pulled from the Nginx RTMP server | add a `Streams` entry with `target: rtmp://...`; `source_index` is the position of that stream among the RTMP entries |
| `lsl` | Lab Streaming Layer | `source_index` is the LSL stream name; needs `pylsl` |
| `file` | replay of a recorded file | `source_index` is the file name inside `Base.<device>.file_dir`; the start time is read from the file name (`<prefix>_<timestamp>.wav`), so the sync time needs no configuration |

### Streams

Every stream a base pulls from is declared once under `Streams`. An entry with an `ssh_profile` is *managed*: the TUI starts and stops an FFmpeg process on that host from the **Streams** tab. An entry without one is *external*, already running somewhere.

```yaml
Streams:
  # external RTMP stream (already running, the base only pulls from the URL)
  rtmp-mic-1:
    target: rtmp://uber-server.local/stream_01

  # managed stream: the TUI starts/stops ffmpeg on a remote Raspberry Pi over SSH
  rpi-mic-1:
    ssh_profile: rpi-table-1        # must match a TUI SSH profile name
    device: hw:1,0                  # ALSA audio device on the remote machine
    target: udp://asr-base.local:5001
    format: s16le
    rate: 16000
    channels: 1
```

See [RTMP Streaming](../rtmp_streaming.md) for the FFmpeg commands behind this.

## Run from the TUI

1. **ASR Server**: `Launcher → Pipelines → ASR → ASR Server`, Host set to the GPU server. Fill in `pipelines/asr-server/config.yml` on the Config tab (speaker embedding backend and model, transcription backend, which services use CUDA), tick the services to launch, and press **Start**. The card runs `docker compose -f docker/docker-compose.asr.yml up -d --build` for those services; the first start builds the images and downloads the models. Details in the [Docker guide](../docker.md).
2. **System services** running and reachable, and `Server.asr` pointing at the server or the gateway. Verify with **Refresh** on the System Services cards.
3. **ASR Base**: Host set to the base station. On the Launch tab choose the number of bases and synchronizers, the **Session** (or `Create MongoDB Session` from an experiment group), the **Mode** and the toggles (store audio, VAD, noise reduction, transcribe, speech separation, dominant speaker, half-scaled recognition). **Start** opens one terminal window per base and synchronizer. Each base asks which `Bases` entry it is and shows its menu: edit the speaker profiles (register and select speakers), switch mode, start.
4. **Session Control**: once every window reports that it is waiting, send **START** for the session; send **STOP** at the end.

Modes: `live` recognizes and transcribes the stream in real time (the TUI default); `capture` only records the audio to `records/` for later; `analyze` processes the `.wav` files found in `records/`. With `asr_scope: participant` (the default) speaker verification is on and each base needs registered speaker profiles before `live` or `analyze` can run; `asr_scope: group` attributes the transcript to the group without verification.

## Manual CLI

```bash
conda activate asr-base
# one per microphone; -b picks the Bases entry, -sid the session (both asked interactively when omitted)
mmla asr-base -p pipelines/asr-base -c pipelines/asr-base/config.yml -m live -sid <session-id> -b <base-id>
# one per session
mmla asr-sync -p pipelines/asr-base -c pipelines/asr-base/config.yml -sid <session-id>
```

`mmla asr-base -h` lists the toggles (`-s`, `-vad`, `-nr`, `-tr`, `-sp`, `-hsr`). The START/STOP signals can be sent with `mmla ses-ctl -c pipelines/asr-base/config.yml` instead of the TUI.

To run the server services without Docker, install one backend extra (`asr-server-wespeaker` or `asr-server-nemo`) and start one gunicorn process per service from `openmmla/services/asr/apps/`. The wespeaker backend itself is not on PyPI, so it is installed separately at the commit the Docker image uses:

```bash
pip install -e '.[asr-server-wespeaker]'
pip install "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@9ce7995648a281ba6ce6f4e7a33941672ef22779"

export PROJECT_DIR=pipelines/asr-server CONFIG_PATH=pipelines/asr-server/config.yml
gunicorn -k gevent -w 1 -b 0.0.0.0:5005 openmmla.services.asr.apps.serve_speech_transcriber:app
```

The modules are `serve_audio_inferer` (5001), `serve_audio_resampler` (5002), `serve_speech_enhancer` (5003), `serve_speech_separator` (5004), `serve_speech_transcriber` (5005) and `serve_voice_activity_detector` (5006).

## Post-time processing

Record the session first with **Collection → Collection Session** (see the [TUI guide](../tui.md#collection)), then replay it: set the base's `source` to `file`, `Base.<device>.file_dir` to the `audio/` directory listed in the collection manifest, and `source_index` to the file name. The Config tab offers the files in `file_dir` as a dropdown. Run the pipeline as usual; the timing is taken from the file names.
