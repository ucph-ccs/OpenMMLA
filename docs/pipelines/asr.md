# ASR with Diarization

Automatic speech recognition with speaker diarization. Audio from microphones or wearable badges is split into speech segments, attributed to registered speakers and transcribed in real time, and the results of every base in a session are synchronized into one timeline.

## Pipeline overview

![Real-time ASR analyzer](../img/real_time_asr_analyzer.png)

1. Audio capture from a microphone, a badge stream, a MediaMTX stream, LSL, or a recorded file
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
| `pyaudio` | USB or built-in microphone on the base station (Jabra Speak2 75, laptop mic, ...) | `source_index` is the PyAudio device index (required); `channel` picks the input channel of a multi-channel device, all channels when unset. |
| `udp` / `tcp` | Nicla Vision or Portenta H7 badge streaming over Wi-Fi, or a managed FFmpeg stream from a Raspberry Pi (see [Streams](#streams)) | badges: flash the firmware under `pipelines/wearables/nicla-vision/asr/` or `pipelines/wearables/portenta-h7/asr/` with the Wi-Fi credentials and the base's host and `port` in `arduino_secrets.h`. The sender's format must match `stream_kwargs` (16 kHz, mono, 16-bit PCM by default). The `audio_streaming_udp_ms` firmware prefixes every packet with an 18-byte header carrying the badge's clock; the other firmware and FFmpeg send header-less PCM. The base tells the two apart on the first packet (`stream_kwargs.packet_format: auto`); set it to `timestamped` or `raw` to force one. |
| `stream` | audio pulled from the MediaMTX server (`rtmp` is the old name) | a `Streams` entry whose `read_target` (else `target`) is an `rtmp://`, `rtsp://` or `srt://` URL; `source_index` names that entry (a number is read as its position among them, as older configs have it). Left empty while there are several, the base asks for it in its window |
| `lsl` | Lab Streaming Layer | `source_index` is the LSL stream name; needs `pylsl` |
| `file` | replay of a recorded file | `source_index` is the file name inside `Base.<device>.file_dir`; the start time is read from the file name (`<prefix>_<timestamp>.wav`), so the sync time needs no configuration |

### Streams

Every stream a base pulls from is declared once under `Streams`. An entry with an `ssh_profile` is *managed*: the TUI starts and stops an FFmpeg process on that host from the **Streams** tab. An entry without one is *external*, already running somewhere.

```yaml
Streams:
  # external stream (already running; the base pulls read_target, else target)
  mic-external:
    target: rtmp://uber-server.local:1935/asr/mic1
    read_target: rtsp://uber-server.local:8554/asr/mic1

  # managed stream straight to this base: the TUI starts/stops ffmpeg on a Raspberry Pi over SSH
  rpi-mic-1:
    ssh_profile: rpi-table-1        # must match a TUI SSH profile name
    device: hw:1,0                  # ALSA audio device on the remote machine
    target: udp://asr-base.local:5001
    format: s16le
    rate: 16000
    channels: 1
    record: true                    # also keep a wav on the Pi for later replay

  # managed stream through MediaMTX (AAC), pulled as RTSP by a `stream` source
  rpi-mic-2:
    ssh_profile: rpi-table-2
    device: hw:1,0
    kind: audio
    target: rtmp://uber-server.local:1935/asr/mic2
    read_target: rtsp://uber-server.local:8554/asr/mic2
```

For a `udp://` or `tcp://` target the Streams tab runs `ffmpeg -f alsa -ac 1 -ar 16000 -i hw:1,0 -c:a pcm_s16le -f s16le udp://asr-base.local:5001` on that host: raw PCM, one packet per ALSA period, which the base re-frames to its `chunk_size`. Header-less packets carry no clock, so the base timestamps them by counting samples from the arrival of the first packet. With `record: true` the same process also writes `<name>_<start time>.wav` next to the stream, in the Collection layout, ready for [post-time processing](#post-time-processing). See the [Streaming guide](../rtmp_streaming.md) for the server, the other commands and the recording layout.

## Run from the TUI

1. **ASR Server**: `Launcher → Pipelines → ASR → ASR Server`, Host set to the GPU server. Fill in `pipelines/asr-server/config.yml` on the Config tab (speaker embedding backend and model, transcription backend, which services use CUDA), tick the services to launch, and press **Start**. The card runs `docker compose -f docker/docker-compose.asr.yml up -d --build` for those services; the first start builds the images and downloads the models. Details in the [Docker guide](../docker.md).
2. **System services** running and reachable, and `Server.asr` pointing at the server or the gateway. Verify with **Refresh** on the System Services cards.
3. **ASR Base**: Host set to the base station. On the Launch tab choose the number of bases and synchronizers, the **Session** (or `Create MongoDB Session` from an experiment group), the **Mode** and the toggles (store audio, VAD, noise reduction, transcribe, speech separation, dominant speaker, half-scaled recognition). Everything a base or the synchronizer used to ask in its window is chosen on the card too: one dropdown per base picks its `Bases` entry (there are as many as the number of bases), and the synchronizer takes the base type (a block of `Base`) and **Sync Waits For**, the number of bases it waits for (`--num_bases`): it starts on the number of `Bases` entries, and is set by hand when bases of the session run on other hosts or fewer run than the list has. **Start** opens one terminal window per base and synchronizer, and nothing is asked in them: each base starts recognizing at once with every registered speaker profile selected, the synchronizer starts synchronizing, and all of them wait for START.
4. **Session Control**: once every window reports that it is waiting, send **START** for the session; send **STOP** at the end. On STOP every base and the synchronizer finish their run and exit, also when STOP comes before START.

When a base cannot start on its own, its window says why and what to do, and shows the base's menu so it can be fixed there (when that menu would clear the screen, the window waits for Enter first). The usual case is `asr_scope: participant` in `live` or `analyze` mode with no speaker profile registered yet: choose **Edit Speaker Profiles** (register from the stream or from files, select the speakers), then **Start**. A `Bases` entry that is not there, or that names no stream or file (or one that is not there), is picked from a menu in the same way. A run that ends with an error rather than STOP also comes back to the menu; a run started again from there exits on STOP like the first.

Modes: `live` recognizes and transcribes the stream in real time (the TUI default); `capture` only records the audio to `records/` for later; `analyze` processes the `.wav` files found in `records/`. With `asr_scope: participant` (the default) speaker verification is on and each base needs registered speaker profiles before `live` or `analyze` can run; `asr_scope: group` attributes the transcript to the group without verification.

### What a session records

When a base joins a session it notes in the session's MongoDB document which `Bases` entry it is and the stream it takes, and notes when it leaves, first thing on its way out, before it finishes its last audio chunks. A `stream` base names its `Streams` entry (also when it was picked from the menu); a `udp` or `tcp` base is matched to the `Streams` entry whose `target` has its port; a `pyaudio`, `lsl` or `file` base takes no stream. **Sessions → Export Streams** reads this, so it fetches the session's own streams, the Stream Server's copy and the capture host's, without being told which. Sessions from before this, or ones no base joined, have no such note: they are exported as before, and the console says so.

## Manual CLI

```bash
conda activate asr-base
# one per microphone; -b picks the Bases entry, -sid the session
mmla asr-base -p pipelines/asr-base -c pipelines/asr-base/config.yml -m live -sid <session-id> -b <base-id>
# one per session; -bt the Base block the bases use, -nb how many bases to wait for
mmla asr-sync -p pipelines/asr-base -c pipelines/asr-base/config.yml -sid <session-id> -bt <base-type> -nb 2
```

With `-sid` a process runs as it does from the TUI: it asks nothing, starts at once and exits when the run ends with STOP. A flag left out then takes its default: `-b` the only `Bases` entry, `-bt` the only block of `Base`, `-nb` the number of `Bases` entries; when there is no single one to take, the window says so and asks. Without `-sid` each process shows its menu and asks for the session and for whatever its flags leave out, as before.

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
