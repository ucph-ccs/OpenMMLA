# Configuration

ASR reads two config files: `pipelines/asr-base/config.yml` for the bases and the synchronizer, and `pipelines/asr-server/config.yml` for the ASR Server. You edit both on the **Config** tab of their card in the management console; this page lists their keys.

## Base config

`pipelines/asr-base/config.yml` lives on every base station. **Save** on the ASR Base card's **Config** tab writes it on the card's host, creating it from `config_template.yml` the first time; by hand, copy the template.

| Section | What it holds |
|---|---|
| [`Base`](#base-blocks) | one block per kind of microphone (a table speakerphone, a badge, a laptop microphone), named as you like: how its audio is processed and the format the base works in |
| [`Bases`](#bases) | one entry per microphone, that is per base process: its id, its base type, its source and the fields that source needs |
| [`Synchronizer`](#synchronizer) | the buckets, and the vote over personal microphones |
| [`Streams`](#streams) | managed and external audio streams |
| [`Server.asr`](#service-endpoints) | the six service endpoints |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings ([System services](../../system_services.md)) |

### Base blocks

Each block under `Base` is a base type, which `Bases` entries name. **+ Add Base** on the **Config** tab adds one. A key marked required has no default and must be set.

| Key | Default | What it does |
|---|---|---|
| `asr_scope` | `individual` | `individual`, `wearer` or `group`: whom the speech goes to, and what **Participant** opens on ([Attribution per base](speakers-and-diarization.md#attribution-per-base)); `participant` is read as `individual`, and `wear` and `worn` as `wearer`; the **Config** tab shows each as the scope it means |
| `speaker_verification` | `auto` | `auto` verifies speakers for `individual` only; `true` or `false` overrides it; a base with a wearer never verifies |
| `max_chunk_duration` | 30 for a `wearer` or `group` base and for any base with a wearer; no cap otherwise | the seconds a chunk of one speaker may grow before it is transcribed on its own; `0` for no cap ([Chunk length](speakers-and-diarization.md#chunk-length)) |
| `register_duration` | required | seconds recorded to register a speaker |
| `recognize_duration` | required | seconds per segment; also the bucket length when the `Synchronizer` section sets none |
| `recognize_sp_duration` | required | seconds per segment with speech separation |
| `recognize_threshold` | required | the similarity above which a segment is a registered speaker; below it, the segment is `unknown` |
| `recognize_sp_threshold` | required | the same with speech separation |
| `keep_threshold` | required | half-scaled recognition: the similarity an `unknown` half segment at a change of speaker needs to take one of the two speakers |
| `keep_sp_threshold` | required | the same with speech separation |
| `update_threshold` | `0.6` | the similarity above which a recognized segment updates the speaker's adaptive embedding |
| `gain` | required | dB of gain applied to each segment before noise reduction and VAD |
| `rms_threshold`, `rms_peak_threshold` | required with `speech_gate: absolute`, else 0 | the rms and peak amplitude a segment needs after gain, noise reduction and VAD to count as speech |
| `speech_gate` | `absolute` | `absolute` or `relative` ([Speech gate](speakers-and-diarization.md#speech-gate)) |
| `speech_gate_snr_db` | `6` | the relative gate's dB over the base's noise floor |
| `score_amplified` | `false` | scale a recognized speaker's similarity, up to 1, by how far the segment stands over the gate: `log(rms) / log(rms_threshold)` with the absolute gate, its snr over `speech_gate_snr_db` with the relative one |
| `stream_kwargs` | | the audio as the base works with it, whatever the source (next table) |

`stream_kwargs`:

| Key | Default | What it does |
|---|---|---|
| `channels` | `1` | channels of the audio: what a badge or FFmpeg stream sends over UDP or TCP (it must match), what a Stream Server stream is decoded to and a replayed file converted to; a `pyaudio` device's count is read from the device |
| `rate` | `16000` | the sample rate in Hz |
| `format` | `int16` | `int16`, `int32` or `float32` |
| `chunk_size` | `512` | frames per chunk the base reads |
| `buffer_duration` | `5.0` | seconds of the stream's ring buffer |
| `resample_method` | `audio_librosa` | `audio_linear`, `audio_polyphase`, `audio_fft`, `audio_lanczos` or `audio_librosa` |
| `timestamp_offset` | `0` | seconds added to every chunk's stamp: the stream's measured delay as a negative number, such as `-0.2` ([Timestamps](../../streaming/index.md#timestamps)) |
| `connect_wait` | `30` | seconds a `stream` or `lsl` source that is not up yet is waited for when the run starts; a microphone, a socket or a file is tried at once ([How the bases pull a stream](../../streaming/index.md#bases-pulling-a-stream)) |
| `reconnect_wait` | `3600` | seconds a `stream` or `lsl` source that dropped is opened again before the run ends |

A block for a table speakerphone whose speech is the group's:

```yaml
Base:
  table-mic:
    asr_scope: group
    register_duration: 10
    recognize_duration: 3
    recognize_sp_duration: 4
    recognize_threshold: 0.3
    recognize_sp_threshold: 0.3
    keep_threshold: 0.1
    keep_sp_threshold: 0.1
    rms_threshold: 1000
    rms_peak_threshold: 5000
    gain: 0
    stream_kwargs:
      rate: 16000
      format: int16
      chunk_size: 512
```

### Bases

One entry per microphone (**+ Add Entry**). The **Config** tab shows only the fields the entry's `source` needs, and the card's **Base** dropdowns offer the entries.

| Key | What it is |
|---|---|
| `id` | the base id; the base is named `<base_type>_<id>` |
| `base_type` | a block of `Base` |
| `source` | `pyaudio`, `udp`, `tcp`, `stream`, `lsl` or `file` ([Input sources](#input-sources)) |
| `source_index` | which device, stream or file, by source; not used by `udp` and `tcp` |
| `channel_select` | `pyaudio`: which channel of a multi-channel device the base keeps, all of them when unset; `channel` is accepted as another name |
| `port` | `udp`, `tcp`: the port the base listens on |
| `host` | `udp`, `tcp`: the address it listens on; `0.0.0.0`, every interface, unless set |
| `packet_format` | `udp`, `tcp`: `auto` (the default: the first packet tells), `timestamped` (a badge) or `raw` (header-less PCM, such as FFmpeg's) |
| `participant` | a wearer's tag, for a base started without `--participant`; the Config tab does not show or keep it ([Attribution per base](speakers-and-diarization.md#attribution-per-base)) |

An entry whose `id` is still a `<...>` placeholder is ignored.

### Synchronizer

| Key | Default | What it does |
|---|---|---|
| `result_expiry_time` | required; `10` in the template | seconds after which a bucket some base has not reported to is written with what it has, counted against the newest result |
| `bucket_duration` | `3` in the template; empty: the segment length (below) | seconds per bucket; it should be as long as the segments the bases send |
| `match_tolerance` | `1.5` in the template; empty: the segment length | seconds within which a result joins an open bucket; about half a bucket |
| `energy_margin_db` | `6` | personal microphones: how far a channel's speech must stand over its own noise floor, and over the others', to be its wearer's ([Noise floor and the vote](speakers-and-diarization.md#noise-floor-and-the-vote)) |
| `energy_tie_db` | `3` | personal microphones: channels within this many dB of the loudest count as speaking too |

Left empty, `bucket_duration` and `match_tolerance` take the segment length (`recognize_duration`, or `recognize_sp_duration` with speech separation) that the base types of the `Bases` entries share; when they differ, the most common one, which the synchronizer's log and the card's Start say. `mmla asr-sync -bt <block>` takes that block's length instead. Set, they hold for every base. A blank or placeholder `energy_*` key keeps its default.

Each bucket lists every speaker the bases recognized in it. A bucket in which none was recognized lists the most confident result it has, such as `silent`, and with **Dominant Speaker** on every bucket lists only its most confident result.

### Service endpoints

`Server.asr` names the six services, each through the Gateway or straight to the GPU server:

| Key | Through the Gateway | Straight to the server |
|---|---|---|
| `audio_inferer` | `infer` | `http://<gpu-server>:5001/infer` |
| `audio_resampler` | `resample` | `http://<gpu-server>:5002/resample` |
| `speech_enhancer` | `enhance` | `http://<gpu-server>:5003/enhance` |
| `speech_separator` | `separate` | `http://<gpu-server>:5004/separate` |
| `speech_transcriber` | `transcribe` | `http://<gpu-server>:5005/transcribe` |
| `voice_activity_detector` | `vad` | `http://<gpu-server>:5006/vad` |

A bare name is composed with the `Gateway` section (`http://<gateway>:8080/transcribe`), and the **Config** tab shows it resolved, as `through the Gateway: http://<gateway>:8080/transcribe`. A full URL is used as it is.

## Input sources

| `source` | What it reads | Setup |
|---|---|---|
| `pyaudio` | a USB or built-in microphone on the base station | `source_index` is the PyAudio device index (required): the **Config** tab lists the input devices PyAudio finds on the card's host, asked in its `asr-base` environment; `channel_select` keeps one channel of a multi-channel device |
| `udp`, `tcp` | a Nicla Vision or Portenta H7 badge over Wi-Fi, or a managed FFmpeg stream from a Raspberry Pi ([Streams](#streams)) | the base listens on the entry's `port`, on the address `host` names; the sender's format must match `stream_kwargs` (16 kHz, mono, 16-bit PCM by default) |
| `stream` | audio pulled from the Stream Server; `rtmp` is accepted as another name | `source_index` names a `Streams` entry whose `read_target` (else `target`) is an `rtmp://`, `rtsp://` or `srt://` URL; a number is read as its position among them. Left empty while there are several, the base asks for it in its window |
| `lsl` | a Lab Streaming Layer stream | `source_index` is the stream name; needs `pylsl` ([What you need](index.md#what-you-need)) |
| `file` | a recorded file, replayed | `source_index` is the file's full path: **Browse…** on the **Config** tab writes it, and the dropdown lists the other files of its folder when the card's host is this machine; on another host, type the path. The start time comes from the file name (`<prefix>_<timestamp>.wav`) ([Replay recordings](run.md#post-time-processing)) |

??? info "Details: badges"
    - Flash the firmware under `pipelines/wearables/nicla-vision/asr/` or `pipelines/wearables/portenta-h7/asr/`, with the Wi-Fi credentials and the base's host and `port` in `arduino_secrets.h`.
    - The `audio_streaming_udp_ms` firmware prefixes every packet with an 18-byte header carrying the badge's clock. The other firmware and FFmpeg send header-less PCM.
    - With `packet_format: auto` the base tells the two apart on the first packet; `timestamped` or `raw` forces one.

??? info "Details: configs with source fields in `Base` blocks"
    A config that keeps `host`, `packet_format` or `file_dir` in its `Base` blocks still works: the bases read them there. The **Config** tab shows them in the `Bases` entries that use them, where the next **Save** writes them; one no entry uses is dropped, and the log names what moved and what went.

## Streams

Every stream a base pulls from, or that pushes to a base, is declared once under `Streams`. An entry with an `ssh_profile` is managed: the console starts and stops its FFmpeg on that host from the **Streams** tab. An entry without one is external, and taken as already running.

```yaml
Streams:
  # external stream (already running; the base pulls read_target, else target)
  mic-external:
    target: rtmp://uber-server.local:1935/asr/mic1
    read_target: rtsp://uber-server.local:8554/asr/mic1

  # managed stream straight to a udp base: the console runs ffmpeg on a Raspberry Pi over SSH
  pi-mic-1:
    ssh_profile: pi-01              # must match an SSH profile name of the console
    device: hw:1,0                  # ALSA audio device on the remote machine
    target: udp://base-01.local:5001
    format: s16le                   # must match stream_kwargs.format (int16)
    rate: 16000                     # must match stream_kwargs.rate
    channels: 1                     # must match stream_kwargs.channels
    record: true                    # also keep a wav on the Pi for later replay

  # managed stream through the Stream Server (AAC), pulled as RTSP by a stream source
  pi-mic-2:
    ssh_profile: pi-02
    device: hw:1,0
    kind: audio                     # audio or video; left out, a udp/tcp target or an audio device is audio
    target: rtmp://uber-server.local:1935/asr/mic2
    read_target: rtsp://uber-server.local:8554/asr/mic2
```

For a `udp://` or `tcp://` target the **Streams** tab runs `ffmpeg -f alsa -ac 1 -ar 16000 -i hw:1,0 -c:a pcm_s16le -f s16le udp://base-01.local:5001` on the capture host: raw PCM, one packet per ALSA period, which the base re-frames to its `chunk_size`. Header-less packets carry no clock, so the base stamps them by counting samples from the arrival of the first packet. With `record: true` the same process also writes `<name>_<start time>.wav` on the capture host, in the Collection layout, ready to [replay](run.md#post-time-processing).

The [Streaming guide](../../streaming/index.md) has every key of an entry, the other FFmpeg commands and the recording layout.

## Server config

`pipelines/asr-server/config.yml` lives on the GPU server, one section per service. **Save** on the ASR Server card's **Config** tab writes it on the card's host, and the services read it when they start ([Once per deployment](run.md#once-per-deployment)). The console stores tokens and API keys (`hf_token`, `subscription_key`, `api_key`) encrypted on save.

### Keys for every service

| Key | Default | What it does |
|---|---|---|
| `cuda` | `true` | run on the GPU; the Audio Resampler has no such key |
| `port` | 5001 to 5006 | the port the card checks the service at and, without Docker, starts it on; a section without one is not among **Services to launch**; a container listens on its own port whatever it says |
| `workers` | `1` | gunicorn workers without Docker; a container runs one |

### Speaker embedding keys

The `AudioInferer` section:

| Key | Default | What it does |
|---|---|---|
| `backend` | `nemo` | `wespeaker` or `nemo`; the card picks the inferer's container from it ([Docker](../../docker.md#switching-the-inferer-backend-wespeaker-or-nemo)) |
| `wespeaker.model` | `w2vbert2_mfa` | `chinese`, `english`, `campplus`, `eres2net`, `vblinkp`, `vblinkf` or `w2vbert2_mfa` |
| `nemo.model` | required for `nemo` | a NeMo speaker embedding model, such as `titanet_large` or `ecapa_tdnn` |
| `nemo.onnx` | `false` | run it in the ONNX runtime: faster, less precise |
| `nemo.model_onnx` | not set | the path of a pre-exported ONNX model |

### Transcription keys

The `SpeechTranscriber` section. `backend` picks the backend, and each backend reads its own block.

| Key | Default | What it does |
|---|---|---|
| `backend` | `local` | `local` (Whisper and WhisperX), `azure` or `dashscope`; `paraformer` is read as `dashscope` |
| `local.model` | `base.en` | a Whisper model (`base.en`, `small.en`, `medium.en` ...) or a WhisperX one (`whisperx/large-v3`, `whisperx/medium`), which gives word timestamps and can diarize |
| `local.language` | `en` | the language when a request names none ([Language](speakers-and-diarization.md#language)); `null` detects it |
| `local.word_level` | `false` | word timestamps; needs a `whisperx/` model, or the service does not start |
| `local.compression_ratio_threshold` | `2.4` | `whisperx/` models: a segment whose text, its loops cut, still compresses more than this many times is dropped with its words; `0` keeps every segment |
| `local.diarize` | `false` | diarize every file for every base; needs a `whisperx/` model, or the service does not start ([Diarization](speakers-and-diarization.md#diarize)) |
| `local.diarize_model` | `pyannote/speaker-diarization-community-1` | the pyannote pipeline: `pyannote/speaker-diarization-community-1`, the default of the image's WhisperX 3.8, which needs pyannote.audio 4, or `pyannote/speaker-diarization-3.1`, the default of the non-Docker extras' WhisperX 3.3 |
| `local.hf_token` | `HF_TOKEN` in `docker/.env` | the Hugging Face token that may fetch the pipeline ([Set up diarization](speakers-and-diarization.md#set-up-diarization)) |
| `local.min_speakers`, `local.max_speakers` | no bound | the fewest and the most speakers the diarizer may find, when the number is known |
| `azure.subscription_key`, `azure.region` | required for `azure` | the Azure Speech resource |
| `azure.language` | `en-US` | the locale when a request names none |
| `azure.profanity_option` | `masked` | `raw`, `masked` or `removed` |
| `azure.word_level` | `false` | word timestamps |
| `dashscope.api_key` | `DASHSCOPE_API_KEY` | the DashScope key |
| `dashscope.model` | `paraformer-realtime-v2` | such as `paraformer-realtime-v2` or `qwen3-asr-flash` |
| `dashscope.region` | `intl` | `intl` (Singapore) or `cn` (Beijing): the endpoint of the qwen3-asr models |
| `dashscope.word_level` | `false` | word timestamps; paraformer only |
| `dashscope.language` | `[zh, en, ja]` | a list of language hints for paraformer, one language for qwen3-asr |
| `dashscope.enable_itn` | `false` | inverse text normalization, Chinese and English; qwen3-asr only |

A stretch of noise can make WhisperX write a word or a phrase over and over (`Mmm. Mmm. Mmm.`). With a `whisperx/` model the speech transcriber cuts such a loop itself, before alignment and diarization: a phrase of up to eight words said more than twice in a row is cut to two turns, and then `compression_ratio_threshold` drops what still repeats. The service's log says what was cut and what was dropped.

### Keys of the other services

| Key | Default | What it does |
|---|---|---|
| `SpeechSeparator.model` | required | a ModelScope speech separation model, such as `damo/speech_mossformer2_separation_temporal_8k` |
| `SpeechSeparator.model_local` | required | its local path, such as `.cache/modelscope/hub/damo/speech_mossformer2_separation_temporal_8k` |
| `VoiceActivityDetector.onnx` | `false` (`true` in the template) | run Silero VAD in the ONNX runtime |

### Config tab dropdowns

| Key | Choices |
|---|---|
| `asr_scope` | `individual`, `wearer`, `group` |
| `speech_gate` | `absolute`, `relative` |
| `SpeechTranscriber.local.diarize_model` | `pyannote/speaker-diarization-community-1`, `pyannote/speaker-diarization-3.1` |

A value that is not in the list is shown and kept. The blank entry keeps the default.

## What a session records

Each session's MongoDB document notes what produced its ASR data ([Databases](../../database.md#mongodb)), so its recognitions and transcripts can be traced to the models and settings that made them.

- **Each base** notes which `Bases` entry it is and the stream it takes, when it joins the session, and notes when it leaves, first thing on its way out. A `stream` base names its `Streams` entry, also when it was picked from the menu; a `udp` or `tcp` base is matched to the `Streams` entry whose `target` has its port; a `pyaudio`, `lsl` or `file` base takes no stream. **Sessions → Export** reads this, so it fetches the session's own streams, the Stream Server's copy and the capture host's, without being told which.
- **Each base** also notes its flags, its parameters and its config with the secrets masked, and for a base that verifies speakers the names of its speakers and a snapshot of their profiles in the session's artifacts.
- **The synchronizer** notes its flags, the number of bases, the expiry, `bucket_duration`, `match_tolerance`, `energy_margin_db` and `energy_tie_db`.
- **The services** each answer `GET /<endpoint>/info` (`/transcribe/info`, also through the Gateway) with what they run: the backend, the model, the language, whether they are on CUDA. A base that joins a session asks and notes the answers. A service without `/info` is noted with an error instead.

Among a base's parameters, besides its `Base` block's settings and its `stream_kwargs`:

| Parameter | What it says |
|---|---|
| `participant`, `wearer_source` | the wearer, and where it came from: `launch` (the card's **Participant**), `config` (the `Bases` entry) or `session` (the Collection Start's pick) |
| `attribution` | `energy` for a worn microphone |
| `diarize` | whether the base diarized; `arguments.diarize` is what `-dia` said, null when it was not given |
| `quiet_cut_seconds` | 10, or none for a base that cuts its chunks at the cap ([Chunk length](speakers-and-diarization.md#chunk-length)) |
| `voice_link_threshold`, `voice_min_seconds` | 0.35 and 1, for a base that diarizes ([Voices across chunks](speakers-and-diarization.md#voices-across-chunks)) |
| `voice_embedding` | `speech`, the kind of speaker embedding the threshold is set for, for a base that diarizes; each linked record names the kind its own speakers were linked by |
| `level_hop_seconds` | 0.1, the step of a worn microphone's levels |
| `service_urls` | the service endpoints the base used |

??? info "Details: when a note is missing"
    A session no base joined has nothing to export, and the console says so. A base that cannot write its note (MongoDB down, or a session the console did not create) warns in its log and runs on.
