# Automatic speech recognition (ASR)

ASR listens to a group through microphones and turns its talk into data: who speaks when, what they say, and how many voices take turns. Use it when a study needs speech measures next to video ([VFA](../vfa/index.md)) and positions ([IPS](../ips/index.md)).

## What ASR produces

| Output | What it gives | Event |
|---|---|---|
| Speaker recognition | per time bucket (as long as a segment, 3 s in the template), who speaks, with the similarity and the seconds of speech | `asr_recognition` |
| Transcripts | per chunk of one speaker, the text, the words with their times when the backend gives them, and the speaker | `asr_transcription` |
| [Speaker turns](speakers-and-diarization.md#diarize) | per chunk, anonymous turns (`SPEAKER_00`, `SPEAKER_01` ...) linked into the voices of the session, without profiles | `asr_transcription`, field `diarization` |
| [Worn-microphone levels](speakers-and-diarization.md#personal-microphones-and-energy-attribution) | per personal microphone, its level every 100 ms and its noise floor, and which wearer each bucket goes to | `asr_recognition` and `asr_transcription`, fields `energies` and `levels` |

The fields of each event are in [Databases](../../database.md#influxdb).

Each base attributes its speech to the speakers registered on its host, to the one person who wears its microphone, or to the group ([Attribution per base](speakers-and-diarization.md#attribution-per-base)).

## How it works

![The path of one audio segment through an ASR base: gain, noise reduction and VAD; a silent segment goes straight to the output; speech may be split by speech separation; the segment is embedded and either registered in the speaker library or recognized against it; recognized segments are aggregated into chunks and transcribed](../../img/real_time_asr_analyzer.png)

1. **Audio capture**: each ASR base reads one microphone: a device on its machine, a badge or Raspberry Pi that pushes over UDP or TCP, a stream from the Stream Server, an LSL stream or a recorded file ([Input sources](configuration.md#input-sources)).
2. **Speech segments**: the base cuts the audio into segments of `recognize_duration` seconds. It applies gain, noise reduction (Speech Enhancer) and voice activity detection (Voice Activity Detector), and decides whether the segment holds speech ([Speech gate](speakers-and-diarization.md#speech-gate)).
3. **Speaker recognition**: a base that verifies speakers has the Audio Inferer turn each speech segment into a speaker embedding and compares it with the registered profiles. With speech separation on, the Speech Separator first splits overlapping voices.
4. **Transcription**: consecutive segments of one speaker form a chunk, which the Speech Transcriber transcribes and, when asked, diarizes. The base writes each transcript to InfluxDB as `asr_transcription`.
5. **Synchronization**: each base publishes its recognitions over MQTT. The ASR synchronizer merges those of all bases into time buckets and writes them as `asr_recognition`.

The dashboard's [Live](../../dashboard/index.md#live) page follows a session as it is written. Joining these events window by window with the other pipelines is analysis rather than capture: see [Window features](../../analytics/window_features.md).

## Components

| Component | Runs on | Started with | Environment |
|---|---|---|---|
| ASR Base, one per microphone | base station | `mmla asr-base` | conda env `asr-base` |
| ASR Synchronizer, one per session | base station | `mmla asr-sync` | conda env `asr-base` |
| ASR Server, six services | GPU server | `docker compose` | images built from `docker/` |

The ASR Server's services, one container each:

| Service | What it does |
|---|---|
| Audio Inferer | speaker embeddings, by WeSpeaker or NeMo |
| Audio Resampler | resamples audio |
| Speech Enhancer | noise reduction |
| Speech Separator | splits overlapping voices, for bases with speech separation on |
| Speech Transcriber | transcription by Whisper, WhisperX, Azure or DashScope, and diarization |
| Voice Activity Detector | keeps the speech of a segment |

The management console starts all of them from cards under `Launcher → Pipelines → ASR`; see [Run ASR](run.md).

## What you need

- **Microphones**: a USB or built-in microphone on a base station, a Nicla Vision or Portenta H7 badge over Wi-Fi, a microphone on a Raspberry Pi that the console streams, an LSL stream, or recorded files ([Input sources](configuration.md#input-sources)).
- **A GPU server** for the ASR Server, with the NVIDIA driver and the NVIDIA container toolkit ([Docker](../../docker.md#host-requirements)).
- **A Hugging Face token**, only for diarization: the pyannote pipeline it runs is gated ([Set up diarization](speakers-and-diarization.md#set-up-diarization)).
- **The system services**: MQTT, Redis, MongoDB and InfluxDB, and the Stream Server for streamed microphones ([Quickstart](../../quickstart.md#one-time-setup)).
- **The `asr-base` environment** on every base station. Create it from the console's **Environment** tab, or by hand:

    ```bash
    conda create -n asr-base python=3.10 -y
    conda activate asr-base
    pip install -e '.[asr-base]'
    ```

    For an `lsl` source, add `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.

## Pages in this guide

- [Run ASR](run.md): set up once, run every session, run from the command line, replay recordings.
- [Speakers and diarization](speakers-and-diarization.md): attribution, speaker profiles, language, diarization, voices across chunks, personal microphones.
- [Configuration](configuration.md): every key of the base and server configs, the input sources, the streams, and what a session records.
