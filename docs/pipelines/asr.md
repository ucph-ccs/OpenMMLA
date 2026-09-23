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
| `Base.<device>` | one block per **kind** of microphone (a table speakerphone, a badge, a laptop mic), named as you like: how its audio is processed (`asr_scope`, recognition thresholds and durations, gain, VAD thresholds, `max_chunk_duration`) and the format the base works in, whatever the source (`stream_kwargs`: `channels`, `rate`, `format`, chunk size). Add one with `+ Add Base`. |
| `Bases` | one entry per **microphone**, that is per base process you start: its `id`, its `base_type` (a block of `Base`), its `source`, and the fields that source needs: `source_index` (a device index, a stream name, the full path of a file), `channel_select` (pyaudio), or `port`, `host` and `packet_format` (udp/tcp); the Config tab shows only those, see [Input sources](#input-sources). Who wears a personal microphone is not part of the entry, as it changes from session to session: it is picked per base on the Launch tab (**Participant**), see [Personal microphones](#personal-microphones-and-energy-attribution). The card's Base dropdowns offer these entries. |
| `Synchronizer` | `bucket_duration`, `match_tolerance`, `result_expiry_time`, `energy_margin_db` (6) and `energy_tie_db` (3) for personal microphones. A bucket should be as long as the segments the bases send (`recognize_duration` of their `Base` blocks, `recognize_sp_duration` with speech separation); left empty, `bucket_duration` and `match_tolerance` take the length the base types of the `Bases` entries share (the most common one when they differ, which the synchronizer's log and the card's Start say), and set, they hold for every base. |
| `Streams` | managed and external streams, see below |
| `Server.asr` | the six service endpoints, either through the gateway (`http://<gateway>:8080/transcribe`) or direct (`http://<server>:5005/transcribe`) |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings, see [System Services](../system_services.md) |

A config from before keeps `host`, `packet_format` and `file_dir` in its `Base` blocks: the bases still read them there, and the Config tab shows them in the entries that use them, where the next **Save** writes them (one no entry uses is dropped; the log names what moved and what went).

### Input sources

`Bases[].source` selects where a base reads audio from:

| Source | Description | Setup |
|---|---|---|
| `pyaudio` | USB or built-in microphone on the base station (Jabra Speak2 75, laptop mic, ...) | `source_index` is the PyAudio device index (required): the Config tab lists the input devices PyAudio finds on the card's host, asked in its `asr-base` env, to pick from; `channel_select` picks which channel of a multi-channel device the base keeps, all of them when unset (`channel` is its old name, still read). `stream_kwargs.channels` is how many channels the audio has, 1 unless set; for pyaudio it is read from the device itself. |
| `udp` / `tcp` | Nicla Vision or Portenta H7 badge streaming over Wi-Fi, or a managed FFmpeg stream from a Raspberry Pi (see [Streams](#streams)) | badges: flash the firmware under `pipelines/wearables/nicla-vision/asr/` or `pipelines/wearables/portenta-h7/asr/` with the Wi-Fi credentials and the base's host and `port` in `arduino_secrets.h`. The base listens on the entry's `port`, on the address `host` names (`0.0.0.0`, every interface, unless set). The sender's format must match `stream_kwargs` (16 kHz, mono, 16-bit PCM by default). The `audio_streaming_udp_ms` firmware prefixes every packet with an 18-byte header carrying the badge's clock; the other firmware and FFmpeg send header-less PCM. The base tells the two apart on the first packet (the entry's `packet_format: auto`, the default); set it to `timestamped` or `raw` to force one. |
| `stream` | audio pulled from the MediaMTX server (`rtmp` is the old name) | a `Streams` entry whose `read_target` (else `target`) is an `rtmp://`, `rtsp://` or `srt://` URL; `source_index` names that entry (a number is read as its position among them, as older configs have it). Left empty while there are several, the base asks for it in its window |
| `lsl` | Lab Streaming Layer | `source_index` is the LSL stream name; needs `pylsl` |
| `file` | replay of a recorded file | `source_index` is the file to replay, by its full path: **Browse…** on the Config tab writes it, and the dropdown lists the other files of its folder (when the card's host is this machine; on another one the path is typed). The start time is read from the file name (`<prefix>_<timestamp>.wav`), so the sync time needs no configuration |

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
3. **ASR Base**: Host set to the base station. On the Launch tab choose the number of bases and synchronizers, the **Session** (or `Create MongoDB Session` from an experiment group), the **Mode**, the toggles (store audio, VAD, noise reduction, transcribe, diarize, speech separation, dominant speaker, half-scaled recognition), the **Language** and **Diarize** (both below), and under each Base its **Participant**, whom that base's speech is attributed to (see [Personal microphones](#personal-microphones-and-energy-attribution)). Everything a base or the synchronizer used to ask in its window is chosen on the card too: one dropdown per base picks its `Bases` entry (there are as many as the number of bases), and the synchronizer takes **Sync Waits For**, the number of bases it waits for (`--num_bases`): it goes along with the number of bases, and is set by hand when bases of the session run on other hosts. **Start** opens one terminal window per base and synchronizer, and nothing is asked in them: each base starts recognizing at once with the speakers of its **Speakers** line (below), the synchronizer starts synchronizing, and all of them wait for START.
4. **Session Control**: once every window reports that it is waiting, send **START** for the session; send **STOP** at the end. On STOP every base and the synchronizer finish their run and exit, also when STOP comes before START.

### Speakers

A base whose **Participant** is Speakers (what a base type with `asr_scope: individual` opens on) recognizes who speaks among the speaker profiles registered on its host: one folder per speaker under `artifacts/runtime/pipelines/asr-base/<host>/profiles/`, shared by every base there. Which of them each base takes is its own: under every Base whose Participant is Speakers, a **Speakers** line says who that base takes at the next Start, and why (or that capture mode takes nobody):

- Nobody ticked any for it: the participants of the session's experiment group (Study → Experiments) that have a profile on the host. A profile stands for a participant when it is named as the participant or as their tag id, in any case. The group is the card's Experiment Group for `Create MongoDB Session`, else the group the picked session belongs to. When no participant has a profile, every profile is taken, as the base did before.
- **Manage** opens the profiles of the card's host for that base. A tick is a speaker the base takes; ticking one yourself makes the pick yours, kept for that `Bases` entry on that host (a badge keeps its speakers whichever row it is on), **Use Group** gives it back to the group, **Use All** and **Use None** tick all or none. **Delete** removes the highlighted profile from the host after a second press; it is gone for every base there.
- **Register a speaker** on the same screen: type the name (the group's participants are suggested). **Record** records it from the source of the base picked under **Record from** on the card's host, for **Seconds** (empty: the base type's `register_duration`), while the speaker reads the sentences shown; **Register Files** registers it from reference audio on this machine, added with **Add File…** and copied to the host first when that is another one. Both go through VAD and noise reduction when the card has them on, and need the ASR Server. When nothing comes out, the message says why: which of these services does not answer and how (the Gateway has no route to it, its route has no server that answers, nothing answers at the address), or, when they all answer, that the recording held no speech. A name that has a profile already adds to it; the new profile is one more for every base of the host to tick.

### Language

**Language** on the Launch tab is the language the bases of this card have their speech transcribed in (`-lang`). It is sent with every transcription request and holds for that request alone, so switching language is a matter of starting the bases again: the speech transcriber keeps running, whatever its own `SpeechTranscriber` config says. `the server's own language` (the first option) sends none, and the service transcribes in its configured language.

The dropdown lists the common languages; `-lang` itself takes any code the backend knows (`yue`), or a locale (`en-GB`). Each backend is asked in its own form: a code for WhisperX, Whisper and DashScope, a locale for Azure (`da` becomes `da-DK`). The answer says which language was used, and a base says so in its window when the service did not take the one it asked for — the sign of a speech transcriber running code from before this existed, which **ASR Server → Start** builds anew.

With `word_level` on, WhisperX aligns the words with a model of that language: the first use of a language fetches it on the server (once), and a language it has none for gives text without word timestamps.

A stretch of noise can make WhisperX write a word or a phrase over and over (`Vi gør det igen. Kom.` twenty times, `Mmm. Mmm. Mmm.`), most often at the end of a segment that began as real speech, and its batched pipeline does not notice. The speech transcriber cuts such a loop itself, before alignment and diarization: a phrase of up to eight words said more than twice in a row is cut to two turns, and a segment whose text still compresses more than `SpeechTranscriber.local.compression_ratio_threshold` times (2.4, Whisper's own criterion, unless set; 0 keeps every segment) — a run of one character is what remains — is dropped with its words. The service's log says what was cut and what was dropped.

### Diarize

**Diarize** on the Launch tab (`-dia`) sends every chunk for its anonymous speaker turns as well: the speech transcriber runs pyannote's diarization on the chunk (through WhisperX, so only a local `whisperx/` model can) and answers with `diarization`, the turns `[{start, end, speaker}]` in seconds from the start of the chunk, the speakers named `SPEAKER_00`, `SPEAKER_01` ... within that chunk. No profile, no name, no enrolment: the transcript record of the chunk (`asr_transcription`) carries the turns next to its words, which is what a group-level base (`asr_scope: group`, one microphone for the group) has of who-of-how-many spoke when. From them a session's speaker changes, overlaps, active speakers per window and the equality of their shares can be computed without knowing who anyone is; only which person a turn belongs to needs a profile, or another modality.

A chunk is the audio of one speaker between two changes of speaker, and a group-level base never hears one: its chunk would end only at silence, in a classroom minutes later. `Base.<device>.max_chunk_duration` caps it (30 s for `asr_scope: group` and `wearer` unless set, no cap for an individual base, 0 for none): a chunk that reaches the cap is transcribed and diarized on its own and the next segment starts a new one, so the turns of a group microphone are per chunk of at most that length.

The turns are the request's alone, as the language is: `SpeechTranscriber.local.diarize: true` in the server config diarizes every file for every base instead. The pyannote pipeline is gated on huggingface.co: accept its terms with an account, and give that account's token as `SpeechTranscriber.local.hf_token` (stored encrypted, like the other keys), or as `HF_TOKEN` in `docker/.env` on the ASR Server's host (`docker-compose.asr.yml` passes it into the container; an exported variable does not reach a stack the console starts over SSH). `diarize_model` names another pyannote pipeline than WhisperX's default, and `min_speakers` / `max_speakers` bound the count when it is known. A base that asked and got no turns says so once in its window (the backend cannot, the pipeline could not be made, or the service runs code from before a request could ask), and the service's log says which.

Start passes each base its own pick as `-spk Alice,Bob` (nothing to the synchronizer). It does not start a base that would stop at its menu for want of speakers, in `live` or `analyze` mode: no profile registered on the host, none ticked for it, or none of those ticked registered there; the log names the base and says what to do. A base already running keeps the profiles it started with. On another host all of this runs `mmla asr-speakers` there, in its `asr-base` env and its own checkout: until that checkout is pulled, Manage says so.

When a base cannot start on its own, its window says why and what to do, and shows the base's menu so it can be fixed there (when that menu would clear the screen, the window waits for Enter first). With speakers that is a host the card could not ask (or a profile deleted since): choose **Edit Speaker Profiles** (register from the stream or from files, select the speakers), then **Start**. A `Bases` entry that is not there, or that names no stream or file (or one that is not there), is picked from a menu in the same way. A run that ends with an error rather than STOP also comes back to the menu; a run started again from there exits on STOP like the first.

Modes: `live` recognizes and transcribes the stream in real time (the TUI default); `capture` only records the audio to `records/` for later; `analyze` processes the `.wav` files found in `records/`. With `asr_scope: individual` (the default) speaker verification is on and each base needs registered speaker profiles before `live` or `analyze` can run; `asr_scope: wearer` is a microphone one person wears, whose speech is labelled with its wearer without verification (see [Personal microphones](#personal-microphones-and-energy-attribution)); `asr_scope: group` attributes the transcript to the group without verification. The Config tab offers the three as a dropdown; on the Launch tab the scope is what each base's **Participant** opens on, and the Participant decides. A config that still says `participant`, the earlier name of `individual`, keeps working, and the Config tab shows it as `individual`.

### What a session records

When a base joins a session it notes in the session's MongoDB document which `Bases` entry it is and the stream it takes, and notes when it leaves, first thing on its way out, before it finishes its last audio chunks. A `stream` base names its `Streams` entry (also when it was picked from the menu); a `udp` or `tcp` base is matched to the `Streams` entry whose `target` has its port; a `pyaudio`, `lsl` or `file` base takes no stream. **Sessions → Export Streams** reads this, so it fetches the session's own streams, the Stream Server's copy and the capture host's, without being told which. Sessions from before this, or ones no base joined, have no such note: they have nothing to export, and the console says so.

## Personal microphones and energy attribution

**Why.** A classroom group can wear one microphone each (a Vimo or badge channel per student) beside a Jabra room microphone, and nobody enrolls a voice. On synchronized channels worn by one person each, the wearer is the loudest voice on their own channel, and the neighbours reach it only as cross-talk. The level of each channel is what tells them apart.

**Config and Launch.** Give the worn microphones' kind `asr_scope: wearer` in its `Base` block. Who wears which microphone changes from session to session, so it is picked on the Launch tab: under each Base, **Participant** offers the participants of the session's experiment group (`<name> (tag <id>)`), **Group** and **Speakers (speaker verification)**, and the base is started with `--participant <tag>|group|speakers`. A row opens on the pick kept for that experiment group and `Bases` entry (a pick made by hand holds for every session of the group), else on the wearer the session's Collection Start noted for the base's stream (`wearers` in the session's MongoDB document), else on what its base type says: `wearer` takes the group's participants in row order, a base that verifies speakers (`individual`) Speakers, any other Group. Start is refused when one tag is picked on two bases. A base worn by a participant verifies no speaker, labels every speech segment with the tag, and chunks like a group microphone (`max_chunk_duration`, 30 s by default); Group attributes its speech to the group; Speakers verifies speakers among its **Speakers** line, the only case that has one. The pick wins over the config for every run, and the provenance says `wearer_source: launch`. A base started without `--participant` (from a terminal, or by the replay runner) goes by the config: the `participant` of its `Bases` entry (which the replay runner writes into the configs it generates; the Config tab no longer shows or keeps it), else for a stream the session's Collection pick (`wearer_source: session`), else its `asr_scope`; a `wearer` base that ends up with no wearer says so in yellow when it starts and attributes its speech to the group for that run. Two `Synchronizer` keys set the vote: `energy_margin_db` (6), how far above its own noise floor a channel's speech must be to be its wearer's, and `energy_tie_db` (3), how close to the loudest another channel must be to count as speaking too. A blank or placeholder keeps the default.

**Speech gate.** A base normally counts a segment as speech when VAD keeps speech in it and its level after gain, noise reduction and VAD is over `rms_threshold` and `rms_peak_threshold` (`speech_gate: absolute`, the default). A worn microphone can sit 30-40 dB below a room microphone, and a new device has its own level, so any fixed threshold passes all of its segments or none. `speech_gate: relative` in its `Base` block instead counts a segment as speech when VAD keeps speech in it and its raw level, before gain, stands at least `speech_gate_snr_db` (6) over the base's own noise floor: the same `rms_db - floor_db` its recognitions carry. Gain and the rms thresholds then no longer decide speech; they still shape what is transcribed. The floor starts with the run, so the first segment is never speech, and until five segments are held the floor is the quietest of them. A blank or placeholder keeps the defaults, the gate is printed when the base starts, and both keys are noted in the session's provenance. The gate works for any base; it does not tell the wearer from neighbours who are just as loud, which is still the synchronizer's vote.

**What the events carry.**

- Every recognition a base publishes on `<sid>/asr` carries `energy`: `rms_db`, `peak_db` and `floor_db`, in dBFS, of the raw segment before any gain. The floor is the 10th percentile of the base's segment levels over the last 60 s; digital silence is left out, and a longer gap starts the floor again.
- A wearer's recognition also carries `participant`.
- For each bucket the synchronizer takes the personal channels with speech, and their snr: `rms_db - floor_db`. The loudest wins when it is at least `energy_margin_db` up, and the others within `energy_tie_db` of it speak too. The losers are made silent before the merge, so they never come back as a fallback. The group microphone never votes, and its speech passes as before.
- `asr_recognition` gains `energies`, a JSON `{tag: snr_db}` of the personal channels that voted. Its other fields are unchanged.
- A wearer's `asr_transcription` carries `participant` and `attribution: energy`. It is stored a few microseconds after its chunk end, so the chunks of several bases that end together stay apart.
- Buckets still open at STOP are merged and uploaded, since a bucket some base never reported to would otherwise be lost.

**How the fusion counts words.** `mmla ses-fuse` adds a `p<tag>_words` column per wearer: their words whose time falls in a 3 s bucket that lists the tag (and its `energies`, when the bucket has them). Words without stamps are spread evenly over their chunk. `words`, the spurts and `dia_*` stay the group microphone's. Without one, `words` is the sum of the wearers' won words, and the spurts and `dia_*` come from every chunk. Sessions without personal microphones fuse as before.

**Binding tags.** Every audio recording of a session's manifests has a `scope` (`personal` or `group`) and a `participant`. `mmla ses-tidy --scope`, `--participant` and `--participants-in-order` set them (see the [TUI guide](../tui.md#collection)). A live Collection recording notes the wearer picked under the Collection form's Participant (else its device's default scope); what it left unbound is bound afterwards with `ses-tidy`.

**Replay.** `scripts/replay_sessions.py` gives each personal recording of a session with several microphones its own `Vimo` base, bound to its participant (the device name when none is bound), beside the group base. The personal bases run without `-dia`, and the synchronizer runs with `-bt Vimo -nb <count>`.

## Manual CLI

```bash
conda activate asr-base
# one per microphone; -b picks the Bases entry, -sid the session, -pt whom its speech is (a tag, group or
# speakers; default: the config), -spk the speakers to verify (default: every profile)
mmla asr-base -p pipelines/asr-base -c pipelines/asr-base/config.yml -m live -sid <session-id> -b <base-id> -pt speakers -spk Alice,Bob
mmla asr-base -p pipelines/asr-base -c pipelines/asr-base/config.yml -m live -sid <session-id> -b <vimo-base-id> -pt 5
# one per session; -nb how many bases to wait for (-bt <Base block> takes that block's segment length for
# the buckets when the Synchronizer section sets none)
mmla asr-sync -p pipelines/asr-base -c pipelines/asr-base/config.yml -sid <session-id> -nb 2
# a room microphone and three worn ones (bases started with -pt <tag>)
mmla asr-sync -c <cfg> -sid <sid> -bt Vimo -nb 4
```

With `-sid` a process runs as it does from the TUI: it asks nothing, starts at once and exits when the run ends with STOP. A flag left out then takes its default: `-b` the only `Bases` entry, `-nb` the number of `Bases` entries; when there is no single one to take, the window says so and asks. Without `-sid` each process shows its menu and asks for the session and for whatever its flags leave out, as before. The synchronizer asks for no base type either way: it merges every base of the session, whatever its type.

`mmla asr-base -h` lists the toggles (`-s`, `-vad`, `-nr`, `-tr`, `-dia`, `-sp`, `-hsr`), `-lang`, the language its speech is transcribed in, and `-pt`, whom its speech is attributed to. The speaker profiles of a host are listed, registered and deleted without a base's menu with `mmla asr-speakers`, which asks nothing:

```bash
mmla asr-speakers -p pipelines/asr-base --list
mmla asr-speakers -p pipelines/asr-base -c pipelines/asr-base/config.yml --register Alice -b <base-id>          # records from that base's source
mmla asr-speakers -p pipelines/asr-base -c pipelines/asr-base/config.yml --register Alice -b <base-id> --files a.wav b.m4a
mmla asr-speakers -p pipelines/asr-base --delete Alice
```

A registration takes the settings of the base type of `-b`; `-t` sets the seconds to record and `-vad`, `-nr`, `-s` the preprocessing and whether its audio is kept in the profile. The START/STOP signals can be sent with `mmla ses-ctl -c pipelines/asr-base/config.yml` instead of the TUI.

To run the server services without Docker, install one backend extra (`asr-server-wespeaker` or `asr-server-nemo`) and start one gunicorn process per service from `openmmla/services/asr/apps/`. The wespeaker backend itself is not on PyPI, so it is installed separately at the commit the Docker image uses:

```bash
pip install -e '.[asr-server-wespeaker]'
pip install "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@9ce7995648a281ba6ce6f4e7a33941672ef22779"

export PROJECT_DIR=pipelines/asr-server CONFIG_PATH=pipelines/asr-server/config.yml
gunicorn -k gevent -w 1 -b 0.0.0.0:5005 openmmla.services.asr.apps.serve_speech_transcriber:app
```

The modules are `serve_audio_inferer` (5001), `serve_audio_resampler` (5002), `serve_speech_enhancer` (5003), `serve_speech_separator` (5004), `serve_speech_transcriber` (5005) and `serve_voice_activity_detector` (5006). Each also answers `GET /<endpoint>/info` (`/transcribe/info`, through the gateway too) with what it runs — the backend, the model, the language, whether it is on CUDA — and a base that joins a session asks and notes the answers in the session's document, so that a session's measurements can be traced to the models that made them (see [Databases](../database.md#mongodb)). A server running an older openmmla has no `/info`, and the session says so instead.

## Post-time processing

Record the session first with **Collection → Collection Session** (see the [TUI guide](../tui.md#collection)), then replay it: set the base's `source` to `file` and its `source_index` to a file of the `audio/` directory listed in the collection manifest, by its full path (**Browse…** picks it). Run the pipeline as usual; the timing is taken from the file names.
