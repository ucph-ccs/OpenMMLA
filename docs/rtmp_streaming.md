# Streaming

How camera and microphone streams move through OpenMMLA: FFmpeg on the capture device pushes to a **MediaMTX** server, the bases pull from it, MediaMTX records every stream on the server, and a managed stream can keep a raw recording on the capture device at the same time.

MediaMTX replaced the Nginx RTMP module. The `rtmp://` push from the devices is unchanged; what changed is that the bases can pull the same stream as RTSP, that every stream is recorded, and that a control API shows what is being published. Nginx remains the load balancer for the AI services only, see [Nginx](nginx.md).

## Server: MediaMTX

All ports are on the host of **System Settings → Connections → Stream Server (MediaMTX)** in the TUI, which need not be the machine that runs Nginx. `<stream-server>` stands for that host in the URLs below:

| Port | Setting | Used for |
|---|---|---|
| 1935 | `rtmp_port` | RTMP publish from cameras and microphones |
| 8554 | `rtsp_port` | RTSP, what the bases pull |
| 8890/udp | | SRT publish or read |
| 9997 | | control API, `http://<stream-server>:9997/v3/paths/list` |
| 9996 | | playback server for the server-side recordings |
| 9998 | | Prometheus metrics |

### Run it

**Docker (default)**: MediaMTX is part of the infrastructure stack in `docker/docker-compose.infra.yml`, next to InfluxDB and MongoDB:

```bash
docker compose -f docker/docker-compose.infra.yml up -d mediamtx
```

In the TUI the card is **Launcher → System Services → Stream Server (MediaMTX)** with **Run mode** `docker`. Server-side recordings land in `artifacts/streams/server/` of the repository (`MEDIAMTX_STREAMS_DIR` in `docker/.env`).

**Native**: install the binary, then run the make target, which opens a tmux session named `mediamtx`:

```bash
# macOS
brew install mediamtx

# Linux: unpack the release for your architecture (linux_amd64, linux_arm64, linux_armv7 ...)
# from https://github.com/bluenviron/mediamtx/releases into /usr/local/bin

cd pipelines/uber-server
make mediamtx          # reads mediamtx/mediamtx.yml, records to artifacts/streams/server/ of the repository
make stop-mediamtx
```

The card's Run mode `native` runs the same targets. Both run modes record to the same folder, `artifacts/streams/server/` of the repository, because the `recordPath` of `mediamtx.yml` is `./streams/server/...`: Docker mounts the folder at `/streams/server` of the container, whose working directory is `/`, and `make mediamtx` starts MediaMTX from `artifacts/` (`MEDIAMTX_RECORD_ROOT`). Older versions recorded to `artifacts/recordings/` (natively, at first, to `pipelines/uber-server/recordings/`); the console no longer reads either folder, and what is there stays there. The variable was `MEDIAMTX_RECORD_DIR` then; it is `MEDIAMTX_STREAMS_DIR` now, so a `docker/.env` copied from the old example, which names the old folder, is ignored and the new default applies. The console creates `artifacts/streams/server` itself before it starts the container, so the folder, and `artifacts/streams/` above it, belong to you and not to root. The Stream Server's host reads its own `mediamtx.yml`, so pull the repository there too.

Capture hosts that recorded before this change keep their files under `<record_root>/streams-<YYYYMMDD>/collection/<host label>/`, which the console no longer lists, prunes or exports. Move them into the new layout on each such host (it leaves nothing it cannot place, and removes the old folders once they are empty):

```bash
cd ~/artifacts && for d in streams-2[0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do [ -d "$d" ] || continue; s=${d#streams-}; day="${s:0:4}-${s:4:2}-${s:6:2}"; mkdir -p "streams/capture/$day" && for h in "$d"/collection/*; do [ -d "$h" ] && mv -n "$h" "streams/capture/$day/"; done; find "$d" -name .DS_Store -delete; find "$d" -depth -type d -empty -delete; done
```


### Configuration

Both ways read `pipelines/uber-server/mediamtx/mediamtx.yml`. Stream paths are created on the fly (`all_others`), so any `<app>/<name>` in a publish URL works without editing the server. Recording is on for every path (fMP4, ten-minute segments, deleted three days after they began: `recordDeleteAfter`), the API, playback and metrics are on, and there is no authentication, which suits a trusted lab network; the file points at the MediaMTX reference for credentials.

### Check what is published

```bash
curl http://<stream-server>:9997/v3/paths/list
```

```bash
ffplay -rtsp_transport tcp rtsp://<stream-server>:8554/vfa/front
```

## Devices: pushing a stream

Streams are declared once per pipeline under `Streams` in `pipelines/<pipeline>-base/config.yml`. An entry with an `ssh_profile` is *managed*: **Launcher → Pipelines → <pipeline> → Streams** builds the FFmpeg command below and runs it inside a tmux session named `mmla-stream-<name>` on that host over SSH, with **Start**, **Stop**, **Logs** and **Probe**. An entry without an `ssh_profile` is *external*, already running somewhere, and only pulled from.

The tab's **Status** is the stream's FFmpeg, not its tmux session, which outlives it: `Running`, `Starting` while a Mac's Terminal window is still to start it (see below), `Exited` when FFmpeg stopped by itself (the pane keeps what it said: **Logs**; **Start** starts it again), `Stopped`, `No answer` from the host. **Start** waits until FFmpeg has run for three seconds, and otherwise quotes its last lines. The **Stream Server** column says what the server of System Settings receives: `● live`, `○ not live`, `no answer`, or `-` for a stream that goes elsewhere.

A Mac captures through AVFoundation rather than V4L2 and ALSA: `device` is a camera's index or name (`0`, the default, or `FaceTime HD Camera`), a microphone is `:0`. macOS lets nothing started over SSH use the camera or the microphone (the process counts as sshd, which is never asked, and FFmpeg waits for frames forever), so on a Mac reached over SSH the Streams tab starts FFmpeg from a Terminal window on the Mac's own screen, as the Collection recorders do. The window closes by itself once FFmpeg runs: FFmpeg goes on in a session of its own, so nothing stays on the Mac's screen and no window closed by hand can stop a stream. The tmux session follows FFmpeg's output and passes **Stop** on to it. Someone has to be logged in on the Mac, with Terminal allowed under **System Settings → Privacy & Security → Camera** (and **Microphone**). A stream `local` to a Mac runs directly, as the console already runs in the Mac's desktop session, unless the console itself was reached over SSH.

| Field | Meaning |
|---|---|
| `target` | publish URL: `rtmp://<stream-server>:1935/<app>/<name>` (also `rtsp://` or `srt://`), or `udp://<base>:<port>` / `tcp://` for raw audio straight to an ASR base |
| `read_target` | what the bases pull, e.g. `rtsp://<stream-server>:8554/<app>/<name>`; empty means `target` |
| `ssh_profile` | TUI SSH profile of the capture host, or `local`; picked in the SSH Profile column of the Streams tab |
| `device` | `/dev/video0` (v4l2 camera) or `hw:1,0` (ALSA microphone); on a Mac `0` (camera index or name) or `:0` (microphone) |
| `kind` | `audio` or `video`, a dropdown on the Config tab. Left empty, a `udp://` or `tcp://` target or a sound device (`hw:1,0`, a Mac's `:0`) makes it audio, and anything else is what the card is for: audio on ASR, video on IPS and VFA. A Mac's first microphone pushed over RTMP names no device, so only the card can tell; **+ Add Stream** on the ASR card writes `kind: audio` |
| `codec`, `resolution`, `fps`, `bitrate` | video encoding (`libx264`, `1920x1080`, `30`, `1M`) |
| `format`, `rate`, `channels` | audio sample format (`s16le`), rate (`16000`) and channels (`1`) |
| `record`, `record_root`, `record_keep_days` | also record on the capture host, and for how many days the recordings stay there (`0`: until deleted), see [Recording](#recording) |

The address of the stream server is typed once, under **System Settings → Connections → Stream Server (MediaMTX)**. In the Config tab of a pipeline, `target` takes the path alone (`ips/cam-1`): **Save** completes it to `rtmp://<stream-server>:<rtmp_port>/ips/cam-1` and fills an empty `read_target` with `rtsp://<stream-server>:<rtsp_port>/ips/cam-1`, and **+ Add Stream** opens a new entry on those two URLs. The config file always holds the full URLs, which is what the bases and FFmpeg read; a full URL typed by hand is kept as written. While the Stream Server is `localhost`, a path is not completed for a stream captured on another machine (that machine would publish to itself): put the server's name under System Settings first. When the Stream Server is saved with another host or port, the stream URLs of the local pipeline configs that named the old address follow it; URLs that point elsewhere are left alone, and **Sync to Remote** on the pipeline's Config tab takes the change to another host.

The commands the Streams tab runs, and what to run by hand on a device without SSH access:

```bash
# camera on a Raspberry Pi -> MediaMTX (RTMP)
ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 \
  -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 30 -keyint_min 30 -sc_threshold 0 \
  -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
  -b:v 1M -maxrate 2M -bufsize 2M \
  -f flv rtmp://<stream-server>:1935/vfa/front
```

```bash
# microphone -> MediaMTX (AAC); the ASR base pulls it as rtsp://<stream-server>:8554/asr/mic1
ffmpeg -f alsa -ac 1 -ar 16000 -i hw:1,0 -c:a aac -b:a 128k -f flv rtmp://<stream-server>:1935/asr/mic1
```

```bash
# microphone -> ASR base directly (raw PCM over UDP, no server in between)
ffmpeg -f alsa -ac 1 -ar 16000 -i hw:1,0 -c:a pcm_s16le -f s16le udp://<base>:5001
```

```bash
# SRT instead of RTMP: loss-tolerant on Wi-Fi with a fixed latency budget
ffmpeg ... -f mpegts "srt://<stream-server>:8890?streamid=publish:vfa/front"
```

```bash
# camera on a Mac (AVFoundation) -> MediaMTX. AVFoundation states no frame rate:
# without -r, ffmpeg takes the wallclock timestamps for 1000k fps and duplicates frames without end
ffmpeg -fflags +genpts -use_wallclock_as_timestamps 1 \
  -f avfoundation -pixel_format nv12 -framerate 30 -video_size 1920x1080 -i 0:none \
  -c:v libx264 -pix_fmt yuv420p -r 30 -preset ultrafast -tune zerolatency \
  -g 30 -keyint_min 30 -sc_threshold 0 \
  -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
  -b:v 1M -maxrate 2M -bufsize 2M \
  -f flv rtmp://<stream-server>:1935/vfa/front
```

`ffmpeg -f v4l2 -list_formats all -i /dev/video0` and `arecord -l` list the devices on a Pi; `ffmpeg -f avfoundation -list_devices true -i ""` on a Mac, where a size the camera cannot deliver makes FFmpeg list the ones it can (`-video_size 1x1`). SRT needs an FFmpeg built with libsrt (`ffmpeg -protocols | grep srt`; the Debian and Raspberry Pi OS packages have it).

## Bases: pulling a stream

A base with `source: stream` pulls the `Streams` entry its `source_index` names (`read_target`, else `target`; only `rtmp://`, `rtsp://` and `srt://` count); the TUI's Bases form offers them in a dropdown and stores the name. A number is still read as the position among those pullable entries, as older configs have it. A name or number that matches none of them stops the base with the list of streams there are, where it used to pull the first one. `rtmp` is still accepted as the old name of that source.

The device publishes over RTMP (`target`) and the bases usually pull the same path over RTSP (`read_target`): MediaMTX serves whatever is published on a path over every protocol it has switched on. Measured against MediaMTX 1.21 with the bases' own `VideoStream`, both on one machine: the steady delay is the same either way (0.2 to 0.3 s at 15 fps with the options below), but an RTSP pull opens in under two seconds where an RTMP pull takes about six, because FFmpeg's FLV reader waits for an audio track that never comes. How the two behave over a lossy Wi-Fi link was not measured. An empty `read_target` pulls the `target`. The video bases open network streams with `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay` unless the variable is already exported (override per base with `stream_kwargs.capture_options`), and the ASR base runs its ffmpeg decoder with the same low-latency flags. On a wired base station expect well under a second end to end; `pipelines/ips-base/docs/clock.html` is a browser clock you can film to measure it.

Frame timestamps: a managed stream's capture-side start time is recorded when the Streams tab starts it (`real-time/runtime/stream_registry.yml`), and a base that connects while the stream starts stamps frames with that time plus the media PTS, independent of the network delay. A base that joins later, or an external stream, calibrates the PTS against its own clock on the first frame, so the pipeline delay at that moment becomes a constant offset. Start the streams first, then the bases.

## Recording

A stream is recorded in up to two places, and neither knows anything of sessions: the Stream Server records every path published to it, and a managed stream with `record: true` also records on the machine that captures it. A session's part of both is taken afterwards, by its time range, with **Sessions → Export Streams** ([below](#a-sessions-part-sessions-export-streams)). Where the files are:

```
# on each capture host (record_root: ~/artifacts on a remote host, artifacts/ of the project for local)
<record_root>/streams/capture/<YYYY-MM-DD>/<host label>/video/<name>_<start time>.mkv
<record_root>/streams/capture/<YYYY-MM-DD>/<host label>/audio/<name>_<start time>.wav

# on the Stream Server's host, MediaMTX's own segments
artifacts/streams/server/<app>/<name>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4

# on the console, a session's part (Sessions → Export Streams)
artifacts/<session>/streams/server/<app>/<name>_<start time>.mp4
artifacts/<session>/streams/capture/<host label>/video/<name>_<start time>.mkv
artifacts/<session>/streams/capture/<host label>/audio/<name>_<start time>.wav

# on the console, whole capture-side files tied to no session (Streams tab → Manage)
artifacts/streams/capture/<YYYY-MM-DD>/<host label>/video|audio/<name>_<start time>.mkv|wav
```

`artifacts/streams/` holds no session, so the **Sessions** tab does not list it. `artifacts/<session>/collection/<host>/` is for the Collection card's own recorders only.

### On the capture device

`record: true` on a managed stream makes the same FFmpeg process write the stream to a file while it pushes it: one H.264 encode with two outputs (the tee muxer) for video, a second PCM output for audio. The recording is not affected by the network or the server, it shares the stream's `bitrate`, and it keeps going when the server is unreachable.

The file goes to `<record_root>/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/<name>_<start time>.<mkv|wav>` on the capture host. `record_root` defaults to `~/artifacts` on a remote host and to `artifacts/` of the project for `local`; `<YYYY-MM-DD>` is the day the stream was started, as the console's clock gives it at **Start**; `<host label>` is the SSH profile name (this machine's short host name for `local`); `video/` or `audio/` (and `.mkv` or `.wav`) is the stream's `kind` as its card reads it (above); `<start time>` is the unix time the stream started, which is what the file source reads back. **Stop** sends Ctrl-C and waits for FFmpeg to finalize the file before the tmux session is closed, and names the file. The console names the file when it starts the stream, so a stream started before this layout keeps writing where it began until it is stopped; its next **Start** records here.

The file is written while the stream runs, not at Stop: FFmpeg appends to it a fraction of a second at a time (at the default `1M`, about 450 MB an hour or 11 GB a day), and one run of a stream, from Start to Stop, is one file however long it lasts, filed under the day it started. What Stop adds is the end of the file, its seek index and duration; a file that never got them (FFmpeg killed, a power cut) still plays, and `ffmpeg -i <file>.mkv -c copy <fixed>.mkv` writes them. The push does not wait on the file either: when the file cannot be written, a full disk for one, FFmpeg 7 drops that output and goes on streaming, so the recording ends while Status still says Running. **Manage** (below) shows the room left on each capture host.

A recording is filed by day and not under a session, because a stream does not belong to one: it is started once and any number of sessions pull it, one after another or at the same time (several groups in one room share a camera, each group being a session of its own). Nothing has to be chosen before a stream is started, and a stream is not restarted between sessions. What ties a recording to a session is the session's record and time: each base notes in the session's MongoDB document which stream it takes when it joins, and when it leaves (its `sources`, see [Database](database.md#mongodb)), the file name carries the capture-side start time, and the session's `start_time` and `end_time` are in MongoDB (**Sessions** tab).

**Manage**, in the Recordings row of the Streams tab, lists what the streams of the card have recorded on each capture host, newest day first, with the size of each file, the room left on the host's disk and the file a stream is writing now. **Download File** copies the selected recording to this machine and **Download Day** every recording of its day on that host, whole and tied to no session, into `artifacts/streams/capture/<YYYY-MM-DD>/<host label>/video|audio/`: a copy already here in full is skipped, and one cut off (**Cancel**, a dropped link) resumes at the next press, like a Collection download. The Streams tab no longer cuts a session's part out of the recordings: that is **Sessions → Export Streams**, [below](#a-sessions-part-sessions-export-streams), which takes the Stream Server's copy along with it. (The Collection card's own Download fetches one *session* of its recorders, which is a different thing: a Collection recorder is started by a session and ends with it, so its files are filed under that session, while a stream is started on its own and shared.) The pipelines replay the copied files with `source: file`, `Base.file_dir` pointing at the `video/` or `audio/` folder and `source_index` naming the file, exactly like a Collection Session recording. A camera or microphone can be either streamed or recorded by the Collection card, not both at once; `record: true` is the way to get both.

Manage also deletes a file, a day, or everything last written longer ago than a given age, each after a second press. Nothing on a capture host deletes these files by itself. **Keep recordings for** there sets `record_keep_days` of every stream of the card (also a field of each stream on the Config tab; `0` keeps them): a recording last written longer ago than that is deleted at its stream's next **Start** and at **Refresh** on the Streams tab, which say what they removed, or at once with **Delete Expired**, and the Record column shows the time (`yes, 7 d`). The file a stream is writing is never deleted, by hand or by the keep time: a deleted file that is still being written keeps its room until the stream stops, and that take is lost, so stop the stream first. The folders a deletion leaves empty go with it, except today's. Deleting on a capture host leaves the copies on this machine alone. The whole-file copies under `artifacts/streams/` belong to no session, so **Sessions → Delete Artifacts** does not reach them: remove a day's folder by hand once it is no longer needed. A session's own part, under `artifacts/<session>/streams/`, goes with that session's Delete Artifacts. The Stream Server keeps its own recordings for as long as its card says (below).

### On the server

While `record` under `pathDefaults` is on in `mediamtx.yml` (the **Server-side recording** switch on the Stream Server card's Config tab; on by default), MediaMTX writes every published path to `artifacts/streams/server/<app>/<name>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4` on the Stream Server's host, in ten-minute fMP4 segments (see [Run it](#run-it) for how both run modes land there). Segment names are the server's clock, so treat them as an archive of what arrived rather than as a capture-side recording. The server knows nothing of sessions: it records a path from the moment something is published until the publisher stops, whether or not a session runs. A segment is deleted `recordDeleteAfter` after it began, by MediaMTX itself: three days in the shipped file (**Keep recordings for** on the card's Config tab; `0s` keeps everything), so a session's footage has to be exported before then. The **Sessions** table says until when, in its **Recordings until** column, and the card's **Recordings** tab shows what the server holds, path by path with sizes and free disk on `artifacts/streams/server/`, and deletes a path or everything older than a given age through the API. The playback server returns any time range of a path as one file:

```bash
curl -o front.mp4 "http://<stream-server>:9996/get?path=vfa/front&start=2026-09-14T10:00:00Z&duration=600"
```

The playback server keeps the file's clock on the requested start, so the start in the file's name plus a frame's time is the server's wall clock again. To replay a file fetched by hand through a pipeline, name it `<prefix>_<unix start time>.mp4` and use it as a `file` source. Audio paths come out as `.mp4` with an AAC track; convert them (`ffmpeg -i mic1_<start>.mp4 mic1_<start>.wav`) before replaying them through ASR.

### A session's part: Sessions → Export Streams

**Export Streams** on the **Sessions** tab takes the selected session's part of every stream it used, both copies of it, with nothing to fill in. It reads the session's record from MongoDB again at the press (a base notes the stream it takes when it joins, which can be after the table was listed) and goes by its `sources`: which stream each base took, the URL it pulled, and the machine that captures and records the stream ([Database](database.md#mongodb)). The part is the time between the session's `start_time` and its `end_time`; for a session that was never ended, until its last base left; while a base is still in it, up to now. The log says which end it went by. Files land under `artifacts/<session>/streams/`:

- **From the Stream Server** (HTTP, no SSH): each source whose URL names the Stream Server of **System Settings** (by name, alias or address, or as `localhost` from a base that ran on the server's host) gives its path there, taken exactly as named, so another group's camera, recorded by the same server at the same time, stays out. The playback server is asked for the stretches of each path, and the part inside the session goes to `artifacts/<session>/streams/server/<app>/<name>_<start>.mp4`, one file per unbroken stretch (a stream that was restarted gives two). A stream published to another server is named in the log and skipped. Audio pushed straight to an ASR base (`udp://`, `tcp://`) never goes through a server, so only its capture host can have it.
- **From the capture hosts** (SSH): each of those streams that the console runs with Record on is cut on its capture host, with the FFmpeg that recorded it and without re-encoding. A video can only begin on a keyframe without re-encoding, so the cut is moved back onto the last keyframe before the session's start (the recorder writes one a second) and named after that frame; audio is cut to the sample. The cuts are staged under `<record_root>/streams/.session-cuts/<session>/<host label>/` on the capture host, fetched like a Collection download (resumable) into `artifacts/<session>/streams/capture/<host label>/video|audio/<name>_<start>.mkv|wav`, and their staging is removed once they have arrived; a stream captured on this machine is cut straight into place. The folders are named in the session's `manifest.yml` as file sources (audio for ASR, video for IPS and VFA). The recordings themselves are never touched, so every session that shared a stream can take its own part. The capture host's clock decides what is cut: keep it on NTP. A stream whose Record was off, or that someone else publishes (no SSH profile), is named in the log with where its only copy is: on the Stream Server.

A file already here in full is not fetched or cut again. One exported while the session was still going has the same name (the name only says where it starts) but is shorter, so its length is checked with `ffprobe` and it is replaced by the full one (without `ffprobe` on this machine, a Stream Server clip of an ended session counts as final). While the export runs, a progress row under the button shows what it is doing; its **Cancel** stops it between steps, and a clip being downloaded at once. What arrived stays, and cuts made on a capture host stay there until the next press fetches them. **Export All** ends with Export Streams, after the measurements and visualizations.

A session with no record of its streams (one begun before the bases noted them, or one no base could note its stream in) takes every path the Stream Server recorded while it ran, and every stream with Record on in this machine's ASR, IPS and VFA configs (`pipelines/<asr|ips|vfa>-base/config.yml`), and the log says so. File names carry the time the file starts at, so a base replays either copy with `source: file` and `Base.file_dir` on one of these folders.

## Troubleshooting

1. **Nothing in `/v3/paths/list`** while the device shows a running ffmpeg: the MediaMTX host or port 1935 is not reachable from the device; check the host in the stream's `target` URL (and `StreamServer.host` in System Settings, which the MediaMTX card uses) and the server firewall (macOS: System Settings → Privacy & Security → Firewall).
2. **Running, but the base gets no frames**: look at the **Stream Server** column (**Refresh** asks again). `○ not live` while FFmpeg runs means nothing arrives: **Logs** shows what FFmpeg says, and a Mac may be asking on its screen whether Terminal may use the camera. **Probe** decodes two seconds of the read URL with ffmpeg on this machine; the error text is the server's answer, and `404 Not Found` means nobody publishes that path.
3. **Grey or torn frames**: the reader chose UDP transport; the bases default to TCP, keep `rtsp_transport;tcp` in the capture options.
4. **Delay grows over the session**: the RTMP push over Wi-Fi is buffering; publish with SRT instead, or move the Pi to Ethernet.
5. **MediaMTX does not start, or the log says `port 1935 answers but 8554 does not`**: an Nginx from before the move to MediaMTX still has its `rtmp { ... }` block and holds port 1935. Press **Start** on the **Gateway (Nginx)** card, which renders the config again from the current template (no RTMP block) and reloads Nginx, then start MediaMTX.
6. **Status says Exited**: FFmpeg stopped by itself; **Start** quoted its last lines and **Logs** shows the whole pane. The usual causes are a wrong device name (`v4l2-ctl --list-devices`, `arecord -l`; on a Mac `ffmpeg -f avfoundation -list_devices true -i ""`) or a resolution or frame rate the camera cannot deliver.
7. **A Mac's stream never starts** (`Could not open a Terminal window`, or `did not start ffmpeg`): nobody is logged in on the Mac's screen, which is where macOS lets FFmpeg use the camera.
