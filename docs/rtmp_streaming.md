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

In the TUI the card is **Launcher → System Services → Stream Server (MediaMTX)** with **Run mode** `docker`. Server-side recordings land in `artifacts/recordings/` of the repository (`MEDIAMTX_RECORD_DIR` in `docker/.env`).

**Native**: install the binary, then run the make target, which opens a tmux session named `mediamtx`:

```bash
# macOS
brew install mediamtx

# Linux: unpack the release for your architecture (linux_amd64, linux_arm64, linux_armv7 ...)
# from https://github.com/bluenviron/mediamtx/releases into /usr/local/bin

cd pipelines/uber-server
make mediamtx          # reads mediamtx/mediamtx.yml, records to artifacts/recordings/ of the repository
make stop-mediamtx
```

The card's Run mode `native` runs the same targets. Both run modes record to the same folder, `artifacts/recordings/` of the repository: Docker mounts it, and `make mediamtx` starts MediaMTX from `artifacts/` (`MEDIAMTX_RECORD_ROOT`), where the `./recordings` of `mediamtx.yml` resolves to it. A native MediaMTX started before this change recorded to `pipelines/uber-server/recordings/`; what is there stays there.

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

| Field | Meaning |
|---|---|
| `target` | publish URL: `rtmp://<stream-server>:1935/<app>/<name>` (also `rtsp://` or `srt://`), or `udp://<base>:<port>` / `tcp://` for raw audio straight to an ASR base |
| `read_target` | what the bases pull, e.g. `rtsp://<stream-server>:8554/<app>/<name>`; empty means `target` |
| `ssh_profile` | TUI SSH profile of the capture host, or `local` |
| `device` | `/dev/video0` (v4l2 camera) or `hw:1,0` (ALSA microphone) |
| `kind` | `audio` or `video`; inferred from the target and the device when omitted |
| `codec`, `resolution`, `fps`, `bitrate` | video encoding (`libx264`, `1920x1080`, `30`, `1M`) |
| `format`, `rate`, `channels` | audio sample format (`s16le`), rate (`16000`) and channels (`1`) |
| `record`, `record_root` | also record on the capture host, see [Recording](#recording) |

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
# macOS capture device (avfoundation) instead of v4l2
ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "0:none" \
  -c:v h264_videotoolbox -realtime true -b:v 2M -f flv rtmp://<stream-server>:1935/vfa/front
```

`ffmpeg -f v4l2 -list_formats all -i /dev/video0` and `arecord -l` list the devices on a Pi; `ffmpeg -f avfoundation -list_devices true -i ""` on a Mac. SRT needs an FFmpeg built with libsrt (`ffmpeg -protocols | grep srt`; the Debian and Raspberry Pi OS packages have it).

## Bases: pulling a stream

A base with `source: stream` picks a URL from the pullable `Streams` entries (`read_target`, else `target`; only `rtmp://`, `rtsp://` and `srt://` count) by `source_index`. `rtmp` is still accepted as the old name of that source.

The device publishes over RTMP (`target`) and the bases usually pull the same path over RTSP (`read_target`): MediaMTX serves whatever is published on a path over every protocol it has switched on. Measured against MediaMTX 1.21 with the bases' own `VideoStream`, both on one machine: the steady delay is the same either way (0.2 to 0.3 s at 15 fps with the options below), but an RTSP pull opens in under two seconds where an RTMP pull takes about six, because FFmpeg's FLV reader waits for an audio track that never comes. How the two behave over a lossy Wi-Fi link was not measured. An empty `read_target` pulls the `target`. The video bases open network streams with `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|fflags;nobuffer|flags;low_delay` unless the variable is already exported (override per base with `stream_kwargs.capture_options`), and the ASR base runs its ffmpeg decoder with the same low-latency flags. On a wired base station expect well under a second end to end; `pipelines/ips-base/docs/clock.html` is a browser clock you can film to measure it.

Frame timestamps: a managed stream's capture-side start time is recorded when the Streams tab starts it (`real-time/runtime/stream_registry.yml`), and a base that connects while the stream starts stamps frames with that time plus the media PTS, independent of the network delay. A base that joins later, or an external stream, calibrates the PTS against its own clock on the first frame, so the pipeline delay at that moment becomes a constant offset. Start the streams first, then the bases.

## Recording

### On the capture device

`record: true` on a managed stream makes the same FFmpeg process write the stream to a file while it pushes it: one H.264 encode with two outputs (the tee muxer) for video, a second PCM output for audio. The recording is not affected by the network or the server, it shares the stream's `bitrate`, and it keeps going when the server is unreachable.

Files follow the Collection layout so the same tools handle them:

```
<record_root>/streams-<date>/collection/<host label>/video/<name>_<start time>.mkv
<record_root>/streams-<date>/collection/<host label>/audio/<name>_<start time>.wav
```

`record_root` defaults to `~/artifacts` on a remote host and to `artifacts/` of the project for `local`; `streams-<date>` is the day the stream was started (`streams-20260917`); `<host label>` is the SSH profile name; `<start time>` is the unix time the stream started, which is what the file source reads back. **Stop** sends Ctrl-C and waits for FFmpeg to finalize the file before the tmux session is closed, and names the file.

A recording is filed by day and not under a session, because a stream does not belong to one: it is started once and any number of sessions pull it, one after another or at the same time (several groups in one room share a camera, each group being a session of its own). Nothing has to be chosen before a stream is started, and a stream is not restarted between sessions. What ties a recording to a session is time: the file name carries the capture-side start time, and the session's `start_time` and `end_time` are in MongoDB (**Sessions** tab).

**Download** in the Recordings row of the Streams tab fetches them. With a session chosen there, it cuts the part between that session's start and end (MongoDB) out of every stream of the card, on the capture host, with the FFmpeg that made the recording and without re-encoding, and copies only the cuts into `artifacts/<session>/collection/<host label>/video|audio/<name>_<start>.mkv|wav`. A video can only begin on a keyframe without re-encoding, so the cut is moved back onto the last keyframe before the session's start (the recorder writes one a second) and the file is named after that frame; audio is cut to the sample. The cuts are staged under `<record_root>/.session-cuts/<session>/` on the capture host and removed once they have arrived; the recordings themselves are never touched, so every session that shared the stream can take its own part. The capture host's clock decides what is cut: keep it on NTP. Without a session, Download copies what the selected stream's capture host has recorded, every day it holds and newest first, into `artifacts/streams-<date>/collection/<host label>/` on this machine; a day that is already here in full is skipped, and an interrupted transfer resumes like a Collection download. (The Collection card's own Download fetches one *session* of its recorders, which is a different thing: a Collection recorder is started by a session and ends with it, so its files are filed under that session, while a stream is started on its own and shared.) The pipelines replay the downloaded files with `source: file`, `Base.file_dir` pointing at the `video/` or `audio/` folder and `source_index` naming the file, exactly like a Collection Session recording. A camera or microphone can be either streamed or recorded by the Collection card, not both at once; `record: true` is the way to get both.

### On the server

While `record` under `pathDefaults` is on in `mediamtx.yml` (the **Server-side recording** switch on the Stream Server card's Config tab; on by default), MediaMTX writes every published path to `recordings/<app>/<name>/<start>.mp4` in ten-minute fMP4 segments (see [Run it](#run-it) for where that folder is). Segment names are the server's clock, so treat them as an archive of what arrived rather than as a capture-side recording. The server knows nothing of sessions: it records a path from the moment something is published until the publisher stops, whether or not a session runs, and its folder, `artifacts/recordings/`, is not listed as a session by the console. A segment is deleted `recordDeleteAfter` after it began, by MediaMTX itself: three days in the shipped file (**Keep recordings for** on the card's Config tab; `0s` keeps everything), so a session's footage has to be exported before then. The **Sessions** table says until when, in its **Recordings until** column, and the card's **Recordings** tab shows what the server holds, path by path with sizes and free disk, and deletes a path or everything older than a given age through the API. The footage of a session is the time range between its `start_time` and `end_time`, which the playback server returns as one file for any path:

```bash
curl -o front.mp4 "http://<stream-server>:9996/get?path=vfa/front&start=2026-09-14T10:00:00Z&duration=600"
```

**Sessions → Export Recordings** in the TUI does this for a session: it asks the control API which paths were recorded, the playback server for the stretches of each, and downloads the part between the session's start and end into `artifacts/<session>/recordings/<app>/<name>_<start>.mp4`, once per unbroken stretch (a stream that was restarted gives two files). The playback server keeps the file's clock on the requested start, so the name plus a frame's time is the server's wall clock again. Audio paths come out as `.mp4` with an AAC track; convert them (`ffmpeg -i mic1_<start>.mp4 mic1_<start>.wav`) before replaying them through ASR.

To replay a file fetched by hand through a pipeline, name it `<prefix>_<unix start time>.mp4` and use it as a `file` source.

## Troubleshooting

1. **Nothing in `/v3/paths/list`** while the device shows a running ffmpeg: the MediaMTX host or port 1935 is not reachable from the device; check the host in the stream's `target` URL (and `StreamServer.host` in System Settings, which the MediaMTX card uses) and the server firewall (macOS: System Settings → Privacy & Security → Firewall).
2. **The Streams tab says Running but the base gets no frames**: press **Probe**, which decodes two seconds of the read URL with ffmpeg on this machine; the error text is the server's answer.
3. **Grey or torn frames**: the reader chose UDP transport; the bases default to TCP, keep `rtsp_transport;tcp` in the capture options.
4. **Delay grows over the session**: the RTMP push over Wi-Fi is buffering; publish with SRT instead, or move the Pi to Ethernet.
5. **MediaMTX does not start, or the log says `port 1935 answers but 8554 does not`**: an Nginx from before the move to MediaMTX still has its `rtmp { ... }` block and holds port 1935. Press **Start** on the **Gateway (Nginx)** card, which renders the config again from the current template (no RTMP block) and reloads Nginx, then start MediaMTX.
6. **ffmpeg on the Pi exits at once**: **Logs** shows the tmux pane; the usual causes are a wrong device name (`v4l2-ctl --list-devices`, `arecord -l`) or a resolution the camera cannot deliver as MJPEG.
