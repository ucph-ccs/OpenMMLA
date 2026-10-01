# Dashboard

The dashboard shows what a session recorded: live while it runs, as a replay once it has ended, and as an analysis report of its speech, space and attention. It reads the raw measurements the pipelines write to InfluxDB, adds the session's MongoDB document when MongoDB answers, and plays live camera video from MediaMTX when the session's cameras stream through it.

It is a Flask app run by gunicorn, an optional Celery worker for the report jobs, and a static frontend (plain HTML, CSS and JavaScript modules, no build step and nothing loaded from the internet) served by the same Flask process. All computation is in `openmmla/analytics/report/`; the backend is `pipelines/uber-server/dashboard/flask-backend/` and the pages are in `pipelines/uber-server/dashboard/frontend/`.

```
browser ──HTTP + Server-Sent Events──> Flask (gunicorn -k gevent -w 1, port 5050)
                                          ├── InfluxDB      measurements (required)
                                          ├── MongoDB       session documents (optional, read only)
                                          ├── Redis db 1    queue of the report worker
                                          └── MediaMTX API  which streams are live (optional)
report jobs: on the Celery worker (celery -A dashboard.celery worker) when one listens,
             else in `python dashboard.py precompute` processes the web process starts
browser ──WebRTC (WHEP)──> MediaMTX :8889   live camera video (when enabled and the session streams)
```

## Prerequisites

- The `uber-server` conda environment: create it from the TUI's Environment tab, or by hand:

    ```bash
    conda create -n uber-server python=3.10 -y
    conda activate uber-server
    pip install -e '.[uber-server]'
    ```

- InfluxDB, which holds the measurements, and Redis, the queue of the report worker. MongoDB is optional: with it the dashboard knows each session's cameras, microphones, streams, IPS camera placement and software versions, and lists sessions that have no measurements yet. MediaMTX is optional and only needed for live video. All four are described in [System Services](system_services.md).

## Configuration

The backend reads `pipelines/uber-server/dashboard/flask-backend/config.yml`. Its `InfluxDB`, `MongoDB` and `Redis` sections are among those the TUI fills from **System Settings → Connections** when they are saved there, and at startup the same sections of `config/system_services.yml` on the dashboard's machine replace them (unless the file lists them under `SystemServicesOverride`). To edit it by hand, copy `config_template.yml` next to it and fill in:

```yaml
InfluxDB:
  url: http://<influxdb-host>:8086
  token: <influxdb-token>
  org: admin
  bucket: mmla-data

MongoDB:
  url: mongodb://<mongodb-host>:27017
  db: openmmla

Redis:
  host: <redis-host>
  port: 6379
  db: 1          # a database number other than 0, so the Celery queue stays apart from the session control bus
```

MediaMTX is found through **System Settings → Stream Server** (`config/system_services.yml` on the dashboard's machine: `host`, `api_port`, `playback_port`, `webrtc_port`), re-read every minute; without it the dashboard tries a `StreamServer` section of `config.yml`, then the InfluxDB host with the MediaMTX default ports. See [Camera tiles and live video](#camera-tiles-and-live-video).

The backend listens on port 5050 by default. Change it under **System Settings → Connections → Dashboard (Flask)**; the TUI passes it to `make flask` as `DASHBOARD_PORT` and the Status tab probes that host and port. The frontend needs no configuration: it talks to the backend over same-origin URLs.

Environment variables, all optional. Set the first four the same way for the web process and the worker:

| Variable | Default | What it does |
|---|---|---|
| `DASHBOARD_CACHE_DIR` | `flask-backend/cache/` | where computed report parts and job states are kept (see [Report jobs](#report-jobs-and-the-worker)) |
| `DASHBOARD_JOBS` | `auto` | `celery` always queues report jobs for the worker, `local` always runs them in processes of the web server, `auto` queues them only while a worker consumes the dashboard's queue |
| `DASHBOARD_CELERY_QUEUE` | `mmla-dashboard` | the Celery queue of the report jobs |
| `DASHBOARD_WORKER_CONCURRENCY` | `2` | how many report jobs one Celery worker runs at once (a video job holds a whole session's video features in memory) |
| `DASHBOARD_PORT` | `5050` | port of `make flask` and of `python dashboard.py serve` |
| `OPENMMLA_SYSTEM_SERVICES_CONFIG` | `config/system_services.yml` of the repository | another System Settings file to merge into `config.yml` |

## Running

From the TUI: **Launcher → System Services → Dashboard (Flask)** and its worker **Dashboard (Celery)**, Start on each. From a shell where `conda` is on the PATH:

```bash
cd pipelines/uber-server
make flask DASHBOARD_PORT=5050   # gunicorn with the gevent worker, in a tmux session named flask
make celery                      # the report worker, in a tmux session named celery
```

Then open `http://localhost:5050` on the server, or `http://<dashboard-host>:5050` from any device on the network. Keep gunicorn at one worker (`-w 1`): the report job bookkeeping and the [shared live feeds](#live-data-stream) live in that process, and a second worker would keep its own.

The Celery worker is optional. Without one listening on the dashboard's queue, the web process runs each report job as a `python dashboard.py precompute` process of its own (see [Report jobs](#report-jobs-and-the-worker)), so the report still gets computed. With one, the jobs run in the worker instead, `DASHBOARD_WORKER_CONCURRENCY` at a time.

After pulling new dashboard code, restart both processes (`make stop-flask flask` and `make stop-celery celery`, or Stop and Start on the two cards): a running gunicorn and a running worker keep the code they started with, while the pages are read from disk on every request.

For development, `python dashboard.py serve [--port N]` in `flask-backend/` runs the same app on gevent's own server, without gunicorn. It behaves like the deployed app, shared live feeds included.

## Deploying on server-01

server-01 runs the dashboard from its own checkout, on the machine that holds Redis and the Docker stack of InfluxDB, MongoDB and MediaMTX. To bring it to new dashboard code:

1. In server-01's checkout of the repository, pull and update the environment (the dashboard needs `pymongo` and no longer `flask-socketio`):

    ```bash
    git pull
    source ~/miniforge3/etc/profile.d/conda.sh     # where conda is installed there: a plain SSH login does not put it on the PATH
    conda activate uber-server
    pip install -e '.[uber-server]'
    ```

2. Restart the web process and the worker (or Stop and Start on the two cards, which run the same targets there):

    ```bash
    cd pipelines/uber-server
    make stop-flask flask           # add DASHBOARD_PORT=<port> when the Dashboard section sets another port
    make stop-celery celery
    ```

    The new worker consumes the `mmla-dashboard` queue. A worker started from older dashboard code consumes the default `celery` queue and never takes these jobs; while only such a worker runs, `auto` mode sees no worker on the queue and runs the jobs in local processes, so nothing waits on it, but restart it all the same. The explorer's **Report worker** chip reads `ok` once the new worker answers (within 30 s) and `local` while jobs run in the web server's processes.

3. Optionally compute every session's report ahead of the first visit. The light jobs take seconds per session; the video jobs take longer, mostly reading the video features out of InfluxDB, and run faster on server-01 itself than over the network. Run it in tmux, and not beside another job that uses every core:

    ```bash
    cd dashboard/flask-backend      # from pipelines/uber-server
    python dashboard.py precompute --all
    ```

4. Turn on live video in MediaMTX (once). `pipelines/uber-server/mediamtx/mediamtx.yml` already has WebRTC on; the container needs to know the address browsers reach the host at, and has to be recreated to read the new configuration and publish the new ports:

    ```bash
    # docker/.env on server-01
    MEDIAMTX_WEBRTC_HOSTS=100.x.y.z      # server-01's tailnet IP (or a name browsers resolve); comma-separated

    # from the repository root
    docker compose -f docker/docker-compose.infra.yml up -d --force-recreate mediamtx
    curl -s http://server-01:9997/v3/config/global/get | grep -o '"webrtc":[a-z]*'   # "webrtc":true
    ```

    Recreating the container stops every stream it serves for a few seconds, so do it between sessions. `INFRA_LOOPBACK_ADDRESS`, which server-01's `docker/.env` already sets, now also publishes the MediaMTX API and playback ports on the loopback address, where the dashboard on the same machine reaches them by the name `server-01`. The details are under [Camera tiles and live video](#camera-tiles-and-live-video).

## Pages

| Page | URL | Older URL, still served |
|---|---|---|
| Sessions | `/` | |
| Live | `/live?session=<id>` | `/realtime?session=<id>` |
| Analysis | `/analysis?session=<id>` | `/posttime?session=<id>` |

Participants are shown by their AprilTag badge id (`Tag 0`), never by name. Each tag keeps one colour on a page; the colours are assigned in tag order to the participants actually seen, and a tag seen in under 2 % of the windows is greyed as rare. A group microphone's speakers are anonymous voices from diarization (`Voice 2`), which are not linked to badges; worn microphones are attributed to their wearer's tag. Every chart card has a **Table** toggle with the numbers behind it, and missing values read `n/a`.

### Sessions

Every session found in InfluxDB, plus those MongoDB knows that have no measurements yet, newest first and grouped by month. A row shows when it was recorded (from the session id, not the time the MongoDB document was made), its duration, the modalities it holds with their coverage (`IPS <n> windows, <p> % coverage`), whether speech came from a group microphone or worn ones, and whether its analysis is ready. The header shows whether InfluxDB, MongoDB, the report worker and the stream server answer, and an InfluxDB error says what failed instead of showing an empty list.

The search field, the **Task** and **Speech** filters and the sort order narrow the list, and the URL keeps them, so a filtered list can be bookmarked or sent: `/?q=group+01&task=<task>&speech=worn&sort=oldest` (`speech` is `group` or `worn`, `sort` is `newest`, `oldest` or `longest`). Every word of the search has to start a word of the session's id, title, date, month or speech setup, or the whole search has to be part of the id. `/` jumps to the search field and Escape clears it.

A **Live now** band at the top lists the sessions that wrote data in the last 20 seconds, each with a running clock and the age of its newest data, re-read every 5 seconds through the light `/api/sessions/<id>/state` route. The list itself refreshes every 30 seconds, and a tab that is hidden asks for nothing.

### Live

A running session is followed as it is written. An ended one opens paused five minutes before its end (a session shorter than ten minutes opens five minutes after its start, or at its end when it is shorter than five), with the five minutes before that moment loaded; **Play** then plays the last five minutes, at 1x to 16x, and the scrubber, **Jump to end** (the final five minutes) and, once the replay reaches the end, **Replay from start** move through the rest. A paused replay keeps no stream open: Play reconnects at the paused moment, and a jump loads the five minutes before the new moment. A live session can also be scrubbed back, which switches to replay until **Back to live**. `?mode=follow` in the URL opens any session in follow mode; on an ended session that shows its last five minutes with a note that it has ended and a **Replay this session** button. The status pill says `Live` and how far behind the newest data is, `Replay 4x`, `Paused`, `Ended` or `Reconnecting`.

- **Health strip**: the age of the newest ASR, IPS and VFA record, how many cameras sent frames, the transcript's lag, and the tags seen in the last 10 seconds.
- **Indicators** over a trailing window (5 minutes by default), each with a per-minute sparkline: speech activity, turn switches per minute, speaking balance, median distance between badges, joint attention above baseline (over the pairs seen together in at least 12 frames a minute, weighted by their frames), and social gaze (the share of every camera frame a pupil is seen in, unreadable gaze included, that rests on a partner's face or hands, as on the Analysis page).
- **Room**: a top-down plan of the badges in metres, with 30-second trails, heading arrows, who faces whom, the cameras, and the distance of a pair on hover. The floor is estimated from the badges' own orientation (gravity), so the plan is level even with a tilted main camera.
- **Transcript**: the newest chunks at the bottom, each with its speaker; diarized words are underlined in their voice's colour, and a worn microphone's crosstalk is dimmed.
- **Activity**: a timeline of the last 2, 5 or 10 minutes: speech activity, who speaks, presence and gaze category per tag, and the distance of each pair.
- **Who looks at whom**, **Speaking share** and **Pairs** (distance now, time within 1 m, joint attention against its baseline).
- **Cameras** (collapsed until opened): one tile per camera with the VFA overlay of the newest frame set: boxes, COCO-17 skeletons, AprilTag centres with their tag, and gaze rays with the category they land on. With a live stream the overlay is drawn over the video, otherwise on a blank tile ("Skeleton view"). See [Camera tiles and live video](#camera-tiles-and-live-video).

What each part needs from the pipelines:

| Part | Needs | Written by |
|---|---|---|
| Speech, transcript, speaking share | `asr_recognition`, `asr_transcription` | ASR synchronizer and bases; anonymous voices need diarization (`-dia`, see [voices across chunks](pipelines/asr.md#voices-across-chunks)) |
| Room, distances, facing, presence | `ips_translation`, `ips_rotation`, `ips_relation` | IPS synchronizer |
| Skeletons, tags and gaze on the camera tiles, gaze categories, who looks at whom, joint attention | `vfa_features` | VFA synchronizer with **Pose** (`-pose`) and **Gaze** (`-gaze`) on; one frame set per `Base.keyframe_interval` (about 1 s for a pose run), see [features endpoint](pipelines/vfa/index.md#features-endpoint-skeletons-and-gazes) |
| Live video under the overlay | a stream on MediaMTX that the session's bases pulled | MediaMTX, see below |

A part whose modality did not run stays empty and says so; the rest of the page works.

### Analysis

The report of one session, built from the four report parts below as each becomes ready (a running job shows its progress, `Reading video features: 20 min of 1 h 01 min`).

- **Overview**: indicators (speech activity, turn switches, speaking balance, median distance, joint attention above baseline, and the share of windows the interaction state calls collaborative), the session timeline (interaction state, speech activity, speakers, presence and social gaze per tag, pair distances, joint attention, and which modalities have data), and a participants table.
- **Speech**: speaking time per voice or wearer, who speaks after whom, turn and pause lengths, speech over time, and the searchable transcript.
- **Space**: occupancy heatmap or trails on the room plan, distances between pairs, facing, distance over time (one small chart per pair), and movement per tag.
- **Attention**: who looks at whom (faces or hands), where each pupil looks (partner's face, partner's hands, other people, task, elsewhere, unreadable), joint attention against each pair's own rate 20 to 40 s earlier, hand activity, which camera saw which tag, and the quality of the video data.
- **Data and exports**: coverage per modality, devices and software versions from MongoDB, and the downloads.

**Time range.** The whole page answers to one time range. Dragging across the session timeline (or across a distance chart) selects one; the bar under the section links then reads `Range 00:11:57 to 00:24:30 (12 min 33 s)` with **Clear**, and the URL keeps it as `#range=a-b` (seconds from the session start), so a link opens the same range. Every section recomputes its numbers for the range in the browser from the per-window, per-minute, turn and track data of the report parts; the heatmap and trails are redrawn from the tracks in the range, and the transcript lists the chunks in it. Without a range the page shows the totals the server computed. A few panels only exist for the whole session and say so when a range is set: facing, camera coverage, video data quality, and the follow, both-active and one-active columns of the pair hands table. A transcript time sets the timeline's cursor rather than the range.

The interaction state labels each 10-second window individual, social, collaborative or absent with fixed thresholds on the speech, gaze, hand and distance features of that window. It is a rough estimate, and the page marks it as one.

**Refresh analysis** recomputes the report. The **Export** menu offers the raw events as JSON lines (ASR recognition and transcription, IPS translation, rotation and relation, and VFA features), the transcript as text or SRT, the fused 10-second window table (`window_features.csv`, once the video job has run) and the whole report as JSON.

## Report jobs and the worker

The report is computed once per session and kept, so opening a session does not wait on InfluxDB for data that has not changed. Two jobs fill it:

| Job | Parts | Reads | Takes |
|---|---|---|---|
| `light` | `speech`, `space` | ASR and IPS events | seconds |
| `video` | `attention`, `timeline` | every event type, VFA in 5-minute chunks (a 1 h session is about 90 MB of JSON), then the 10-second window fusion | from a few seconds to a minute or two, most of it reading the video features |

A part is opened from the cache while InfluxDB holds nothing newer than what it was computed from. For a live session a part is also served while it is younger than a minute (`light`) or five minutes (`video`), so a running session is recomputed at that pace and not on every page poll; a stale part is shown, marked stale, while its job runs again. A part that is missing starts its job and the page shows the job's progress until it is done.

Where a job runs:

- **On the Celery worker** when one consumes the dashboard's queue (`mmla-dashboard`, or `DASHBOARD_CELERY_QUEUE`). The web process asks Redis which workers consume which queues at most every 30 s; a worker on another queue, such as one started from older dashboard code on the default `celery` queue, does not count. The worker runs `DASHBOARD_WORKER_CONCURRENCY` jobs at once (2 by default).
- **Otherwise in a process of the web server**: `python dashboard.py precompute --force --job <job> <session>`, started from `flask-backend/` with the web server's Python; its errors go to the web server's log. At most three light jobs and one video job run at once; the others wait as `queued` and start as a slot frees. A process, not a thread, keeps a video job's CPU work away from the pages and the live streams, and keeps the InfluxDB clients of the gevent web process out of native threads. `DASHBOARD_JOBS=local` always runs jobs this way, `DASHBOARD_JOBS=celery` never does.

A job reports through a status file in the cache, whichever runs it. A job whose process has gone, or that has not reported for 15 minutes, counts as dead and is started again on the next request. A job that failed is not retried on every page poll: the part answers with its error until the session's data changes or five minutes pass (for a live session, whose data changes every second, after the live recompute pace: 1 minute for the light job, 5 for the video job), and **Refresh analysis** always starts it again. When an older part exists, the page keeps showing it and says that recomputing it failed. A cached part written by an older version of the report code is recomputed on its next request, so an upgrade needs no cache clearing.

The cache lives in `flask-backend/cache/` (or `DASHBOARD_CACHE_DIR`): `<session>/<part>.json`, `<session>/<job>.status.json` and `<session>/window_features.csv`. It holds transcripts and other data derived from the recordings, so it stays on the dashboard's machine (it is gitignored); deleting a session's folder makes the dashboard compute it again. The worker writes the parts into its own cache folder, so it runs on the web server's machine with the same `DASHBOARD_CACHE_DIR` (`make flask` and `make celery` both start in `flask-backend/`, where the default folder is).

To fill the cache ahead of time, for instance before a meeting:

```bash
cd pipelines/uber-server/dashboard/flask-backend
conda activate uber-server
python dashboard.py precompute --all                 # every session with InfluxDB data, both jobs
python dashboard.py precompute <session-id> --job light
python dashboard.py precompute <session-id> --force  # even when a job of it looks active
```

It runs the jobs in this process, one after the other, prints how long each took, and exits with 1 when any failed. It waits up to 5 s for MongoDB, so the space parts get the IPS camera placement.

## Live data stream

The Live page gets its data from `GET /api/sessions/<id>/stream`, a Server-Sent Events stream (plain HTTP, no client library) that sends `hello`, then `batch` events with the records in a slim form, `floor` when the room's floor becomes known, `status` every 5 s and `end`; a comment line every 15 s keeps proxies from closing it.

**Follow mode** sends the last five minutes first (`backfill`, up to 15, in one batch per minute), then each record as the pipelines write it. Every page that follows the same session shares one poller in the web process: it starts with the first follower, asks InfluxDB once a second for what each event type wrote after its newest record (six small queries side by side), slims the records once and hands each batch to all the session's followers. After a minute without new data it asks every five seconds, and it stops ten seconds after the last follower leaves, so a page reload finds it still running. Each follower still reads its own history (five minutes by default): it joins the poller first, reads its history up to where the poller has read, and then passes on the poller's batches without the records its history already held, so a page that joins late misses nothing and gets nothing twice. A page that falls more than about five minutes of batches behind (a stalled connection) is dropped with an `end` event; it reconnects and reads the gap as history. The voice keys of a diarized group microphone are shared too, so a voice has the same key on every page. The poller needs gevent, which both gunicorn's gevent worker and `python dashboard.py serve` provide; anywhere else each follower polls on its own.

**Replay mode** walks a virtual clock through the session at the chosen speed, and records are sent when the clock reaches them: a window at its start, a transcript chunk at its end, since that is when its text exists. Each replay is a stream of its own: it reads InfluxDB in pieces of 10 to 32 s of session time and keeps 30 s (at 1x) to 240 s (at 16x) read ahead of its clock, and it ends with `end` when the clock passes the session's last record.

**`GET /api/sessions/<id>/state`** answers whether a session is live (its newest record is under 20 s old), its newest event time, how far behind that is, and its span, for pages that poll every few seconds. It asks InfluxDB one small query per session at most every 2 s, and none while the session's live poller runs, since the poller already knows.

What a viewer costs:

- **InfluxDB**: six small queries a second per followed session, however many pages follow it, plus each page's own history when it connects (five minutes of it is five or six reads of six queries); for the explorer's live band, one query every 2 s per live session whatever the number of open explorers, plus the session's full metadata once a minute per explorer; the session list once every 15 s and the health check once every 10 s for everyone. A replay reads for its viewer alone, 30 s (at 1x) to 240 s (at 16x) of session time ahead of its clock. The data sent to a following page is about 4 KB per second of session time for four cameras.
- **The pipelines**: none. The dashboard reads what the synchronizers already wrote, and never talks to the bases, the session control bus or MQTT. What it shares with them is InfluxDB, where its heaviest read is the video job's (a whole session's VFA features, in 5-minute chunks). That job runs once per session, when the Analysis page first asks for its attention or timeline part. The Live page also reads the speech and timeline parts, which give voices and tags the colours they have on the Analysis page, but only from the cache (`?submit=0`), so watching never starts a job. For a live session the video job runs again when an Analysis page loads more than five minutes after its last run; at most one video job runs per session at a time, however many pages are open.
- **Report jobs while a session runs**: an open Live page starts none; it rereads the cached speech part every 2 minutes (while its tab is visible) to colour voices that appeared since. An open Analysis page asks for each part when it loads and again only until a stale part has been recomputed, so it starts no job on its own after that; the light job runs at most once a minute and the video job at most once every five minutes per session, however many pages are open.
- **Live video**: see the next section.

Behind a reverse proxy, the stream must not be buffered: the backend sends `X-Accel-Buffering: no`, which Nginx honours.

## Camera tiles and live video

Each camera tile draws the VFA overlay (skeletons, AprilTag centres, gaze rays) of the newest frame set. Live video plays under it when all of these hold:

1. The session's cameras streamed through MediaMTX and its bases pulled them from there (a `stream` source; the session's MongoDB document lists each such path as its `server_path`). A session that read files, a local camera or a raw UDP stream has no live video, and the tile says why.
2. The stream is publishing now (MediaMTX's API lists the path as ready).
3. MediaMTX has WebRTC on and the browser can reach it: TCP 8889 (`webrtc_port`) for the handshake, and UDP 8189 (or TCP 8189 when UDP is blocked) for the video.

The browser plays the stream straight from MediaMTX over WebRTC (WHEP, `POST http://<stream-server>:8889/<path>/whep`); the dashboard only tells it which paths are live. `pipelines/uber-server/mediamtx/mediamtx.yml` turns WebRTC on (`webrtc: yes`, `webrtcAddress: :8889`, `webrtcLocalUDPAddress` and `webrtcLocalTCPAddress` `:8189`, `webrtcAllowOrigins: ['*']`, since the page comes from another port).

- **Docker** (`docker/docker-compose.infra.yml`) publishes 8889 and 8189 (udp and tcp) on `INFRA_BIND_ADDRESS`. Inside the container MediaMTX only knows the container's own addresses, which no browser reaches, so name the host as browsers reach it in `docker/.env`, then recreate the container:

    ```bash
    # docker/.env
    MEDIAMTX_WEBRTC_HOSTS=100.x.y.z        # the host's tailnet IP (or LAN IP, or a name browsers resolve); comma-separated
    MEDIAMTX_WEBRTC_PORT=8889              # keep System Settings -> Stream Server -> webrtc_port in step

    docker compose -f docker/docker-compose.infra.yml up -d --force-recreate mediamtx
    ```

- **Native** (`make mediamtx`): MediaMTX announces the addresses of the host's interfaces itself; restart it (`make stop-mediamtx mediamtx`) after pulling the new `mediamtx.yml`.

A running MediaMTX keeps the configuration it started with: after any change to `mediamtx.yml` or to these variables, recreate or restart it. `curl http://<stream-server>:9997/v3/config/global/get` shows `"webrtc":true` once it took effect; until then the stream server chip's tooltip on the Sessions page says WebRTC is off.

What live video costs: MediaMTX forwards the H.264 packets it receives without transcoding, so each tile that plays is one WebRTC session carrying the camera's own bitrate from the server to that browser (four cameras at 0.8 Mbit/s are about 3.2 Mbit/s per viewer); the camera's upload to the server is the same whether nobody or ten browsers watch. A tile connects only while the Cameras card is open, the tile is on screen and the browser tab is visible, at most four tiles play at once, and a closed card or a hidden tab drops its connections, so a dashboard nobody looks at pulls no video. The bases are not affected: they pull their own copy over RTSP.

The overlay arrives later than the video, after inference and the synchronizer (one to two seconds). With **Sync overlay** on (the default), a browser that supports it holds the video back by that lag, up to 4 s, so the skeletons line up with the picture; the tile says by how much. The overlay updates once per frame set (once a second at the usual `keyframe_interval`), so it steps while the video runs smoothly. Audio is not played: the microphone paths are AAC, which WebRTC does not carry. A replay always shows the skeleton view.

The camera tiles show children. MediaMTX here has no authentication and the dashboard has no login: keep both on the lab network or the tailnet, and do not expose ports 5050, 8889 or 8189 to the internet.

### Pipelines that run without a display

The dashboard is where a session is watched, so the processes that produce it need no screen. An IPS or VFA base whose source is a stream opens no window by default (the card's **Graphics** `off for streams`; see [IPS](pipelines/ips.md#run-from-the-tui) and [VFA](pipelines/vfa/index.md)), which suits a base started over SSH on another host from its card's **Host**. Their results reach the dashboard through InfluxDB like any other: the IPS synchronizer's positions on the Room card, the VFA synchronizer's skeletons, tags and gazes on the camera tiles.

## API

All JSON, never cached by the browser; an invalid session id answers 400, an unknown one 404, and an unreachable InfluxDB 503 with the reason.

| Route | Answer |
|---|---|
| `GET /api/health` | whether InfluxDB, MongoDB, the report worker and MediaMTX answer (checked at most every 10 s, `checked_at` says when), and `stream`: how many sessions are followed live and by how many connections; never 5xx |
| `GET /api/sessions` | the session list with each session's report state (cached 15 s) |
| `GET /api/get_sessions` | the session ids alone, as the old dashboard answered |
| `GET /api/sessions/<id>` | the session's metadata: span, live state, modalities and coverage, devices, streams, provenance |
| `GET /api/sessions/<id>/state` | `{"live", "last_event", "lag", "t0", "t1"}` (epoch seconds; `lag` in seconds), at most 2 s old; `t0` and `t1` come from the cached session list (`t1` is at least `last_event`), and `t0` is `null` until the list was read |
| `GET /api/sessions/<id>/report/<part>` | `speech`, `space`, `attention` or `timeline`: 200 with the data, 202 with the job's progress, 500 with its error |
| `POST /api/sessions/<id>/report/refresh` | body `{"job": "light" \| "video" \| "all"}`: computes the job again |
| `GET /api/sessions/<id>/stream?mode=follow\|replay&at=<epoch>&speed=<1..16>&backfill=<s>` | the Server-Sent Events stream (`backfill` 0 to 900 s, default 300) |
| `GET /api/sessions/<id>/media` | where the browser plays live video, which of the session's streams are live, and the playback server's address; with `?recordings=1` also the stretches of each stream MediaMTX recorded during the session (no page plays recorded footage yet: a replay shows the skeleton view) |
| `GET /api/sessions/<id>/export/<name>` | `<event_type>.jsonl` (`asr_recognition`, `asr_transcription`, `ips_translation`, `ips_rotation`, `ips_relation` or `vfa_features`; any other type answers 404), `transcript.txt`, `transcript.srt`, `window_features.csv`, `report.json` |

## Troubleshooting

- **Port in use**: `make clean-ports 5050`.
- **Stop**: `make stop-flask`, `make stop-celery`, or the Stop buttons on the two cards.
- **Logs**: the Logs button on the cards, or attach to the tmux sessions (`tmux attach -t flask`, `tmux attach -t celery`; `Ctrl+B` then `D` detaches).
- **No sessions**: the Sessions page shows InfluxDB's error. Check that InfluxDB is reachable from the dashboard's machine and that `flask-backend/config.yml` (or `config/system_services.yml` there) carries the right URL, token, org and bucket. The TUI's Status tab shows what it can reach.
- **The analysis stays at "Computing"**: the Report worker chip says where jobs run: `ok` (the Celery worker; `tmux attach -t celery` shows the job), `local` (processes of the web server, which write their errors to the flask log), `unreachable` (`DASHBOARD_JOBS=celery` with no worker on the queue: start one, or switch to `auto`). A worker started from older code listens on another queue and takes nothing: restart it.
- **The page shows old behaviour after an update**: restart gunicorn and the worker (see [Running](#running)).
- **A part shows "stale"**: new data arrived since it was computed; it is being computed again and replaces itself.
- **The Live page of an ended session shows only its end**: an ended session opens paused near its end; press Play, or scrub to where the session has data.
- **The Live page says it fell behind and reconnects**: its connection could not take the data as fast as the session wrote it (a slow link, or a tab the browser starved) for several minutes. It reconnects on its own and reads what it missed.
- **A camera tile says "Skeleton view"**: the session has no live stream (it read files, or its streams are not publishing now), or the page is replaying; the tile names the reason. When the session should have video, check the three conditions under [Camera tiles and live video](#camera-tiles-and-live-video): the browser's developer console shows a failed `whep` request (8889 not reachable, WebRTC off) or a connection that never starts (8189 not reachable, or `MEDIAMTX_WEBRTC_HOSTS` unset under Docker).
- **The stream server chip says unreachable** while MediaMTX runs in Docker on the dashboard's own machine: with `INFRA_BIND_ADDRESS` set to one interface (a tailnet IP, say), the host's name resolves to another address on the host itself. Set `INFRA_LOOPBACK_ADDRESS` in `docker/.env` (see `docker/.env.example`), which publishes the API and playback ports there too, and recreate the container.
- **No skeletons or gaze**: the VFA synchronizer ran without Pose or Gaze, so the session has no `vfa_features`.
