# Dashboard reference

This page describes the live data stream that feeds the Live page, what a viewer costs the services, and the dashboard's HTTP API. Use it to build on the dashboard, to debug it, or to plan for many viewers.

## Live data stream

The Live page gets its data from `GET /api/sessions/<id>/stream`, a Server-Sent Events stream: plain HTTP, no client library. A comment line every 15 s keeps proxies from closing it.

| Parameter | Default | What it does |
|---|---|---|
| `mode` | `follow` | `follow` sends records as the pipelines write them; `replay` walks a clock through the session |
| `at` | the session's start | the epoch second a replay starts at |
| `speed` | `1` | the replay speed, 1 to 16 |
| `backfill` | `300` | the seconds of history sent first, 0 to 900: before the newest record for a follower, before `at` for a replay |

| Event | When | What it carries |
|---|---|---|
| `hello` | when the stream opens | the session, mode, span (`t0`, `t1`), whether it is live, the speed, the clock, the floor and the group |
| `batch` | as records are due | the clock, whether it is history (`backfill`), and the records in a slim form |
| `floor` | when the room's floor becomes known | the floor, from the badges' orientation |
| `status` | every 5 s | the clock, whether the session is live, the lag and the newest event time |
| `tick` | in a replay, when no batch went out for a second | the clock alone |
| `end` | when the stream ends | a `reason` (`replay_end` or `error`) and a message |

!!! note "Behind a reverse proxy"
    The stream must not be buffered. The backend sends `X-Accel-Buffering: no`, which Nginx honours.

### Follow mode

A follower first gets its history, the last five minutes by default (`backfill`, up to 15 minutes, one batch per minute), then each record as the pipelines write it. Every page that follows the same session shares one poller in the web process, so InfluxDB is asked once however many pages watch.

??? info "Details: the shared poller"
    - The poller starts with the first follower. It asks InfluxDB once a second for what each event type wrote after its newest record (six small queries side by side), slims the records once, and hands each batch to all the session's followers.
    - After a minute without new data it asks every five seconds. It stops ten seconds after the last follower leaves, so a page reload finds it still running.
    - Each follower reads its own history: it joins the poller first, reads its history up to where the poller has read, and then passes on the poller's batches without the records its history already held. A page that joins late misses nothing and gets nothing twice.
    - A page that falls more than about five minutes of batches behind (a stalled connection) is dropped with an `end` event. It reconnects and reads the gap as history.
    - The voice keys of a diarized group microphone are shared too, so a voice has the same key on every page.
    - The poller needs gevent, which gunicorn's gevent worker and `python dashboard.py serve` both provide. Anywhere else each follower polls on its own.

### Replay mode

A replay walks a virtual clock through the session at the chosen speed, and sends records when the clock reaches them: a window at its start, a transcript chunk at its end, since that is when its text exists. Each replay is a stream of its own, and it ends with `end` when the clock passes the session's last record.

??? info "Details: how a replay reads and keeps time"
    - A replay reads InfluxDB in pieces of 10 to 32 s of session time, and keeps 30 s (at 1x) to 240 s (at 16x) read ahead of its clock.
    - A playing replay tells the page its clock at least once a second, with a `tick` when no batch went out. The page's clock, the recorded video and the sound thus play on through stretches with few records, such as a recognition bucket every 3 s and nothing else.
    - A replay waiting for InfluxDB has a clock that stands still and sends no tick. The page holds its own clock 1.5 s after the last one, so the video and the sound stop with it instead of running ahead.

### Session state

`GET /api/sessions/<id>/state` answers whether a session is live (its newest record is under 20 s old), its newest event time, how far behind that is, and its span, for pages that poll every few seconds. It asks InfluxDB one small query per session at most every 2 s, and none while the session's live poller runs, since the poller already knows.

## What a viewer costs

The dashboard reads what the synchronizers already wrote. It never talks to the bases, the session control bus or MQTT, so the only thing it shares with the pipelines is InfluxDB.

| Resource | Cost |
|---|---|
| InfluxDB, following | six small queries a second per followed session, however many pages follow it, plus each page's own history when it connects (five minutes of it is five or six reads of six queries) |
| InfluxDB, replaying | each replay reads for its viewer alone, 30 s (at 1x) to 240 s (at 16x) of session time ahead of its clock |
| InfluxDB, Sessions page | for the **Live now** band, one query every 2 s per live session whatever the number of open pages, plus the session's full metadata once a minute per page; the session list once every 15 s and the health check once every 10 s for everyone |
| InfluxDB, report jobs | the heaviest read is the video job's: a whole session's VFA features, in 5-minute chunks, once per session, when an Analysis page first asks for its `attention` or `timeline` part ([When a part is computed](deploy.md#when-a-part-is-computed)) |
| Network to a following page | about 6 KB per second of session time for four cameras, about 2 KB of it the skeletons of the people without a counted badge |
| Live video and sound | one WebRTC session per playing tile and per listened microphone ([What live video costs](live-video-and-sound.md#live-video), [How live sound works](live-video-and-sound.md#live-sound)) |

## API

Every answer is JSON and never cached by the browser, except the exports and the recording files. An invalid session id answers 400, an unknown one 404, and an unreachable InfluxDB 503 with the reason.

| Route | Answer |
|---|---|
| `GET /api/health` | whether InfluxDB, MongoDB, the report worker and MediaMTX answer (`influx`, `mongo`, `worker`, `media`), checked at most every 10 s (`checked_at` says when), and `stream`: how many sessions are followed live and by how many connections; never 5xx |
| `GET /api/sessions` | the session list with each session's report state, cached 15 s |
| `GET /api/get_sessions` | the session ids alone |
| `GET /api/sessions/<id>` | the session's metadata ([Session metadata](#session-metadata)) |
| `GET /api/sessions/<id>/state` | `{"live", "last_event", "lag", "t0", "t1"}` in epoch seconds (`lag` in seconds), at most 2 s old. `t0` and `t1` come from the cached session list (`t1` is at least `last_event`), and `t0` is `null` until the list was read |
| `GET /api/sessions/<id>/report/<part>` | `speech`, `space`, `attention` or `timeline`: 200 with the data, 202 with the job's progress, 500 with its error |
| `POST /api/sessions/<id>/report/refresh` | body `{"job": "light" \| "video" \| "all"}`: computes the job again |
| `GET /api/sessions/<id>/stream` | the [live data stream](#live-data-stream) |
| `GET /api/sessions/<id>/media` | where the browser plays live video and sound, and which of the session's streams are live ([Media answer](#media-answer)) |
| `GET /api/media-origin` | `{"media_port", "media_ports", "media_instance"}`: the media ports the Live page loads recorded files from, in order (`[]` without one; `media_port` is the first, or `null`), and a mark of the dashboard's process, new at each start. The answer goes across origins (CORS) only to a page of the same host |
| `GET /api/sessions/<id>/recordings` | the session's raw recordings on the dashboard's machine ([Recordings answer](#recordings-answer)) |
| `GET /api/sessions/<id>/recordings/<recording>` | one listed file, with HTTP ranges (`206` for a `Range` request), its `ETag` and `Last-Modified` (`304` when unchanged), kept by the browser for a day (`Cache-Control: private, max-age=86400`). An attachment named `<session>_<recording>.<ext>`, or shown in the browser with `?inline=1`. 403 while the recordings are turned off, 404 for an id not on the list, 400 for an invalid one |
| `GET /api/sessions/<id>/export/<name>` | `<event_type>.jsonl` (`asr_recognition`, `asr_transcription`, `ips_translation`, `ips_rotation`, `ips_relation` or `vfa_features`; any other type answers 404), `transcript.txt`, `transcript.srt`, `window_features.csv` or `report.json` |

### Session metadata

`GET /api/sessions/<id>` answers the session's span, live state, modalities and coverage, devices, provenance, and:

| Field | What it holds |
|---|---|
| `streams` | the session's streams, each with its `stream` name and `rotate` |
| `video` | one item per camera, VFA cameras first: `{"key", "label", "vfa", "ips", "paths", "rotate"}` ([Which tile is which camera](live-video-and-sound.md#tile-buttons)) |
| `archive` | `{"status", "files", "bytes", "location", "verified_at"}` from the session's document, `null` without one |
| `report` | the state of the session's two report jobs, keyed `light` and `video` |

### Media answer

`GET /api/sessions/<id>/media` answers:

| Field | What it holds |
|---|---|
| `webrtc` | the base URL browsers play live video and sound from, `null` while MediaMTX has WebRTC off |
| `playback` | the base URL of MediaMTX's playback server, `null` while it is off or the raw recordings are turned off |
| `streams` | the session's streams, each `{"path", "pipeline", "base_id", "camera", "kind", "stream", "rotate", "ready", "listen"}`; `ready` says whether it publishes now, and `listen` is the path a browser hears a microphone from (`listen/<path>`), `null` for a camera |
| `listen_reason` | why the microphones have no `listen` path, `null` otherwise: WebRTC off, no `listen/` entry (or MediaMTX did not say), or the raw recordings turned off |
| `recordings` | with `?recordings=1`, the stretches of each stream MediaMTX recorded during the session; empty while the raw recordings are turned off |
| `reason` | why the session has no live video, `null` otherwise |

The Analysis page's downloads link to the recorded stretches through the playback server; a replay on the Live page plays the files of the recordings route instead.

### Recordings answer

`GET /api/sessions/<id>/recordings` answers `{"enabled", "files", "server", "archive", "reason", "media_port", "media_ports", "media_instance"}`:

| Field | What it holds |
|---|---|
| `enabled` | whether the raw recordings are on; `false` with a `reason`, empty `files`, `server` and `media_ports`, and `null` `archive`, `media_port` and `media_instance` |
| `files` | the session's files on the dashboard's machine, the Collection files first, then the archived stream cuts, each video first, by device (fields below) |
| `server` | the stream recordings MediaMTX holds that no archived cut here holds, one per unbroken stretch, each `{"path", "kind", "spans", "url"}`: `kind` is `audio` or `video`, `spans` is `[[start, seconds]]`, and `url` the playback server's `/get` of it |
| `archive` | where the session is archived, `{"status", "files", "bytes", "location", "verified_at", "here"}` (`verified_at` in epoch seconds, `here` whether this machine's session folder is the archive), `null` when it is not archived |
| `media_ports`, `media_port` | the ports the Live page loads the files from, in order (`[]` without one; `media_port` is the first, or `null`) |
| `media_instance` | the mark of the dashboard's process that `/api/media-origin` on those ports has to name |

Each item of `files`:

| Field | What it holds |
|---|---|
| `id` | the recording id of the file route |
| `source` | `collection`, or `stream` for an archived stream cut |
| `modality`, `kind` | `audio` or `video` (the two are the same) |
| `host`, `device` | the recording host and device; for a cut, `device` is the stream's name |
| `stream_path`, `camera`, `pipeline` | for a cut: its path on the Stream Server, the IPS or VFA base that took the path, and that base's pipeline |
| `format`, `size`, `duration` | the file format, its size in bytes, and its length in seconds |
| `start`, `offset` | the start in epoch seconds, and in seconds from the session start (`null` when InfluxDB and MongoDB do not know the session) |
| `scope`, `participant` | for a microphone: group or worn, and the wearer's tag |
| `channels`, `sample_rate` | for a microphone: its channels and sample rate |
| `label` | the label the page shows (`Group mic jabra-1`) |
| `url` | the file route of the file |

## Troubleshooting

**The Live page says it fell behind and reconnects.** Its connection could not take the data as fast as the session wrote it, for several minutes: a slow link, or a tab the browser starved. It reconnects on its own and reads what it missed as history.
