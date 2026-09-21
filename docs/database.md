# Database Reference

OpenMMLA stores data in two databases. Installation and start/stop are covered in [System Services](system_services.md); this page documents what is stored where and how to inspect, reset or migrate it.

- **InfluxDB 2.x** (time series): every measurement event produced by the pipelines and the analytics, tagged by session.
- **MongoDB** (documents): session metadata written when a session starts and ends, and the streams its bases took.

## InfluxDB

### Schema

One bucket, one measurement. Every point is written to the `sensor_events` measurement and told apart by two tags, `session_id` and `event_type`.

```
Organization (default: admin)
└── Bucket (default: mmla-data)
    └── Measurement: sensor_events
        ├── Tags:   session_id, event_type
        ├── Fields: depend on event_type (below)
        └── Timestamp
```

| `event_type` | Written by | Fields |
|---|---|---|
| `asr_transcription` | ASR base | `window_start_time`, `window_end_time`, `text`, `words`, `speaker`, and `diarization` when the chunk came back with speaker turns (the base's `-dia`, or the transcriber's own `SpeechTranscriber.local.diarize`): `[{start, end, speaker}]` in seconds from `window_start_time`, the speakers `SPEAKER_00`, `SPEAKER_01` ... within the chunk |
| `asr_recognition` | ASR synchronizer | `window_start_time`, `window_end_time`, `speakers`, `similarities`, `durations`, `segment_start_times` |
| `ips_translation` | IPS synchronizer | `window_start_time`, `window_end_time`, `translations` |
| `ips_rotation` | IPS synchronizer | `window_start_time`, `window_end_time`, `rotations` |
| `ips_relation` | IPS synchronizer | `window_start_time`, `window_end_time`, `graph` |
| `vfa_action` | VFA synchronizer | `window_start_time`, `window_end_time`, `action_recognition` |
| `vfa_features` | VFA synchronizer (with Body & Gaze Features on, `-f`) | `window_start_time`, `window_end_time`, `features` (the frames of one synchronized set as the [features endpoint](pipelines/vfa/index.md#features-endpoint-skeletons-and-gazes) answered them: per angle, the persons with their tag, skeleton, head yaw and gaze, the tags, the zones and the pairs), `pose_model`, `gaze` (1 when the gaze model ran) |
| `participant_indicators`, `participant_summary`, `group_indicators`, `group_summary` | analytics (`mmla ses-ana`, Sessions tab) | derived indicators per participant and per group |

The names are defined in `openmmla/utils/constants.py`; the pipelines, the dashboard and the TUI all read and write through the shared wrapper in `openmmla/utils/client/influx_client.py`.

<details>
<summary><strong>Terms</strong></summary>

+ Organization: a workspace for a group of users. OpenMMLA uses one, `admin` by default.
+ Bucket: a named location where time series data are stored. OpenMMLA uses one bucket, `mmla-data`, for all sessions.
+ Measurement: a logical group of time series. OpenMMLA uses one measurement, `sensor_events`, for all event types.
+ Tags: indexed key-value pairs used for filtering. `session_id` identifies the recording session and `event_type` the pipeline output.
+ Fields: key-value pairs whose values change over time (transcripts, similarities, positions, ...).
+ Timestamp: the time of each point, used for sorting and range queries.
+ Point: one record, identified by its measurement, tags, fields and timestamp.

</details>

### InfluxDB CLI

The `influx` CLI ships with `influxdb2-cli` (apt) or `influxdb-cli` (brew); in the Docker stack run it inside the container with `docker compose -f docker/docker-compose.infra.yml exec influxdb influx ...`.

```bash
# create a connection profile and make it active
influx config create --active -n openmmla -u http://localhost:8086 -t <API-TOKEN> -o admin
cat ~/.influxdbv2/configs

# organisations, users and tokens
influx org create -n <org-name>
influx user create -n <user-name> -p <password> -o <org-name>
influx auth create -u <user-name> --all-access -o <org-name>   # all access within one org
influx auth create -u <user-name> --operator                   # all access to every org
influx auth list                                               # recover an existing token

# query the number of events of one session
influx query 'from(bucket:"mmla-data") |> range(start:-30d) |> filter(fn:(r) => r._measurement == "sensor_events" and r.session_id == "<session-id>") |> count()'
```

### Reset InfluxDB

This deletes every bucket and user; afterwards open `http://localhost:8086` and run the initial setup again.

<details>
<summary>macOS</summary>

```bash
brew services stop influxdb
rm -rf ~/.influxdbv2
brew services start influxdb
```

</details>

<details>
<summary>Linux</summary>

```bash
sudo systemctl stop influxdb
sudo rm -rf /var/lib/influxdb/ /etc/influxdb/
sudo systemctl start influxdb
```

</details>

For the Docker stack, do not delete `influxd.bolt` by hand; see the [known caveats](docker.md#known-caveats) in the Docker guide.

### Migrating from the old per-session buckets

Versions before the single-bucket schema wrote one bucket per session with six measurements (`speaker_transcription`, `badge_translation`, ...). `scripts/migrate_influxdb.py` reads every old bucket and rewrites it into `mmla-data` with the tags above:

```bash
python scripts/migrate_influxdb.py -c pipelines/asr-base/config.yml --dry-run   # list what would move
python scripts/migrate_influxdb.py -c pipelines/asr-base/config.yml             # migrate
```

The config only needs a valid `InfluxDB` section with `bucket: mmla-data`.

## MongoDB

### Schema

```
Database: openmmla
└── Collection: sessions           (unique index on session_id)
    └── Document {
          session_id, experiment_id, group_id,
          participants: [{...}],
          start_time, end_time, status: active | ended,
          metadata: {...},
          sources: [{                          # one per base that joined the session
            key,                               # <pipeline>:<base id>@<stream>, e.g. ips:0@ips-cam-1 (ips:0 when it
                                               # takes no stream); a base on another stream gets a second entry
            pipeline,                          # ips | vfa | asr
            base_id,                           # the id of its Bases entry, as text
            source, source_index,              # the Bases entry's source (stream, opencv, udp, file ...) and source_index
            stream,                            # the Streams entry it takes, or null
            url,                               # what it pulls (rtsp://..., srt://..., udp://...), or null
            server_path,                       # that stream's path on the Stream Server (ips/cam-1), or null
            capture: {ssh_profile, record, record_root, kind} | null,
            host,                              # the machine the base runs on
            joined_at, left_at                 # UTC; left_at is null while the base is in
          }],
          components: [{                       # one per base, synchronizer and visualizer that ran in the session
            key,                               # <pipeline>:<role>[:<id>], e.g. asr:base:1, ips:synchronizer,
                                               # asr:synchronizer:Jabra (the base type it merges)
            pipeline, role, id,                # asr | ips | vfa; base | synchronizer | visualizer; the id, as text
            host, pid, started_at,             # where and when it started (UTC)
            software: {openmmla, python, platform, git_commit},
            arguments: {...},                  # the flags it was started with (-m, -vad, -lang, -b, -mc ...)
            parameters: {...},                 # what it resolved and runs with: thresholds and durations, the
                                               # camera and its intrinsics, the stream it takes, the speaker
                                               # profiles it recognizes, the main camera, the service URLs
            files: {...},                      # what it read besides the config: the IPS transformation
                                               # matrices (inline), the speaker profile snapshot's folder
            services: {<name>: {url, ...}},    # what its servers answered on /info (the transcriber's backend,
                                               # model and language, the frame analyzer's models and prompt
                                               # profile), or {url, error} for one without /info or unreachable;
                                               # null until asked
            config: {...},                     # the pipeline config it loaded, secrets masked
            config_path, config_sha256         # the file and its digest, to tell two runs' configs apart at a glance
          }]
        }
```

A document is inserted when a session is started (from the TUI or `mmla ses-ctl`) and its `end_time`/`status` are updated when it stops. Experiments and participant assignments are not stored here; they live in `config/experiments.yaml` and are edited from the TUI.

`sources` is written by the bases, not by the console: every IPS, VFA and ASR base adds its entry as soon as it knows its session (the id the console launched it with, or the session picked in its menu) and sets its `left_at` once, on its way out. A base that joins the same session again gets its entry back, open again, with its first `joined_at`. `stream` and `url` are what the base resolved (an ASR base picks its stream from a menu when its `source_index` is empty), otherwise what its Bases entry names; `capture` is null for a base that takes no stream, and otherwise holds what the console needs to find the capture-side recording again: the Streams entry's `ssh_profile`, whether it has `record` on, its `record_root`, and `kind` (`audio` or `video`). Synchronizers and visualizers write nothing here. Writing never stops a base: when MongoDB is down, or the session is not in it, the base logs a warning and runs on. The helpers are in `openmmla/utils/session_sources.py`.

The console reads it back with **Sessions → Export Streams**, which fetches both copies of every stream the session used into `artifacts/<session>/streams/`: the Stream Server's, by the `url` of each entry (its path on the Stream Server of System Settings, the host checked, so a stream published to another server is named in the log and skipped), into `streams/server/<app>/<name>_<start>.mp4`; and the capture host's, by `capture` (the streams the console runs with `record` on, cut on their capture host), into `streams/capture/<host label>/<video|audio>/`. A session that was never ended runs until the last `left_at`, once every base has left. A document without `sources` (a session from before the bases wrote them, or one no base joined) has nothing to export, and the log says so. See [Streaming](rtmp_streaming.md#a-sessions-part-sessions-export-streams).

`components` is what the session ran with, written by every component as soon as it knows its session: each base, synchronizer and the IPS visualizer adds its entry (one per `key`; a component started again in the same session replaces its own) with the flags it was started with, the values it resolved from them and its config, the files it read that the config does not hold, the config itself with its secrets masked, and the software it runs. What its servers run is asked right after, in a thread that does not hold the component up: each server answers `GET /<endpoint>/info` with its backend, model and language (the ASR services) or its models, prompt profile and action schema, and the pose model, weights and thresholds of its features endpoint (the frame analyzer), and the answers are set into the entry as `services`; a server that runs an older openmmla answers 404 there, and one that does not answer at all, are noted with the error instead. The measurements in InfluxDB carry only the session id, so this is the record to set a run up again from, or to compare two sessions' numbers against: the IPS transformation matrices and camera intrinsics, the ASR thresholds and the transcriber's model, the VFA prompt profile and models. The same entry is written on the component's machine as `artifacts/<session>/pipelines/<pipeline>/<host>/config/<role>[_<id>].json`, next to the `config.yml` it copied there (and, for IPS, the `transformation_matrices_<main>.json` it loaded), which **Sessions → Export Base Files** brings over; **Sessions → Export Measurements** writes the document's part out as `measurements/<session>_parameters.json`. Writing never stops a component either. The helpers are in `openmmla/utils/session_provenance.py`.

Seen as layers, each level of a session's data has its own record of how it was made, and each points at the level below:

| Layer | The data | What made it, and with what |
| --- | --- | --- |
| Raw | the streams (`artifacts/<session>/streams/`, from the Stream Server and the capture hosts) and the Collection recordings | the session's `sources` (which stream each base took, where it was captured) and the `Streams` entries in every component's `config`; a Collection recording's device, channels, sample rate and format in its `manifest`; the Stream Server's `mediamtx.yml` (`recordFormat`, `recordSegmentDuration`) |
| Measurements | the InfluxDB events, exported to `artifacts/<session>/measurements/` | `components`: the flags, resolved values, files, config and server models of every base and synchronizer, exported as `measurements/<session>_parameters.json` |
| Analysis | the plots and files under `artifacts/<session>/analysis/` | `analysis/parameters.json`: the measurement files read (digests, record counts), the steps run, the software; the plots take no thresholds yet, and when they do, those go into its `parameters` |

What the raw layer still lacks is the capture command itself: the ffmpeg the Streams tab runs is built from the `Streams` entry by the console's version of openmmla, so the entry and the version pin it, but the command line, the capture host's ffmpeg version and the device's real mode are not written next to the recording yet.

### mongosh

```bash
mongosh                                   # local instance on 27017
mongosh "mongodb://<host>:27017"          # remote instance
mongosh "mongodb://<user>:<pass>@<host>:27017/?authSource=admin"   # with authentication enabled

use openmmla
show collections
db.sessions.find().sort({start_time: -1}).pretty()
db.sessions.countDocuments({status: "active"})
db.sessions.find({session_id: "<session-id>"}, {_id: 0, sources: 1})   # the streams its bases took
db.sessions.find({session_id: "<session-id>"}, {_id: 0, components: 1})   # what each component ran with
db.sessions.find({}, {_id: 0, session_id: 1, "components.key": 1, "components.services": 1})   # models per session
```

### Reset MongoDB

<details>
<summary>macOS</summary>

```bash
brew services stop mongodb-community
rm -rf /opt/homebrew/var/mongodb/*
brew services start mongodb-community
```

</details>

<details>
<summary>Linux</summary>

```bash
sudo systemctl stop mongod
sudo rm -rf /var/lib/mongodb/*
sudo systemctl start mongod
```

</details>

## Backups

Native installs:

```bash
influx backup ./influx-backup -t <admin-token>
mongodump --uri "mongodb://localhost:27017" --db openmmla --archive=openmmla.archive
```

Restore with `influx restore ./influx-backup --full` and `mongorestore --archive=openmmla.archive --db openmmla`. The container equivalents are in the [Docker guide](docker.md#backups).
