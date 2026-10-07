# Databases

OpenMMLA keeps its data in two databases: InfluxDB holds every measurement event of the pipelines, tagged by session, and MongoDB holds one document per session. This page is the reference of what each holds, and of how to inspect, delete, reset and back them up; [System services](system_services.md) installs and starts them.

## InfluxDB

### Layout

Every point goes into one bucket and one measurement, and two tags tell the points apart.

| Part | Name | What it is |
|---|---|---|
| Organization | `admin` | the workspace; the `org` of the InfluxDB form |
| Bucket | `mmla-data` | where the points of every session are stored; the `bucket` of the InfluxDB form |
| Measurement | `sensor_events` | the one measurement every event type is written to |
| Tag `session_id` | the session id | which session the point belongs to |
| Tag `event_type` | see [Event types](#event-types) | which output the point is |
| Fields | per event type | the values of the point |
| Time | `window_end_time` | when the point happened, in UTC |

Every event has the fields `window_start_time` and `window_end_time`, in Unix seconds. A number is stored as a float and anything else as a string; lists and maps are JSON strings. The names are defined in `openmmla/utils/constants.py`, and the pipelines, the dashboard and the console all read and write through `openmmla/utils/client/influx_client.py`.

### Event types

| `event_type` | Written by | One point per |
|---|---|---|
| `asr_transcription` | ASR base | transcribed audio chunk |
| `asr_recognition` | ASR synchronizer | time window, merged over the bases |
| `ips_translation`, `ips_rotation`, `ips_relation` | IPS synchronizer | time window, one point of each type |
| `vfa_action` | VFA synchronizer, with **Action Labels** on (`-a`) | frame set sent for action labels |
| `vfa_features` | VFA synchronizer, with **Pose** or **Gaze** on (`-pose`, `-gaze`) | frame set |
| `participant_indicators`, `participant_summary`, `group_indicators`, `group_summary` | nothing | reserved names; no session holds them, and the [dashboard](dashboard/deploy.md#report-cache) keeps its analysis report in its own cache |

### ASR event fields

`asr_transcription`:

| Field | What it holds |
|---|---|
| `text` | the transcript of the chunk |
| `words` | JSON: the words of the chunk with their times; with `voices`, each diarized word also carries its `voice` |
| `speaker` | the speaker the base names for the chunk |
| `diarization` | JSON, when the chunk came back with speaker turns: `[{start, end, speaker}]`, in seconds from `window_start_time`, the speakers `SPEAKER_00`, `SPEAKER_01` ... within the chunk. The base's `-dia` (on for a group microphone unless set) or the transcriber's `SpeechTranscriber.local.diarize` asks for it |
| `voices` | JSON `{SPEAKER_NN: {voice, similarity}}`, when the base linked the chunk's speakers into the voices of its session ([Voices across chunks](pipelines/asr/speakers-and-diarization.md#voices-across-chunks)); each turn then also carries its `voice` (1, 2, 3 ...) |
| `voice_registry` | with `voices`: the registry the voices are numbered in; a base launched again into the session numbers anew |
| `voice_embedding` | with `voices`: the kind of speaker embedding the voices were linked by |
| `participant` | the tag id of a worn microphone's wearer ([Personal microphones](pipelines/asr/speakers-and-diarization.md#personal-microphones-and-energy-attribution)) |
| `attribution` | `energy`, with `participant` |
| `levels` | JSON, with `participant`: the chunk's raw level every 100 ms and its floor, `{hop, floor_db, db}` |

A worn microphone's transcript is stored at its chunk end plus an offset fixed per base, between 1 µs and 0.1 s, so the transcripts of several bases that end together do not overwrite each other.

`asr_recognition`:

| Field | What it holds |
|---|---|
| `speakers` | JSON: the speaker of each segment of the window |
| `similarities` | JSON: each segment's speaker similarity |
| `durations` | JSON: each segment's duration |
| `segment_start_times` | JSON: each segment's start |
| `energies` | JSON, with worn microphones: the energy vote, `{tag: snr_db}` |
| `levels` | JSON, with worn microphones: every worn microphone's level over the segment every 100 ms and its floor, `{participant: {start, hop, floor_db, db}}` |

### IPS event fields

| Event | Field | What it holds |
|---|---|---|
| `ips_translation` | `translations` | JSON `{tag: [[x], [y], [z]]}`: metres in the main camera's frame, the cameras' raw detections of the window fused |
| | `detections` | JSON `{tag: {camera: {t, f, d, m, dt, n, out}}}`: every camera's own detection ([What is stored](pipelines/ips/configuration.md#what-is-stored)) |
| `ips_rotation` | `rotations` | JSON `{tag: 3x3}`: the rotation of the tag's best raw detection in the window, in the main camera's frame |
| `ips_relation` | `graph` | JSON `{tag: [tags it faces]}` in the window, from raw poses; every tag of the window is a key |

### VFA event fields

A VFA event's `window_start_time` and `window_end_time` are both the start of its frame set's time bucket.

| Event | Field | What it holds |
|---|---|---|
| `vfa_action` | `action_recognition` | JSON: the frame analyzer's answer for the frame set ([Action labels](pipelines/vfa/action-labels.md)) |
| `vfa_features` | `features` | JSON: the frames of one frame set as the [features endpoint](pipelines/vfa/pose-and-gaze.md#features-endpoint) answered them: per angle, the persons with their tag, skeleton, head yaw and gaze, the tags, the zones and the pairs |
| | `pose_model` | the pose model the features came from |
| | `gaze` | `1` when the gaze model ran, else `0` |

### InfluxDB CLI

The `influx` CLI ships with `influxdb2-cli` (apt) or `influxdb-cli` (brew). In the Docker stack, run it inside the container: `docker compose -f docker/docker-compose.infra.yml exec influxdb influx ...`.

```bash
# create a connection profile and make it active
influx config create --active -n openmmla -u http://localhost:8086 -t <API-TOKEN> -o admin
cat ~/.influxdbv2/configs

# organizations, users and tokens
influx org create -n <org-name>
influx user create -n <user-name> -p <password> -o <org-name>
influx auth create -u <user-name> --all-access -o <org-name>   # all access within one org
influx auth create -u <user-name> --operator                   # all access to every org
influx auth list                                               # recover an existing token

# count the events of one session
influx query 'from(bucket:"mmla-data") |> range(start:-30d) |> filter(fn:(r) => r._measurement == "sensor_events" and r.session_id == "<session-id>") |> count()'
```

### Reset InfluxDB

This deletes every bucket and user of a native install; afterwards, open `http://localhost:8086` and run the initial setup again.

=== "macOS"

    ```bash
    brew services stop influxdb
    rm -rf ~/.influxdbv2
    brew services start influxdb
    ```

=== "Linux"

    ```bash
    sudo systemctl stop influxdb
    sudo rm -rf /var/lib/influxdb/ /etc/influxdb/
    sudo systemctl start influxdb
    ```

For the Docker stack, do not delete `influxd.bolt` by hand; see [Known caveats](docker.md#known-caveats).

??? info "Details: data in one bucket per session"
    Data written as one bucket per session, with a measurement per event (`speaker_transcription`, `badge_translation`, ...), is moved into `mmla-data` with the tags above by `scripts/migrate_influxdb.py`. The config it is given only needs an `InfluxDB` section with `bucket: mmla-data`.

    ```bash
    python scripts/migrate_influxdb.py -c pipelines/asr-base/config.yml --dry-run   # list what would move
    python scripts/migrate_influxdb.py -c pipelines/asr-base/config.yml             # migrate
    ```

## MongoDB

### The session document

The `openmmla` database has one collection, `sessions`, with one document per session and a unique index on `session_id`. The console's cards, `mmla ses-ctl`, `mmla ses-man` and the session menu of a base started by hand insert the document when a session is created; STOP, **End Session** and the Collection card's **Stop** set its `end_time` and `status`.

| Field | Written by | What it holds |
|---|---|---|
| `session_id` | creation | the session id, `<experiment>_<group>_<YYMMDDTHHMMZ>` |
| `experiment_id`, `group_id` | creation | the experiment and group the session was created in |
| `participants` | creation | the group's participants from `config/experiments.yaml`: `[{participant_id, tag_id, description}]`, where `participant_id` is the participant's name |
| `start_time`, `end_time` | creation, end | UTC; `end_time` is null until the session ends |
| `status` | creation, end | `active`, then `ended` |
| `metadata` | creation | `{created_by}` for a document the console made (`tui_collection`, `tui_<pipeline>` ...), with `registered_again: true` when a launch registered again a session id MongoDB did not have; `{}` otherwise |
| [`sources`](#sources) | bases | the streams and devices the bases took |
| [`components`](#components) | bases and synchronizers | what each component ran with |
| [`collection_hosts`, `collection_recorders`](#collection-recorders) | Collection card | where the session was recorded |
| [`recording_windows`](#recording-windows-and-archive) | START and STOP | when the Stream Server recorded the session's streams |
| [`archive`](#recording-windows-and-archive) | **Archive** | where the session is archived, and how completely |

Experiments and participant assignments are not stored here: they live in `config/experiments.yaml` and are edited from the console.

### Sources

Every IPS, VFA and ASR base adds its entry as soon as it knows its session, and sets its `left_at` once, on its way out. Synchronizers write none.

| Key | What it holds |
|---|---|
| `key` | `<pipeline>:<base id>@<stream>`, such as `ips:0@ips-cam-1`, or `ips:0` for a base that takes no stream; a base on another stream gets a second entry |
| `pipeline` | `ips`, `vfa` or `asr` |
| `base_id` | the id of its `Bases` entry, as text |
| `source`, `source_index` | the `Bases` entry's source (`stream`, `opencv`, `udp`, `file` ...) and `source_index` |
| `stream` | the `Streams` entry it takes, or null |
| `url` | what it pulls (`rtsp://...`, `srt://...`, `udp://...`), or null |
| `server_path` | that stream's path on the Stream Server (`ips/cam-1`), or null |
| `capture` | `{ssh_profile, record, record_root, kind}`: where the stream is captured, whether it records there, where, and `audio` or `video`; null for a base that takes no stream |
| `host` | the machine the base runs on |
| `joined_at`, `left_at` | UTC; `left_at` is null while the base is in |

**Sessions → Export** reads `sources` to fetch both copies of every stream into `artifacts/<session>/streams/` ([Export a session's part](streaming/recording.md#a-sessions-part-sessions-export)).

??? info "Details: how sources are written and read"
    - A base knows its session from the id the console launched it with, or from the session picked in its menu.
    - A base that joins the same session again gets its entry back, open again, with its first `joined_at`.
    - `stream` and `url` are what the base resolved (an ASR base picks its stream from a menu when its `source_index` is empty), otherwise what its `Bases` entry names.
    - `capture.record` is what the config of the machine the base runs on says, which need not be the config the stream was started with.
    - Writing never stops a base: when MongoDB is down, or the session is not in it, the base logs a warning and runs on. The helpers are in `openmmla/utils/session_sources.py`.
    - Export takes the Stream Server's copy by each entry's `url` (its path on the Stream Server of System Settings, the host checked, so a stream published to another server is named in the log and skipped) into `streams/server/<app>/<name>_<start>.mp4`.
    - It takes the capture host's copy by `capture`, cut on its capture host, into `streams/capture/<host label>/<video|audio>/`. The host is asked whatever `record` the bases noted, since they noted their own machine's config.
    - A session that was never ended runs until the last `left_at`, once every base has left. A document without `sources` has nothing to export, and the log says so.

### Components

Each base and synchronizer adds its entry as soon as it knows its session, one per `key`; a component started again in the same session replaces its own. The measurements in InfluxDB carry only the session id, so this is the record to set a run up again from, or to compare two sessions against.

| Key | What it holds |
|---|---|
| `key` | `<pipeline>:<role>[:<id>]`, such as `asr:base:1`, `ips:synchronizer`, or `asr:synchronizer:Jabra` (the base type it merges) |
| `pipeline`, `role`, `id` | `asr`, `ips` or `vfa`; `base` or `synchronizer`; the id, as text |
| `host`, `pid`, `started_at` | where and when it started (UTC) |
| `software` | `{openmmla, python, platform, git_commit}` |
| `arguments` | the flags it was started with (`-m`, `-vad`, `-lang`, `-b`, `-mc` ...) |
| `parameters` | what it resolved and runs with: thresholds and durations, the camera and its intrinsics, the stream it takes, the speaker profiles it recognizes, the main camera, the service URLs |
| `files` | what it read besides the config: the IPS transformation matrices (inline), the folder of the speaker profile snapshot |
| `services` | `{<name>: {url, ...}}`: what its servers answered on `GET /<endpoint>/info`; `{url, error}` for one without `/info` or one that does not answer; null until asked |
| `config` | the pipeline config it loaded, with its secrets masked |
| `config_path`, `config_sha256` | the file and its digest, to tell two runs' configs apart |

??? info "Details: how components are written"
    - The servers are asked right after the entry is written, in a thread that does not hold the component up. The ASR services answer with their backend, model and language; the frame analyzer with its models, prompt profile and action schema, and the pose model, weights and thresholds of its features endpoint.
    - The same entry is written on the component's machine as `artifacts/<session>/pipelines/<pipeline>/<host>/config/<role>[_<id>].json`, next to the `config.yml` it copied there and, for IPS, the `transformation_matrices_<main>.json` it loaded. **Sessions → Export** brings these over, and writes the document's part out as `measurements/<session>_parameters.json`.
    - Writing never stops a component. The helpers are in `openmmla/utils/session_provenance.py`.

### Collection recorders

The console writes these when the Collection card starts recording into the session. A second Start adds to them. When the document is not there yet, or MongoDB does not answer, the log says so and Start goes on.

| Field | Key | What it holds |
|---|---|---|
| `collection_hosts` | | the hosts a Collection Start recorded on: SSH profile names, and the console's short name for its own recorders |
| `collection_recorders` | `host`, `host_label` | one entry per recorder: its host, and the folder under `collection/` its files are filed in |
| | `folder` | where they are written on that host (`~/artifacts/<session>/collection/<label>`) |
| | `kind`, `device_label`, `device` | `audio` or `video`, its Device Label (`jabra-1`), and the device it opens |

**Sessions → Export** reads them to know which hosts to ask for the session's recordings. A session without them goes by the recordings its manifest names, else asks every SSH profile.

### Recording windows and archive

| Field | Key | What it holds |
|---|---|---|
| `recording_windows` | `paths` | one entry per START: the Stream Server paths recorded for the session; a second START adds its new paths to the open window |
| | `start`, `end` | UTC; `end` is set at STOP, and is null while the window is open |
| `archive` | `location` | `<archive host>:<path>` of the session's archive |
| | `status` | `complete`, or `partial` while something is missing |
| | `files`, `bytes` | how many files the archive holds, and their size |
| | `verified_at` | UTC, when the files were last checked by their sha256 |

### What a session ran with

Each level of a session's data has its own record of how it was made:

| Layer | The data | The record of how it was made |
|---|---|---|
| Raw | the streams in `artifacts/<session>/streams/` and the Collection recordings | the session's `sources`, and the `Streams` entries in every component's `config`; a Collection recording's device, channels, sample rate and format in its `manifest`; the Stream Server's `mediamtx.yml` (`recordFormat`, `recordSegmentDuration`) |
| Measurements | the InfluxDB events, exported to `artifacts/<session>/measurements/` | `components`, exported as `measurements/<session>_parameters.json` |
| Analysis | the plots and files under `artifacts/<session>/analysis/` | `analysis/parameters.json`: the measurement files read (digests, record counts), the steps run, the software |

The FFmpeg command line of a stream, the capture host's FFmpeg version and the device's real mode are not recorded; the `Streams` entry and the console's openmmla version determine the command.

### mongosh

```bash
mongosh                                   # local instance on 27017
mongosh "mongodb://<host>:27017"          # remote instance
mongosh "mongodb://<user>:<pass>@<host>:27017/?authSource=admin"   # with authentication enabled

use openmmla
show collections
db.sessions.find().sort({start_time: -1}).pretty()
db.sessions.countDocuments({status: "active"})
db.sessions.find({session_id: "<session-id>"}, {_id: 0, sources: 1})      # the streams its bases took
db.sessions.find({session_id: "<session-id>"}, {_id: 0, components: 1})   # what each component ran with
db.sessions.find({}, {_id: 0, session_id: 1, "components.key": 1, "components.services": 1})   # models per session
```

### Reset MongoDB

This deletes every database of a native install.

=== "macOS"

    ```bash
    brew services stop mongodb-community
    rm -rf /opt/homebrew/var/mongodb/*
    brew services start mongodb-community
    ```

=== "Linux"

    ```bash
    sudo systemctl stop mongod
    sudo rm -rf /var/lib/mongodb/*
    sudo systemctl start mongod
    ```

## Deleting a session

**Delete Session** on the Sessions tab removes one session everywhere central; `mmla ses-delete <session> --yes` and **Delete Session** in `mmla ses-man`, which asks y/N first, do the same. Without `--yes`, `mmla ses-delete` says what would go (the archive's size, the events by type, whether the document exists) and deletes nothing.

The steps run in this order, each only once the one before it succeeded:

1. The archive on the archive host (`artifacts/<session>/` there), and the dashboard's cached report of the session on its host: in `flask-backend/cache/` and in each cache folder `flask-backend/cache.location` lists there.
2. Every InfluxDB point tagged with its `session_id`, of every event type and over all time. InfluxDB is counted again after the delete, and events that are still there stop the next step.
3. Its document in MongoDB's `sessions` collection.

A session whose archive could not be deleted is therefore still listed. Copies on other machines, the console's own `artifacts/<session>/` among them, are left alone; **Delete Files** on the [Sessions tab](tui/sessions.md#delete-files) removes those, one host at a time.

By hand, the two database steps for one session are:

```bash
influx delete --bucket mmla-data --org admin --start 1970-01-01T00:00:00Z --stop $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --predicate 'session_id="<session-id>"'
mongosh openmmla --eval 'db.sessions.deleteOne({session_id: "<session-id>"})'
```

!!! warning "Check the predicate"
    An InfluxDB delete with an empty predicate removes every point of the bucket in that time range.

## Backups

For native installs:

```bash
influx backup ./influx-backup -t <admin-token>
mongodump --uri "mongodb://localhost:27017" --db openmmla --archive=openmmla.archive
```

Restore with `influx restore ./influx-backup --full` and `mongorestore --archive=openmmla.archive --db openmmla`. The container equivalents are in the [Docker guide](docker.md#backups).
