# Database Reference

OpenMMLA stores data in two databases. Installation and start/stop are covered in [System Services](system_services.md); this page documents what is stored where and how to inspect, reset or migrate it.

- **InfluxDB 2.x** (time series): every measurement event produced by the pipelines and the analytics, tagged by session.
- **MongoDB** (documents): session metadata written when a session starts and ends.

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
| `asr_transcription` | ASR base | `window_start_time`, `window_end_time`, `text`, `words`, `speaker` |
| `asr_recognition` | ASR synchronizer | `window_start_time`, `window_end_time`, `speakers`, `similarities`, `durations`, `segment_start_times` |
| `ips_translation` | IPS synchronizer | `window_start_time`, `window_end_time`, `translations` |
| `ips_rotation` | IPS synchronizer | `window_start_time`, `window_end_time`, `rotations` |
| `ips_relation` | IPS synchronizer | `window_start_time`, `window_end_time`, `graph` |
| `vfa_action` | VFA synchronizer | `window_start_time`, `window_end_time`, `action_recognition` |
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
          metadata: {...}
        }
```

A document is inserted when a session is started (from the TUI or `mmla ses-ctl`) and its `end_time`/`status` are updated when it stops. Experiments and participant assignments are not stored here; they live in `config/experiments.yaml` and are edited from the TUI.

### mongosh

```bash
mongosh                                   # local instance on 27017
mongosh "mongodb://<host>:27017"          # remote instance
mongosh "mongodb://<user>:<pass>@<host>:27017/?authSource=admin"   # with authentication enabled

use openmmla
show collections
db.sessions.find().sort({start_time: -1}).pretty()
db.sessions.countDocuments({status: "active"})
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
