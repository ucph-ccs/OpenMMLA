# System Services

`Launcher → System Services` has one card per system service: it starts, stops and probes the service on its host. Installing the services is described in [System services](../../system_services.md); the Launcher card's screenshot is on the [Launcher tab](index.md) page.

## Cards

| Card | Address in System Settings | Start and Stop | Run mode |
|---|---|---|---|
| InfluxDB | `InfluxDB.url` | compose stack, or `make influxdb` | yes |
| MongoDB | `MongoDB.url` | compose stack, or `make mongodb` | yes |
| Redis | `Redis.host` | `make redis`, `make stop-redis` | no |
| MQTT (Mosquitto) | `MQTT.host` | `make mosquitto`, `make stop-mosquitto` | no |
| Gateway (Nginx) | `Gateway.host` | `make nginx`, `make stop-nginx` | no |
| Stream Server (MediaMTX) | `StreamServer.host` | compose stack, or `make mediamtx` | yes |
| Dashboard (Flask) | `Dashboard.host` | `make flask DASHBOARD_PORT=<port>`, in a tmux session named `flask` | no |
| Dashboard (Celery) | `Dashboard.host` | the worker, in a tmux session named `celery` | no |

The `make` targets run as `make -C pipelines/uber-server <target>` on the card's host.

A card opens on the machine its address names ([How each card picks its host](../index.md#how-each-card-picks-its-host)); pick another host to start or stop the service there once. Nginx and the dashboard need the `uber-server` conda environment on the host, since the Makefile renders the Nginx config and runs gunicorn and Celery in it; the other cards say `env: none needed`.

## Card controls

| Control | Cards | What it does |
|---|---|---|
| **Start**, **Stop** | all | start or stop the service on the card's host |
| **Logs** | all | compose logs in `docker` mode; Homebrew or journald logs for a native service; the tmux pane for the dashboard and its worker |
| **Refresh** | all | probes the service again |
| **Run mode** | InfluxDB, MongoDB, Stream Server | `docker` (default): `docker compose -f docker/docker-compose.infra.yml up -d`, `stop` or `logs <service>`; `native`: the `make` targets |
| **Fetch Token** | InfluxDB, `docker` mode | reads the admin token of the compose stack on the selected host, from the running container or from `docker/.env`, and stores it encrypted as `InfluxDB.token`; shows only its first and last four characters |

Choose `native` on a machine that runs Homebrew or systemd databases; otherwise Start brings up a container next to them. The choice is remembered per host while the console runs.

??? info "Details: native MediaMTX and the dashboard's logs"
    - A native MediaMTX runs as `make mediamtx`, a tmux session named `mediamtx` around the installed binary.
    - The dashboard's output is also kept in `pipelines/uber-server/logs/flask.log` and `celery.log` on its host.

??? info "Details: what Start and Stop run"
    - Redis, Mosquitto and Nginx: the `make` target installs the package first when the host has none (Homebrew on macOS, apt on Debian and Ubuntu; `make` runs on the card's host, so it sees that host's platform), and rewrites the service's listener config to bind on all interfaces.
    - Dashboard (Flask) and Dashboard (Celery): `restart_on_failure.sh` starts the process again when it exits with an error, at most three times in a row ([Start and stop](../../dashboard/deploy.md#start-and-stop)).
    - On a remote host the targets run in the command session below the card, as on `Local`, where the shell has conda.
    - A `sudo` prompt on a remote host is answered with the SSH profile's password, which is the login password `sudo` asks for. On `Local` it is answered with the [Sudo password](system-settings.md#sudo-local-admin) of System Settings.

## Status

A card's status is a TCP probe from this machine to the address under System Settings, which is the path the pipelines take. The card's description names the probed address.

??? info "Details: how each card is probed"
    - A `localhost` address names no particular machine, so it is probed on the card's selected host (over SSH for a remote host). So is a card moved off its configured machine: it reports the host it is on, while the sidebar marker keeps following the configured address.
    - An address that still reads `<uber-server>` is never looked up. The card says which form has no host yet and, as for `localhost`, reports the port of the host it is on.
    - The Celery worker is detected by its tmux session.
    - MediaMTX counts as running only when its RTMP and its RTSP ports both answer. Another program, such as an Nginx built with the RTMP module, can hold port 1935 and would otherwise read as a MediaMTX that Stop can never find. When only one of the two answers, Refresh, Start and Stop say so in the log.

## Stream Server Config tab

The **Config** tab of the Stream Server card shows the `mediamtx.yml` of the card's host as text, comments included. Two controls on top change the text; **Save** writes it.

| Control | Key | Default | What it does |
|---|---|---|---|
| **Server-side recording** | `pathDefaults.record` | `sessions only` | `sessions only` records the paths of a running session, from START to STOP; `every path` records every published stream |
| **Keep recordings for** | `pathDefaults.recordDeleteAfter` | three days | MediaMTX deletes a segment that long after it began; `for ever` keeps everything |

!!! warning "Export before the recordings expire"
    A session's footage has to be exported ([Export](../sessions.md#export)) before **Keep recordings for** has passed.

The address and ports this console uses for MediaMTX are under `System Settings → Connections → Stream Server (MediaMTX)` ([Stream Server form](system-settings.md#stream-server-form)). **Sync to Host** and **Sync from Host** copy this one file between the card's host and the one picked; the editor reads it again after a Sync from Host.

??? info "Details: when a change takes effect"
    - A native MediaMTX reloads the file when it changes. One in `docker`, the default Run mode, reads it only when its container starts: **Stop** and **Start** the card, then **Send START** again for a session that is running, since a new start records none of its paths.
    - A change to `readTimeout` or `writeTimeout` closes every stream and every pull as it takes effect, natively too, so make it between sessions ([Server settings](../../streaming/index.md#configuration)).
    - A reload drops the recording that START switched on for each path of a running session. A few seconds after a Save or a sync, the console switches those paths on again, and the status line says which.

## Stream Server Streams tab

The **Streams** tab of the Stream Server card lists every stream of the IPS, VFA and ASR Base cards in one table, so a stream left publishing is found without opening each card. Streams are started on their own card's [Streams tab](pipelines/streams.md).

It lists the base cards' `Streams` entries, left-over captures, tmux sessions started from another console, and paths the server receives from elsewhere; what runs comes first.

| Column | What it shows |
|---|---|
| **Stream** | the stream's name, followed by a row mark where one applies |
| **Pipeline** | the card whose `Streams` name the stream (`IPS`); for a capture no entry names, the first part of its path |
| **Machine** | the SSH profile that runs its FFmpeg, or the address a path comes from when the console does not run it |
| **Capture** | what that machine says: `Running`, `Exited`, `Stopped`, `No answer`, `Offline`, `External` |
| **Stream Server** | `● live` with how long, `○ not live`, or `-` for a stream that goes elsewhere |
| **Readers** | how many pull it now |
| **Path** | its path on the Stream Server |

| Row mark | What it means |
|---|---|
| `(in no Streams)` | a capture no entry names, listed so it can be stopped; it goes once its machine has nothing left of it |
| `(not from here)` | a path the server receives from something the console does not run |
| `(name clash)` | the cards give this name to another path on the server (another target, where it goes elsewhere), or run it on more than one machine; the cards do not start it until one entry is renamed, and its target can stay ([Stream names](pipelines/streams.md#stream-names)) |

| Button | What it does |
|---|---|
| **Refresh** | reads the table again |
| **Stop** | stops the selected capture as its card does: Ctrl-C to FFmpeg, which finishes its recording, then the tmux session is closed; the log names the recording the stream's Start noted. On a path published from elsewhere, it asks the server to close that connection, after which the publisher may connect again |
| **Stop All** | stops every capture still running or exited, on all machines at once; takes a second press; paths published from elsewhere stay |
| **Logs** | writes the last lines of the selected capture's pane into the log |
| **Manage** | the [Manage](pipelines/streams.md#recordings-and-manage) of a pipeline card, over the streams of every pipeline at once |

??? info "Details: how the table is read"
    - The rows: the `Streams` entries of the IPS, VFA and ASR Base cards, each read from the config of the host its card is on; captures the stream registry notes as started from this console that no entry names any more; `mmla-stream-<name>` tmux sessions a machine still has that neither knows of; and every path the server receives that none of those accounts for.
    - A capture two cards name (an IPS and a VFA Base pulling one camera) is one row.
    - The table is asked for when the tab is first opened and at **Refresh**: each machine once over SSH, all at the same time, and the server through its API.
    - The machines asked are every SSH profile that is not Windows, and this machine. One the host check reports offline is not asked and reads `Offline`.
    - Entries of two cards with the same name, machine and target are one capture and one row.

??? info "Details: Manage over every pipeline"
    - It lists what the streams recorded on each capture host (`record: true`), with the room left there, and offers **Delete File**, **Delete Day**, **Delete Older Than** and **Delete Expired** across all of them. Each takes a second press and never deletes the file a stream is writing.
    - How long recordings stay is each card's own setting (**Keep recordings for** in its Manage), so this one offers no keep time. **Delete Expired** deletes what is past each stream's own time, the longer one where two cards name the stream.
    - It lists what the table last read, so it opens once the table is filled.

## Stream Server Recordings tab

The **Recordings** tab lists what the Stream Server holds, path by path: the number of ten-minute segments, the first and the last, and their size on disk. It is read over a shell on the card's host, over SSH for a remote one.

- Above the table: the free space of that disk, whether the server records every path or the sessions' only, and the retention in force.
- Below the table: what it records now. A path it records that is publishing and has nothing written is shown in red.

| Button | What it does |
|---|---|
| **Delete Path** | removes every segment of the selected path |
| **Delete Older Than** | removes every segment of every path that began before the chosen age |

Both go through the server's API, so they work for a `docker` and a `native` run alike, and both need a second press. This tab is the server's inventory; a session's footage is exported from the [Sessions tab](../sessions.md#export).
