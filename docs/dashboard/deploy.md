# Deploy the dashboard

This page sets the dashboard up on the machine that holds the system services (`uber-server` below), keeps it running and up to date, and lists its configuration. The console's **Dashboard (Flask)** and **Dashboard (Celery)** cards run the same `make` targets as the commands shown here.

!!! note "Before you start"
    InfluxDB and Redis run, and the console points at them; MongoDB and MediaMTX are optional ([System Services](../system_services.md), and the [one-time setup](../quickstart.md#one-time-setup) of the Quickstart).

## Once per deployment

1. **Environment.** On the dashboard's machine, create the `uber-server` environment from the console's **Environment** tab, or by hand from the repository root:

    ```bash
    conda create -n uber-server python=3.10 -y
    conda activate uber-server
    pip install -e '.[uber-server]'
    ```

2. **Connections.** Under `System Settings → Connections`, fill in InfluxDB, MongoDB, Redis and the **Dashboard (Flask)** port (5050 by default), and **Save**. The console writes them into the dashboard's config ([Backend config](#backend-config)).
3. **Raw recordings.** Decide whether this dashboard serves the sessions' recordings and live microphones. They are on unless `Exports.raw_media` is `false` ([Raw recordings switch](#raw-recordings-switch)).
4. **Live video and sound**, only with MediaMTX: tell MediaMTX the address browsers reach it at ([Live video and sound in MediaMTX](#live-video-and-sound-in-mediamtx)).
5. **Start.** Press **Start** on `Launcher → System Services → Dashboard (Flask)`, then on **Dashboard (Celery)** ([Start and stop](#start-and-stop)). Open `http://uber-server:5050`.
6. **Start after a reboot**, on Linux: `make autostart` ([Start after a reboot](#start-after-a-reboot)).
7. **Fill the report cache**, optional: compute every session's report ahead of the first visit ([Precompute reports](#precompute-reports)).

## Keep the dashboard private

Read this before you start the dashboard on a machine with more than one network.

!!! warning "No login"
    The dashboard has no login, and it serves over plain HTTP. Anyone who reaches port 5050, or the media ports 5051 and 5052 (the same app), can read every session's report and transcript and, with the raw recordings on, download its footage and the participants' voices.

- **Bind addresses.** `make flask` binds gunicorn, on all three ports, where the Docker stack publishes InfluxDB, MongoDB and MediaMTX: the `INFRA_BIND_ADDRESS` and `INFRA_LOOPBACK_ADDRESS` of `docker/.env`, such as a tailnet address and a loopback address. Ports 5050 to 5052 then stay off the machine's campus LAN and Wi-Fi addresses. A machine whose `docker/.env` names no address binds every interface (`0.0.0.0`); `make flask DASHBOARD_BIND="<address> <address>"` picks the addresses by hand.
- **Check it.** Before relying on it, try the ports from another machine on the same LAN, not from the tailnet.
- **Recordings.** To keep the recordings and live microphones off a dashboard, turn the [raw recordings switch](#raw-recordings-switch) off.
- **MediaMTX.** MediaMTX has no login either: anyone who reaches its port 8889 can read a `listen/` path directly, as anyone who reaches 8554 can read any stream. The switch keeps the voices off the dashboard, not off the Stream Server.
- **Report cache.** The cache holds transcripts and other data derived from the recordings, so it stays on the dashboard's machine (it is gitignored).

## Live video and sound in MediaMTX

`pipelines/uber-server/mediamtx/mediamtx.yml` already turns WebRTC on and holds the `listen/` entry that turns a microphone into sound for the browser. What is left depends on how MediaMTX runs.

**MediaMTX in Docker** knows only the container's own addresses, which no browser reaches:

1. In `docker/.env` on the Stream Server, name the host as browsers reach it:

    ```bash
    MEDIAMTX_WEBRTC_HOSTS=<address>   # the host's tailnet or LAN address, or a name browsers resolve; comma-separated
    MEDIAMTX_WEBRTC_PORT=8889         # keep webrtc_port of System Settings → Connections → Stream Server (MediaMTX) in step
    ```

2. Recreate the container from the repository root, between sessions:

    ```bash
    docker compose -f docker/docker-compose.infra.yml up -d --force-recreate mediamtx
    ```

3. Check that WebRTC is on, that the container runs the image with FFmpeg, and that the server holds the `listen/` entry:

    ```bash
    curl -s http://uber-server:9997/v3/config/global/get | grep -o '"webrtc":[a-z]*'          # "webrtc":true
    docker inspect openmmla-infra-mediamtx-1 --format '{{.Config.Image}}'                     # bluenviron/mediamtx:1.21.0-ffmpeg
    curl -s http://uber-server:9997/v3/config/paths/list | grep -o '"name":"~^listen[^"]*"'   # "name":"~^listen/(.+)$"
    ```

**MediaMTX run natively** (`make mediamtx`) announces the addresses of the host's interfaces itself. It needs an `ffmpeg` with libopus on the PATH for the live sound; restart it with `make stop-mediamtx mediamtx` after pulling a new `mediamtx.yml`.

!!! warning "Recreate MediaMTX between sessions"
    A running MediaMTX keeps the configuration it started with, so recreate or restart it after any change to `mediamtx.yml` or to these variables. Recreating the container stops every stream for a few seconds and drops the paths START switched on, so a running session would need **Send START** again.

??? info "Details: what the MediaMTX config and the compose file set"
    - `mediamtx.yml`: `webrtc: yes`, `webrtcAddress: :8889`, `webrtcLocalUDPAddress` and `webrtcLocalTCPAddress` `:8189`, and `webrtcAllowOrigins: ['*']`, since the page comes from another port. The `listen/` entry is described in [How live sound works](live-video-and-sound.md#live-sound).
    - `docker/docker-compose.infra.yml` publishes 8889 and 8189 (UDP and TCP) on `INFRA_BIND_ADDRESS`, and runs `bluenviron/mediamtx:1.21.0-ffmpeg`, the image with the FFmpeg the `listen/` entry needs. A container on the plain image (`bluenviron/mediamtx:1.21.0`, without FFmpeg), or one started from a `mediamtx.yml` without the `listen/` entry, has no live sound; recreating it pulls the image the compose file names and reads the current file.
    - `INFRA_LOOPBACK_ADDRESS`, when `docker/.env` sets it, also publishes the MediaMTX API and playback ports on that loopback address, where the dashboard on the same machine reaches them by the machine's own name.
    - Until WebRTC is on, the **Stream server** chip's tooltip on the Sessions page says it is off.

## Start and stop

Run the commands from `pipelines/uber-server`, in a shell where `conda` is on the PATH.

| Action | In the console | From a shell |
|---|---|---|
| Start the web process | **Start** on **Dashboard (Flask)** | `make flask DASHBOARD_PORT=5050`: gunicorn with the gevent worker on 5050 and the media ports, in a tmux session named `flask` |
| Start the report worker | **Start** on **Dashboard (Celery)** | `make celery`: `celery -A dashboard.celery worker`, in a tmux session named `celery` |
| Stop | **Stop** on each card | `make stop-flask`, `make stop-celery` |
| Restart | **Stop**, then **Start** | `make stop-flask flask`, `make stop-celery celery` |
| Read the logs | **Logs** on each card | `tmux attach -t flask` or `tmux attach -t celery` (`Ctrl+B`, then `D` detaches), or `logs/flask.log` and `logs/celery.log` |

Then open `http://localhost:5050` on the server, or `http://<dashboard-host>:5050` from any device on the network. The report worker is optional: without one, the web process runs each report job in a process of its own ([Where a job runs](#where-a-job-runs)).

!!! warning "Keep one gunicorn worker"
    Keep gunicorn at one worker (`-w 1`). The report job bookkeeping and the [shared live feeds](reference.md#follow-mode) live in that process, and a second worker would keep its own.

??? info "Details: restarts and logs"
    - Both processes run through `pipelines/uber-server/restart_on_failure.sh`. It starts gunicorn or the worker again five seconds after it exits with an error, at most three times in a row (a run that lasted a minute starts the count again), and leaves it stopped after Ctrl+C, which **Stop** sends.
    - What their panes show is also written to `pipelines/uber-server/logs/flask.log` and `logs/celery.log`, which keep what a crashed process said after its tmux session is gone. A log past 10 MB becomes `flask.log.1` (or `celery.log.1`) at the next start.

??? info "Details: the development server"
    `python dashboard.py serve [--port N] [--media-ports "N M" | --media-port N]`, in `flask-backend/`, runs the same app on gevent's own server, without gunicorn. It behaves like the deployed app, shared live feeds included, and opens the media ports too: the two ports after `--port`, unless `--media-ports`, `--media-port`, `DASHBOARD_MEDIA_PORTS` or `DASHBOARD_MEDIA_PORT` names others. `0` opens none, and an empty `DASHBOARD_MEDIA_PORTS` counts as unset. A media port in use is skipped with a warning, and the page then loads the files from the other one, or from `--port`.

## Update the dashboard

A deployed dashboard runs from a checkout of its own on `uber-server`. To bring it to new dashboard code:

1. **Pull and update the environment** in that checkout:

    ```bash
    git pull
    source <conda-base>/etc/profile.d/conda.sh   # a plain SSH login may not put conda on the PATH
    conda activate uber-server
    pip install -e '.[uber-server]'
    ```

2. **Restart both processes**: **Stop** and **Start** on the two cards, or from `pipelines/uber-server`:

    ```bash
    make stop-flask flask           # add DASHBOARD_PORT=<port> when the Dashboard section sets another port
    make stop-celery celery
    ```

    A running gunicorn and a running worker keep the code they started with, while the pages are read from disk on every request.

3. **Check.** The **Report worker** chip reads `ok` once the new worker answers (within 30 s). From another machine, `curl -s http://<dashboard-host>:5052/api/media-origin` answers `"media_ports":[5051,5052]` and the same `media_instance` as port 5050.

The report cache needs no clearing: a part written by another version of the report code is recomputed on its next request.

## Start after a reboot

On Linux, from `pipelines/uber-server` in a shell where conda is on the PATH, add a line to the login user's crontab (`make no-autostart` takes it out again):

```bash
make autostart DASHBOARD_PORT=5050
```

After every reboot, cron then runs `make boot`, which starts the web process and the worker in their tmux sessions and writes what it did to `logs/boot.log`. Redis, Mosquitto and Nginx are system services and come back by themselves.

!!! note "Native MediaMTX"
    A MediaMTX run natively (Run mode `native`) does not come back by itself. Add it to the line: `make autostart DASHBOARD_PORT=5050 AUTOSTART='flask celery mediamtx'`.

??? info "Details: what a boot does and what the line keeps"
    - Boot runs in `pipelines/uber-server`, in a login shell. It waits up to five minutes (`BOOT_WAIT_SECONDS`) for the addresses the dashboard binds, since a tailnet address comes up some seconds after the network.
    - It then starts the containers of the Docker stack that failed to start because their ports are published on such an address and it was not up yet. Docker does not try those again by itself, while a container stopped on purpose stays stopped.
    - Last, it starts the targets of `AUTOSTART` as **Start** does, and leaves one that a Start opened in the meantime as it is.
    - The line keeps this checkout's path, the port, conda's folder, and any `DASHBOARD_MEDIA_PORTS`, `DASHBOARD_MEDIA_PORT`, `DASHBOARD_BIND`, `ENV_NAME` or `BOOT_WAIT_SECONDS` given to `make autostart`. Run `make autostart` again after changing one of them.
    - [Environment variables](#environment-variables) that should hold after a reboot go into `~/.profile`, which the login shell reads.
    - The tmux server that boot starts serves every tmux session opened on the machine afterwards, so the line also sets `SHELL` to the shell of the user who ran `make autostart`: those sessions get the login environment and shell, not cron's.

## Configuration

### Backend config

The backend reads `pipelines/uber-server/dashboard/flask-backend/config.yml`. The console fills its `InfluxDB`, `MongoDB` and `Redis` sections from `System Settings → Connections` when they are saved there. At startup, the same sections of `config/system_services.yml` on the dashboard's machine replace them, unless that file lists them under `SystemServicesOverride`. To edit it by hand, copy `config_template.yml` next to it and fill it in.

| Key | Default | What it does |
|---|---|---|
| `InfluxDB.url` | none | the InfluxDB the dashboard reads every measurement from, such as `http://<influxdb-host>:8086` |
| `InfluxDB.token`, `InfluxDB.org` | none | the InfluxDB token and organization |
| `InfluxDB.bucket` | `mmla-data` | the bucket of the measurements |
| `MongoDB.url` | none | the MongoDB of the session documents, such as `mongodb://<mongodb-host>:27017`; optional, and only read |
| `MongoDB.db` | `openmmla` | the database of the session documents |
| `Redis.host` | none | the Redis that carries the report jobs to the Celery worker; without it every report job runs in a process of the web server |
| `Redis.port` | `6379` | the port of that Redis |
| `Redis.db` | `0` | the Redis database of the job queue: use one other than 0, such as `1`, to keep the queue out of the default database |
| `Exports.raw_media` | `true` | serves the raw recordings and the live microphones ([Raw recordings switch](#raw-recordings-switch)) |
| `StreamServer.host`, `api_port`, `playback_port`, `webrtc_port` | not in the template | MediaMTX, used only when System Settings has no Stream Server |

The dashboard finds MediaMTX through `System Settings → Connections → Stream Server (MediaMTX)`, in the `config/system_services.yml` of its own machine (`host`, `api_port`, `playback_port`, `webrtc_port`), and reads it again every minute. Without it, the dashboard uses a `StreamServer` section of `config.yml`, then the InfluxDB host with MediaMTX's default ports.

### Raw recordings switch

`Exports.raw_media` is one switch for three things, all on by default:

- the recordings in the Analysis page's downloads, and the stream recordings MediaMTX holds ([Raw recordings](live-video-and-sound.md#raw-recordings));
- a replay's recorded video and sound on the Live page;
- a running session's microphones heard live.

To turn them off, set it to `false` (or set `DASHBOARD_RAW_MEDIA=0`) and restart the dashboard, which reads the setting only when it starts:

```yaml
Exports:
  raw_media: false
```

The Downloads card then says the recordings are turned off, the file route answers 403, and `/media` leaves out the playback server, the recorded stretches and the microphones' `listen/` paths.

??? info "Details: values that turn the recordings off"
    - `DASHBOARD_RAW_MEDIA` (`0` or `1`) overrides `Exports.raw_media`.
    - The accepted values are `true`, `false`, `yes`, `no`, `on`, `off`, `1` and `0`. Any other value, an empty `raw_media:` among them, turns the recordings off.
    - A `config.yml` or `config/system_services.yml` that cannot be parsed turns them off too, and the card says the configuration could not be read.

### Ports

| Port | Default | What it carries |
|---|---|---|
| Dashboard port | `5050` | the pages, the API and the live data stream; set under `System Settings → Connections → Dashboard (Flask)`, passed to `make flask` as `DASHBOARD_PORT`, and probed by the Status tab |
| Media ports | `5051`, `5052` | the same app, from which the Live page loads a replay's recorded video and sound ([How many recorded cameras play at once](live-video-and-sound.md#recorded-video-in-a-replay)) |
| MediaMTX WebRTC | `8889` TCP (`webrtc_port`) | the handshake of live video and sound |
| MediaMTX WebRTC media | `8189` UDP, or TCP when UDP is blocked | live video and sound |
| MediaMTX API | `9997` (`api_port`) | which streams are live, and whether WebRTC and the `listen/` entry are there |
| MediaMTX playback | `9996` (`playback_port`) | the stream recordings in the downloads, fetched by the browser |

The media ports are the two after the Dashboard port, on the same addresses: a Dashboard port of 6000 opens 6001 and 6002. The frontend needs no configuration: it talks to the backend over same-origin URLs, and the recordings route tells it the media ports.

??? info "Details: choosing the media ports"
    - `make flask DASHBOARD_MEDIA_PORTS="<port> <port>"` picks other media ports, spaces or commas between them, and `DASHBOARD_MEDIA_PORT=<port>` a single one; `DASHBOARD_MEDIA_PORTS` wins when both are set. `make flask` passes them to gunicorn, which binds them beside `DASHBOARD_PORT`.
    - `0`, an empty `DASHBOARD_MEDIA_PORTS=` or an empty `DASHBOARD_MEDIA_PORT=` opens none. The page then loads the files from the dashboard's port, fewer at once.

### Environment variables

All are optional. Set the first four and `DASHBOARD_ARTIFACTS_DIR` the same way for the web process and the worker.

| Variable | Default | What it does |
|---|---|---|
| `DASHBOARD_CACHE_DIR` | `flask-backend/cache/` | where computed report parts and job states are kept ([Report cache](#report-cache)) |
| `DASHBOARD_JOBS` | `auto` | `auto` queues report jobs for the worker only while one consumes the dashboard's queue; `celery` always queues them; `local` always runs them in processes of the web server |
| `DASHBOARD_CELERY_QUEUE` | `mmla-dashboard` | the Celery queue of the report jobs |
| `DASHBOARD_WORKER_CONCURRENCY` | `2` | how many report jobs one worker runs at once; a video job holds a whole session's video features in memory |
| `DASHBOARD_ARTIFACTS_DIR` | `artifacts/` of the repository | where the dashboard looks for the sessions' folders: their raw recordings, and the `manifest.json` whose declared pupils the attention and timeline parts use |
| `DASHBOARD_PORT` | `5050` | the port of `make flask` and of `python dashboard.py serve` |
| `DASHBOARD_MEDIA_PORTS` | the two ports after `DASHBOARD_PORT` | the media ports ([Choosing the media ports](#ports)) |
| `DASHBOARD_MEDIA_PORT` | unset | one media port instead |
| `DASHBOARD_RAW_MEDIA` | unset | `0` keeps the raw recordings and the live microphones off the dashboard and `1` offers them, whatever `Exports.raw_media` says |
| `OPENMMLA_SYSTEM_SERVICES_CONFIG` | `config/system_services.yml` of the repository | another System Settings file to merge into `config.yml` |
| `DASHBOARD_BIND` | the `INFRA_BIND_ADDRESS` and `INFRA_LOOPBACK_ADDRESS` of `docker/.env`, else `0.0.0.0` | `make flask` only: the addresses gunicorn binds, space-separated ([Keep the dashboard private](#keep-the-dashboard-private)) |
| `AUTOSTART` | `flask celery` | `make autostart` only: the targets a boot starts |
| `BOOT_WAIT_SECONDS` | `300` | `make autostart` only: how long a boot waits for the bound addresses |

## Report jobs and the worker

The report is computed once per session and kept, so opening a session does not wait on InfluxDB for data that has not changed. Two jobs fill it:

| Job | Parts | Reads | Takes |
|---|---|---|---|
| `light` | `speech`, `space` | ASR and IPS events | seconds |
| `video` | `attention`, `timeline` | every event type, VFA in 5-minute chunks (a 1 h session is about 90 MB of JSON), then the 10-second window fusion | from a few seconds to a minute or two, most of it reading the video features |

### When a part is computed

- A part is served from the cache while InfluxDB holds nothing newer than what it was computed from.
- For a live session, a part is also served while it is younger than a minute (`light`) or five minutes (`video`), so a running session is recomputed at that pace and not on every page poll. A stale part is shown, marked stale, while its job runs again.
- A missing part starts its job, and the page shows the job's progress. **Refresh analysis** always starts the job again.

??? info "Details: which pages start jobs"
    - An open Analysis page asks for each part when it loads, and again only until a stale part has been recomputed. At most one video job runs per session at a time, however many pages are open.
    - An open Live page starts no job. It reads the `speech` and `timeline` parts only from the cache (`?submit=0`), to give voices and tags the colours they have on the Analysis page, and rereads the speech part every 2 minutes while its tab is visible.

??? info "Details: failed and dead jobs"
    - A job reports through a status file in the cache, wherever it runs. A job whose process has gone, or that has not reported for 15 minutes, counts as dead and is started again on the next request.
    - A job that failed is not retried on every page poll: the part answers with its error until the session's data changes or five minutes pass. For a live session, whose data changes every second, it waits the live recompute pace instead: 1 minute for the light job, 5 for the video job.
    - When an older part exists, the page keeps showing it and says that recomputing it failed.

### Where a job runs

| Where | When | How |
|---|---|---|
| the Celery worker | a worker consumes the dashboard's queue (`mmla-dashboard`, or `DASHBOARD_CELERY_QUEUE`) | `DASHBOARD_WORKER_CONCURRENCY` jobs at once, 2 by default |
| a process of the web server | no worker consumes the queue, or `DASHBOARD_JOBS=local` | `python dashboard.py precompute --force --job <job> <session>`, started from `flask-backend/` with the web server's Python; its errors go to the web server's log. At most three light jobs and one video job run at once; the others wait as `queued` |

In `auto` mode the web process asks Redis which workers consume which queues, at most every 30 s. A worker on another queue, such as the default `celery` queue, does not count and takes no dashboard jobs.

??? info "Details: why a process and not a thread"
    A process keeps a video job's CPU work away from the pages and the live streams, and keeps the InfluxDB clients of the gevent web process out of native threads.

### Report cache

The cache lives in `flask-backend/cache/`, or in `DASHBOARD_CACHE_DIR`: `<session>/<part>.json`, `<session>/<job>.status.json` and `<session>/window_features.csv`. Deleting a session's folder makes the dashboard compute it again. The worker writes the parts into its own cache folder, so run it on the web server's machine with the same `DASHBOARD_CACHE_DIR`; `make flask` and `make celery` both start in `flask-backend/`, where the default folder is.

??? info "Details: cache.location and Delete Session"
    Every process that loads `dashboard.py` (the web process, the worker, a `python dashboard.py precompute` run) adds the cache folder it uses to `flask-backend/cache.location` (gitignored; one path a line, the latest 16; nothing is lost when the file cannot be written). [Delete Session](../tui/sessions.md#delete-session) on the console's **Sessions** tab deletes a session's cache at the default place and in every folder listed there, so a run with another `DASHBOARD_CACHE_DIR` never hides the one the dashboard uses.

### Precompute reports

To fill the cache ahead of time, for instance before a meeting:

```bash
cd pipelines/uber-server/dashboard/flask-backend
conda activate uber-server
python dashboard.py precompute --all                 # every session with InfluxDB data, both jobs
python dashboard.py precompute <session-id> --job light
python dashboard.py precompute <session-id> --force  # even when a job of it looks active
```

It runs the jobs in this process, one after the other, prints how long each took, and exits with 1 when any failed. It waits up to 5 s for MongoDB, so the space parts get the IPS camera placement. The video jobs run faster on `uber-server` itself than over the network; run `--all` in tmux, and not beside another job that uses every core.

## Troubleshooting

**A port is in use.** gunicorn does not start while any of its ports is taken by another program. `make clean-ports 5050 5051 5052` frees them. To keep that program, give the media ports other numbers, a single one or none ([Choosing the media ports](#ports)).

**The dashboard stopped and its pane says `giving up`.** gunicorn or the worker failed three times in a row, each time within a minute, so `restart_on_failure.sh` stopped starting it. The lines above say why: a port in use, a missing package, a broken `config.yml`. The tmux session stays open with the reason, so the Celery card, which goes by its tmux session, still reads running: **Stop** and **Start** it once the cause is fixed.

**`the dashboard needs ..., which the Python environment at ... lacks`.** The web process, the worker and `python dashboard.py` check at startup that the environment has every package of the `uber-server` extra they import (Flask, Celery with Redis, gevent, influxdb-client, pymongo, PyYAML, cryptography), and stop with the names of the missing ones. Run `pip install -e '.[uber-server]'` in that environment from the repository root, and start it again.

**No sessions.** The Sessions page shows InfluxDB's error. Check that InfluxDB is reachable from the dashboard's machine and that `flask-backend/config.yml` (or `config/system_services.yml` there) carries the right URL, token, org and bucket. The console's Status tab shows what it can reach.

**The analysis stays at "Computing".** The **Report worker** chip says where jobs run. `ok`: the Celery worker runs them, and `tmux attach -t celery` shows the job. `local`: processes of the web server run them and write their errors to the flask log. `unreachable`: `DASHBOARD_JOBS=celery` and no worker is on the queue; start one, or switch to `auto`. A worker on another queue takes nothing: restart it with `make stop-celery celery`.

**A part shows "stale".** New data arrived since it was computed. It is being computed again and replaces itself.

**The page shows old behaviour after an update.** Restart gunicorn and the worker ([Start and stop](#start-and-stop)).

**The Stream server chip says unreachable** while MediaMTX runs in Docker on the dashboard's own machine. With `INFRA_BIND_ADDRESS` set to one interface, such as a tailnet address, the host's name resolves to another address on the host itself. Set `INFRA_LOOPBACK_ADDRESS` in `docker/.env` (see `docker/.env.example`), which publishes the API and playback ports there too, and recreate the container.
