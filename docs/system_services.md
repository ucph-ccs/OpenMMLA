# System services

The system services are shared by every pipeline: the two databases, the message broker, the session-control channel, and optionally the Gateway (Nginx), the Stream Server (MediaMTX) and the dashboard. They usually run together on one machine, the uber server, but any split across machines works. This page lists them, starts them from the console or the shell, installs them by hand, and is the reference for the forms that point everything at them; the [Quickstart](quickstart.md#start-the-system-services) starts them in order.

## The services

| Service | Console card | Port | Required | Runs | Role |
|---|---|---|---|---|---|
| InfluxDB 2.x | **InfluxDB** | 8086 | yes | Docker (default) or native | every measurement event, in `sensor_events` ([Databases](database.md#influxdb)) |
| MongoDB | **MongoDB** | 27017 | yes | Docker (default) or native | the session documents ([Databases](database.md#mongodb)) |
| Redis | **Redis** | 6379 | yes | native | START and STOP between Session Control, the bases and the synchronizers; the queue of the dashboard's report worker |
| Mosquitto | **MQTT (Mosquitto)** | 1883 | yes | native | the MQTT broker that carries results from the bases to the synchronizers |
| Nginx | **Gateway (Nginx)** | 8080 | optional | native | a load balancer in front of the AI services ([Nginx](nginx.md)) |
| MediaMTX | **Stream Server (MediaMTX)** | 1935 RTMP, 8554 RTSP, 8890 SRT, 8889 WebRTC, 9996 playback, 9997 API | optional | Docker (default) or native | cameras and microphones publish to it, the bases pull from it, and it records the streams of a running session from START to STOP ([Run the Stream Server](streaming/index.md#run-the-stream-server)) |
| Dashboard | **Dashboard (Flask)**, **Dashboard (Celery)** | 5050, and the two ports after it for media | optional | native, in the `uber-server` env | the live view, the replay and the analysis report ([Dashboard](dashboard/index.md)) |

The Docker stack of InfluxDB, MongoDB and MediaMTX is `docker/docker-compose.infra.yml` ([Docker: Database stack](docker.md#database-stack-influxdb-and-mongodb)). What the two databases hold is in [Databases](database.md).

## Starting and stopping

### From the console { #from-the-tui }

`Launcher → System Services` has one card per service. Press **Start**, **Stop** or **Logs** on it ([Card controls](tui/system-services.md#card-controls)).

- Each card opens on the machine its address in System Settings names: this machine, or the SSH profile that matches it. It can be moved to another host for a one-off.
- A card's status is a TCP probe of that address, so it shows what the pipelines will see.
- The InfluxDB, MongoDB and Stream Server cards have a **Run mode**: `docker`, the default, drives the compose stack; `native` runs the make targets below.
- A native start needs `sudo`. Store this machine's password under `System Settings → Credentials → Sudo (local admin)`, and the console types it at the prompt; on a remote host it types the SSH profile's password.

??? info "Details: where a card opens"
    - Each card's name starts with the name of the Connections form that holds its address.
    - A card whose address is `localhost` remembers the host it was last pointed at.
    - So does a card whose address still reads `<uber-server>`. The placeholder is never looked up, so until the address is filled in the card reports the port of the host it is on ([A new machine](#a-new-machine)).

### From the shell

The Makefile in `pipelines/uber-server` wraps brew services, systemctl and the tmux sessions of the dashboard:

```bash
cd pipelines/uber-server
make all                                # free the default ports, then (re)start every service
make all without=nginx,flask,celery     # everything except some services
make influxdb mongodb redis mosquitto   # start (and reconfigure) individual services
make flask celery                       # dashboard backend and worker (uber-server env), started again when they fail
make autostart DASHBOARD_PORT=5050      # start the dashboard and its worker after every reboot (crontab); make no-autostart undoes it
make mediamtx                           # Stream Server (MediaMTX) in a tmux session (needs the mediamtx binary)
make stop                               # stop everything
make stop-redis                         # stop one service
make clean-ports 8086 5050 5051 5052    # kill whatever holds those ports
```

!!! warning "`make all` kills whatever holds the default ports"
    `make all` and `make clean-ports` send `kill -9` to whatever listens on the default ports, the `docker-proxy` of containerized databases included. On a machine that runs the Docker database stack, use `make all without=influxdb,mongodb`.

## Installation

InfluxDB, MongoDB and MediaMTX run in Docker by default and need no install. **Start** on the Redis, MQTT (Mosquitto) and Gateway (Nginx) cards, like their `make` targets, installs the package when the machine has none, with brew on macOS and apt on Debian and Ubuntu. The commands below install a service by hand; each project's own guide has the rest: [InfluxDB](https://docs.influxdata.com/influxdb/v2/install/), [MongoDB](https://www.mongodb.com/docs/manual/installation/), [Redis](https://redis.io/downloads/), [Mosquitto](https://mosquitto.org/download/), [Nginx](https://nginx.org/en/docs/install.html), [MediaMTX](https://github.com/bluenviron/mediamtx).

### InfluxDB

=== "macOS"

    ```bash
    brew install influxdb influxdb-cli
    brew services start influxdb
    ```

=== "Ubuntu and Debian"

    ```bash
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
    sudo apt update && sudo apt install -y influxdb2 influxdb2-cli
    sudo systemctl enable --now influxdb
    ```

Then open `http://localhost:8086` and run the initial setup: create the admin user, with `admin` as the organization and `mmla-data` as the bucket, the names the Docker stack uses. Save the operator API token; it goes into the `token` of the InfluxDB form under **System Settings → Connections**.

### MongoDB

=== "macOS"

    ```bash
    brew tap mongodb/brew
    brew install mongodb-community
    brew services start mongodb-community
    ```

=== "Ubuntu and Debian"

    ```bash
    curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    sudo apt update && sudo apt install -y mongodb-org
    sudo systemctl enable --now mongod
    ```

MongoDB needs no initial setup: the `openmmla` database and its `sessions` collection are created on first use.

!!! warning "No authentication"
    MongoDB runs without authentication in this setup, so keep it on a trusted network. The Docker stack can turn it on ([MongoDB authentication](docker.md#mongodb-authentication-optional-strongly-recommended)).

### Redis

=== "macOS"

    ```bash
    brew install redis
    brew services start redis
    ```

=== "Ubuntu and Debian"

    ```bash
    sudo apt install -y redis-server
    sudo systemctl enable --now redis-server
    ```

### Mosquitto

=== "macOS"

    ```bash
    brew install mosquitto
    brew services start mosquitto
    ```

=== "Ubuntu and Debian"

    ```bash
    sudo apt install -y mosquitto mosquitto-clients
    sudo systemctl enable --now mosquitto
    ```

### Nginx (optional)

=== "macOS"

    ```bash
    brew install nginx
    ```

=== "Ubuntu and Debian"

    ```bash
    sudo apt install -y nginx
    ```

The upstreams are rendered from `pipelines/uber-server/nginx/config.yml` ([Nginx](nginx.md)). That file is gitignored, so a fresh clone has none, and **Start** on the Gateway card stops with a note until the card's **Config** tab saves one on that machine.

??? info "Details: copying this machine's Gateway config"
    Instead of saving a new one, copy this machine's `config.yml` there: **Sync from Host** with `Local` picked on that host's **Config** tab, or **Sync to Host** with **Host** on `Local`.

### MediaMTX (optional)

=== "macOS"

    ```bash
    brew install mediamtx ffmpeg
    ```

=== "Linux"

    ```bash
    # unpack the release for your architecture from
    # https://github.com/bluenviron/mediamtx/releases into /usr/local/bin
    sudo apt install -y ffmpeg     # Debian and Ubuntu
    ```

A native MediaMTX needs an FFmpeg with libopus on the `PATH`, which it runs to make the dashboard's live sound (the `listen/` paths of `mediamtx.yml`); Homebrew's, Debian's and Ubuntu's `ffmpeg` have it. The Docker image (`-ffmpeg`) carries its own. MediaMTX reads `pipelines/uber-server/mediamtx/mediamtx.yml` ([Streaming](streaming/index.md)).

### Dashboard (optional)

The dashboard is Python code from this repository, not a package. It needs the `uber-server` env (the Environment tab, or `pip install -e '.[uber-server]'`), InfluxDB, and Redis for the queue of its report worker. MongoDB and the Stream Server add to what it shows when they answer ([What you need](dashboard/index.md#what-you-need)).

## Listener configuration

Out of the box Redis, Mosquitto and MongoDB listen on loopback only, and every base station has to reach them. `make <service>` in `pipelines/uber-server`, which the console's **Start** runs, rewrites the config before restarting the service:

| Service | What the make target sets |
|---|---|
| Redis | `bind 0.0.0.0 ::` and `protected-mode no` in `redis.conf` |
| Mosquitto | `listener 1883 0.0.0.0` and `allow_anonymous true` in a `conf.d/openmmla-listener.conf` drop-in |
| MongoDB | `net.bindIp: 0.0.0.0` in `mongod.conf` |
| InfluxDB | `http-bind-address = ":8086"` in `config.toml`, which is InfluxDB's default already |

The original files are backed up as `*.openmmla.bak`. Two variables change what is set:

| Variable | Default | What it does |
|---|---|---|
| `OPENMMLA_BIND_ADDRESS` | `0.0.0.0` | the address the services bind to; a LAN IP binds one interface only |
| `OPENMMLA_MQTT_ALLOW_ANONYMOUS` | `true` | `false` when you configure MQTT credentials yourself |

!!! warning
    These settings assume a trusted lab network. Nothing here adds authentication.

## Pointing the pipelines at the services

Open `Launcher → System Settings → Connections` with the **Host** selector on the machine whose settings they are (`Local` for this one). **Save** on a form writes its section into that machine's `config/system_services.yml`, next to the sections already there, and copies it into those of `pipelines/asr-base/config.yml`, `pipelines/vfa-base/config.yml`, `pipelines/ips-base/config.yml` and the dashboard backend config that exist.

The file is gitignored, like the pipeline configs, because it names your machines and holds the token and the sudo password. `config/system_services_template.yml` shows its layout.

| Section | Key | Default | What it does |
|---|---|---|---|
| `InfluxDB` | `url` | `http://<uber-server>:8086` | the InfluxDB server, by the name every machine reaches it by |
| | `token` | empty | the operator API token; **Fetch Token** on the InfluxDB card fills it from the Docker stack |
| | `org` | empty | the organization: type `admin`, the one the Docker stack creates, or the one you made |
| | `bucket` | `mmla-data` | the bucket every event goes into |
| `MongoDB` | `url` | `mongodb://<uber-server>:27017` | the MongoDB server; with authentication, `mongodb://<user>:<pass>@<host>:27017/?authSource=admin` |
| | `db` | `openmmla` | the database of the `sessions` collection |
| `MQTT` | `host` | `<uber-server>` | the Mosquitto broker |
| | `port` | `1883` | |
| `Redis` | `host` | `<uber-server>` | the Redis server |
| | `port` | `6379` | |
| | `db` | `0` | type `1`, which keeps the dashboard's Celery queue out of the default database; START and STOP go over pub/sub, which no db number separates |
| `Dashboard` | `host` | `<uber-server>` | where the Flask backend runs; the console places and probes the Dashboard cards with it |
| | `port` | `5050` | passed to `make flask`, which also binds the next two ports for media |
| `Gateway` | `host` | `<uber-server>` | the Gateway (Nginx), the load balancer the bases reach the AI services through |
| | `http_port` | `8080` | |
| | `scheme` | `http` | `http` or `https` |
| `StreamServer` | `host` | `<uber-server>` | the Stream Server (MediaMTX); it may be another machine than the Gateway |
| | `rtmp_port` | `1935` | what cameras and microphones publish to |
| | `rtsp_port` | `8554` | what the bases pull from |
| | `api_port` | `9997` | the control API: which streams are live (the Streams tab, a base card's Start), what was recorded, what to delete |
| | `playback_port` | `9996` | hands out the recording of a time range to **Sessions → Export** |
| | `webrtc_port` | `8889` | what the [dashboard](dashboard/live-video-and-sound.md#live-video)'s camera tiles play live video from, and its **Sound** control a microphone's `listen/` path |
| `Sudo` | `password` | empty | this machine's sudo password for native service starts; under **Credentials**, never copied into a pipeline config or to another machine, and the one form without a **Host** selector |

??? info "Details: how the settings reach the components"
    - When a component starts, the `InfluxDB`, `MongoDB`, `MQTT` and `Redis` sections of the `config/system_services.yml` found above its `config.yml` are laid over those of the `config.yml`, field by field. A field the file does not have keeps the pipeline's value. `OPENMMLA_SYSTEM_SERVICES_CONFIG` names the file instead, when set.
    - A section the pipeline lists under `SystemServicesOverride:` in its `config.yml` stays the pipeline's own, at startup and when the console syncs. So does a section of the file that still holds a `<...>` placeholder.
    - On the console machine the file and the pipeline configs always agree. On a remote host the file exists once a console has run there, or its Connections forms were saved or synced from another console. Pick that host in the [System Settings](tui/system-settings.md#whose-settings-a-form-shows) forms to see and edit what it will use.
    - Do not edit the synced sections in the pipeline configs by hand: the console overwrites them, and `config/system_services.yml` is what counts at runtime.
    - `Dashboard` is carried by no pipeline config. **Sync to Host** writes it into the other machine's own `config/system_services.yml`, and **Sync from Host** reads it from there.

??? info "Details: the `StreamServer` section"
    - The console reads it to place and probe the MediaMTX card, and to complete a stream written as a bare path (`ips/cam-1`) into the full URLs of its `Streams` entry. The dashboard reads it from the `config/system_services.yml` of its own machine, to ask which streams are live and where browsers play them.
    - Streams and bases read those full URLs, so the section itself is not copied into the pipeline configs.
    - Saved with another address, it moves the stream URLs that named the old one in the pipeline configs of the host it is saved on.
    - **Sync to Host** and **Sync from Host** on its form carry the address between two machines' own `config/system_services.yml`, and merge the `Streams` entries of the source's pipeline configs into the destination's by name.
    - A `config/system_services.yml` with a `Gateway` section but no `StreamServer` section is read as one machine for both: the Stream Server takes the `host`, `rtmp_port` and `rtsp_port` of its `Gateway`. The next **Save** of any form writes them out as a `StreamServer` section.

!!! tip "Host names"
    `<name>.local` needs mDNS and works on one LAN only. Across a Tailscale network, use the MagicDNS name or the tailnet address. See the [FAQ](faq.md#server-name-not-known) when a name does not resolve.

### One master key per machine

A secret in a config file (a token, a password, an API key) is stored as `ENC(...)`: encrypted with the master key of the machine the file is on, `~/.openmmla/master.key`, and decrypted by the services there at startup. Keys named `token`, `hf_token`, `password`, `api_key`, `secret`, `secret_key` or `subscription_key` are encrypted.

- Every machine has its own key, a Fernet key readable by its owner alone (mode `0600`). The console makes it the first time it writes something encrypted to that machine; `openmmla crypto init` makes one by hand, and `openmmla crypto status` checks it.
- The console re-encrypts secrets for the machine it writes to: a Save with that machine picked, **Sync to Host**, **Sync from Host**, the settings a Start pushes there, an SSH profile. The rest of the file stays byte for byte, comments included.
- A colleague's computer running its own console has its own key, like any other machine, and works with the same hosts.

!!! warning "Never copy, replace or delete a `master.key`"
    Every `ENC(...)` value stored on that machine would stop opening there. Back the key up with the machine. `openmmla crypto init --force` overwrites it.

??? info "Details: how the console handles the keys"
    - To re-encrypt, the console reads a host's `~/.openmmla/master.key` over SSH when a write needs it, and makes one there when it has none, never over one that is there.
    - It keeps the keys in memory while it runs and never shows, logs or writes them. The secrets are in plain text only in its memory and inside its SSH connection. Anyone who can log in over SSH as that user can read the key anyway, so this needs no access the console does not have already.
    - A value encrypted with another machine's key is opened with that key and encrypted again. A value the destination's key opens is left as it is, so a copy between two machines with the same key changes nothing.
    - A value none of the machines involved can open (its machine's key is gone) is not copied. A file holding one stays where it is, named in the status line (`<file> on <host> holds an encrypted value none of the keys involved can open`). A Connections field such as the InfluxDB token keeps the destination's own value, and an SSH profile keeps the destination's password, or arrives without one. Type such a secret again where it is needed.
    - When a machine's key cannot be read (the host did not answer in time, or the file cannot be read), nothing that holds a secret is written, and the status line says whose key could not be read and why (`the master key of <host> could not be read (...)`). The same sync carries it once the host answers.
    - Machines that share one key work as one: every value opens everywhere and nothing is re-encrypted.

### A new machine

On a machine without `config/system_services.yml`, every address of the Connections forms reads `<uber-server>`, unless a pipeline config of that machine already names one, as a Start or a sync from another console writes it there. Until it is filled in, the address counts as not set, and so does any section with a `<...>` left in it.

- **One machine that runs everything**: type `localhost` into each form, with InfluxDB `org` `admin` and Redis `db` `1`, and **Save**. Then fill in the InfluxDB token (**Fetch Token** on the InfluxDB card) and the sudo password.
- **Joining a deployment**: add an SSH profile for a machine that is set up, with the **Host** selector on `Local`, and take its other profiles with **Sync from Host** on SSH Profiles. Then press **Sync from Host** on each Connections form, with that machine picked; Experiments and Tasks come the same way when this console is to run sessions. This machine makes its own master key at its first Save, and the secrets it receives are re-encrypted with it.

??? info "Details: what a section that is not set does"
    - It is neither copied into the pipeline configs nor carried to another machine, and the components skip it at startup and keep their own.
    - **Start** refuses a pipeline whose config carries a section with no address (`... has no host yet`). Session Control sends no START or STOP without a Redis host.
    - The Sessions tab and its Export, the Recordings tab and the stream URL completion name the form to fill instead of trying `localhost`.
    - A System Services card whose saved address reads `<uber-server>` says so and reports the port of the host it is on. With no address saved at all, the cards behave as for `localhost`.
    - Saving one form writes only that section, so a section nobody saved never overrides what the pipeline configs say.
    - Only the console looks for placeholders. A component started by hand (`mmla asr-base ...`) whose `config.yml` still says `<uber-server>` tries to reach a machine of that name and fails, and one whose config has no `Gateway.host` at all reaches the AI services through `localhost`.
    - Instead of the forms, start from the template: `sed 's/<uber-server>/localhost/g' config/system_services_template.yml > config/system_services.yml`. The template has `admin` and `1` already. `<influxdb-token>` and `<local-sudo-password>` are placeholders too, and the InfluxDB section counts as not set while its token is one.

## Verify from a base station

```bash
curl -sf http://<uber-server>:8086/health
mongosh "mongodb://<uber-server>:27017" --eval 'db.adminCommand({ping:1})'
redis-cli -h <uber-server> ping
mosquitto_sub -h <uber-server> -t '$SYS/broker/version' -C 1
```

## Troubleshooting

**Redis says the address is already in use.** Another Redis holds the port. Shut it down and let the Makefile restart it with the OpenMMLA listener config:

```bash
redis-cli shutdown
make -C pipelines/uber-server redis
```

**The InfluxDB container restarts in a loop.** `INFLUXDB_INIT_PASSWORD` in `docker/.env` is empty or shorter than 8 characters; see [Docker → Troubleshooting](docker.md#known-caveats).

**A port is held by a process from an earlier run, or a name does not resolve.** See the [FAQ](faq.md#network).
