# System Services

The services below are shared by every pipeline. They usually run together on one dedicated machine, the **Uber Server**, but any split across hosts works: the addresses are entered once under **System Settings → Connections** in the TUI, stored in `config/system_services.yml`, and synced into every pipeline's `config.yml` from there. The TUI's start/stop cards for them sit under **Launcher → System Services**.

| Service | Port | Required | Role |
|---|---|---|---|
| [InfluxDB 2.x](#influxdb) | 8086 | yes | time-series store for every measurement event (`sensor_events`) |
| [MongoDB](#mongodb) | 27017 | yes | session metadata (start/end time, experiment, group, participants) |
| [Redis](#redis) | 6379 | yes | session start/stop control between bases and synchronizers; Celery broker for the dashboard |
| [Mosquitto](#mosquitto) | 1883 | yes | MQTT broker that carries results between *Bases* and *Synchronizers* |
| [Nginx](#nginx-optional) | 8080 | optional | load balancer in front of the AI services |
| [MediaMTX](#mediamtx-optional) | 1935 RTMP, 8554 RTSP, 8890 SRT, 9997 API | optional | streaming server: cameras and microphones publish to it, the bases pull from it, and it records the paths of a running session, START to STOP ([On the server](rtmp_streaming.md#on-the-server)) |
| [Dashboard](#dashboard-optional) | 5050, 5051 and 5052 (media ports, the two after the Dashboard port) | optional | Flask backend and static frontend for live and post-time views |

InfluxDB, MongoDB and MediaMTX can run natively (this page) or as containers from `docker/docker-compose.infra.yml`; see [Docker: Database stack](docker.md#database-stack-influxdb-and-mongodb). Redis, Mosquitto, Nginx and the dashboard run natively. What is stored in the two databases is described in the [Database Reference](database.md).

## Installation

### InfluxDB

```bash
# macOS
brew install influxdb influxdb-cli
brew services start influxdb

# Ubuntu / Debian
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update && sudo apt install -y influxdb2 influxdb2-cli
sudo systemctl enable --now influxdb
```

Open `http://localhost:8086` and run the initial setup: create the admin user, use `admin` as the organisation and `mmla-data` as the bucket (the defaults the pipelines expect), and save the operator API token somewhere safe. That token goes into **System Settings → InfluxDB → token**.

### MongoDB

```bash
# macOS
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Ubuntu / Debian (MongoDB 7.0 repository)
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
sudo systemctl enable --now mongod
```

No initial setup is needed; the `openmmla` database and its `sessions` collection are created on first use. MongoDB runs without authentication in this setup, so keep it on a trusted network.

### Redis

**Start** on the Redis card (or `make redis` in `pipelines/uber-server`) installs it when the host has none, brew on macOS and apt on Debian or Ubuntu, then configures the listener and starts it. By hand:

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu / Debian
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

### Mosquitto

Installed by **Start** on the MQTT card (`make mosquitto`) the same way when missing. By hand:

```bash
# macOS
brew install mosquitto
brew services start mosquitto

# Ubuntu / Debian
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

### Nginx (optional)

Installed by **Start** on the Gateway card (`make nginx`) the same way when missing. By hand:

```bash
# macOS
brew install nginx

# Ubuntu / Debian
sudo apt install -y nginx
```

Configuration (the upstreams) is rendered from `pipelines/uber-server/nginx/config.yml`; see the [Nginx Setup Guide](nginx.md). That file is gitignored, so a host that got the project by `git pull` has none until the Gateway card's Config tab saves one there, or this machine's is copied there (**Sync from Host** with `Local` picked on that host's Config tab, or **Sync to Host** with Host = `Local`); Start stops with a note until then.

### MediaMTX (optional)

The streaming server. With Docker it is a service of `docker/docker-compose.infra.yml` and needs no install. Natively:

```bash
# macOS
brew install mediamtx

# Linux: unpack the release for your architecture from
# https://github.com/bluenviron/mediamtx/releases into /usr/local/bin
```

It reads `pipelines/uber-server/mediamtx/mediamtx.yml`; see the [Streaming guide](rtmp_streaming.md).

### Dashboard (optional)

The dashboard is Python code from this repository, not a package. It needs the `uber-server` conda environment (TUI Environment tab, or `pip install -e '.[uber-server]'`), plus InfluxDB and Redis (the queue of its report worker); MongoDB and the Stream Server add to what it shows when they answer. See the [Dashboard Setup Guide](dashboard.md).

## Listener configuration

Out of the box Redis, Mosquitto and MongoDB only listen on loopback. Every base station has to reach them, so they must bind to a network interface. `make <service>` in `pipelines/uber-server` (and the Start button of the TUI cards, which runs the same target) rewrites the config for you before restarting the service:

| Service | What the make target sets |
|---|---|
| Redis | `bind 0.0.0.0 ::` and `protected-mode no` in `redis.conf` |
| Mosquitto | `listener 1883 0.0.0.0` and `allow_anonymous true` in a `conf.d/openmmla-listener.conf` drop-in |
| MongoDB | `net.bindIp: 0.0.0.0` in `mongod.conf` |
| InfluxDB | `http-bind-address = ":8086"` in `config.toml` (the default already binds all interfaces) |

The original files are backed up as `*.openmmla.bak`. Set `OPENMMLA_BIND_ADDRESS=<LAN IP>` to bind to one interface only, and `OPENMMLA_MQTT_ALLOW_ANONYMOUS=false` if you configure MQTT credentials yourself. These settings assume a trusted lab network; nothing here adds authentication.

## Starting and stopping

### From the TUI

`mmla tui` → **Launcher → System Services** lists one card per service (InfluxDB, MongoDB, Redis, MQTT (Mosquitto), Gateway (Nginx), Stream Server (MediaMTX), Dashboard (Flask), Dashboard (Celery)); each card starts with the name of the Connections form that holds its address. Each card opens on the machine its address in System Settings names (this machine, or the SSH profile that matches it) and can be moved to another host for a one-off; a card whose address is `localhost` remembers the host it was last pointed at, and so does one whose address still reads `<uber-server>`: that placeholder is never looked up, so until the address is filled in the card reports the port of the host it is on (see [A new machine](#a-new-machine)). Press **Start**, **Stop** or **Logs**. The card status is a TCP probe of the address configured in System Settings, so it reflects what the pipelines will see. The InfluxDB, MongoDB and MediaMTX cards have a **Run mode** dropdown: `docker` (default) drives the compose stack, `native` runs the make targets below. Native starts need `sudo`; store the password under **System Settings → Credentials → Sudo (local admin)** and the console types it when the prompt appears; on a remote host it types the SSH profile's password. See the [TUI guide](tui.md#system-services).

### From the shell

The Makefile in `pipelines/uber-server` wraps brew services / systemctl and the tmux sessions of the dashboard:

```bash
cd pipelines/uber-server
make all                                # free the default ports, then (re)start every service
make all without=nginx,flask,celery     # everything except some services
make influxdb mongodb redis mosquitto   # start (and reconfigure) individual services
make flask celery                       # dashboard backend + worker (uber-server conda env), started again when they fail
make autostart DASHBOARD_PORT=5050      # start the dashboard and its worker after every reboot (crontab; make no-autostart; docs/dashboard.md)
make mediamtx                           # streaming server in a tmux session (needs the mediamtx binary)
make stop                               # stop everything
make stop-redis                         # stop one service
make clean-ports 8086 5050 5051 5052    # kill whatever holds those ports
```

`make all` and `make clean-ports` send `kill -9` to whatever listens on the default ports, including the `docker-proxy` of containerized databases. On a host that runs the Docker database stack use `make all without=influxdb,mongodb`.

## Pointing the pipelines at the services

Open **Launcher → System Settings → Connections** in the TUI and fill in the sections below, with the Host selector on the machine whose settings they are (`Local` for this one); **Save** writes that section into its `config/system_services.yml`, where the sections already there stay as they are, and copies it into `pipelines/asr-base/config.yml`, `pipelines/vfa-base/config.yml`, `pipelines/ips-base/config.yml` and the dashboard backend config, those of them that exist. The file is gitignored, like the pipeline configs, because it names your machines and holds the token and sudo password; `config/system_services_template.yml` shows its layout.

| Section | Fields | Notes |
|---|---|---|
| `InfluxDB` | `url`, `token`, `org`, `bucket` | `http://<host>:8086`, org `admin`, bucket `mmla-data` |
| `MongoDB` | `url`, `db` | `mongodb://<host>:27017`, db `openmmla` |
| `MQTT` | `host`, `port` | Mosquitto, 1883 |
| `Redis` | `host`, `port`, `db` | use a db number other than 0 for the Celery queue, e.g. 1 |
| `Dashboard` | `host`, `port` | where the Flask backend runs, 5050. The console places and probes the Dashboard cards with it and passes the port to `make flask`; no pipeline config carries it, so Sync to Host writes it into the other machine's own `config/system_services.yml`, and Sync from Host reads it from there |
| `Gateway` | `host`, `http_port`, `scheme` | the Nginx load balancer the bases reach the AI services through |
| `StreamServer` | `host`, `rtmp_port`, `rtsp_port`, `api_port`, `playback_port`, `webrtc_port` | MediaMTX (`rtmp_port` to publish, `rtsp_port` to pull; `api_port` 9997 and `playback_port` 9996 are what **Sessions → Export** asks for a session's server copy; `webrtc_port` 8889 is what the [dashboard](dashboard.md#camera-tiles-and-live-video)'s camera tiles play live video from); it may run on another machine than Nginx. Read by the console: to place and probe the MediaMTX card, and to complete a stream written as a bare path (`ips/cam-1`) into the full URLs of its `Streams` entry; and by the dashboard, from the `config/system_services.yml` of its own machine, to ask which streams are live and where browsers play them. Streams and bases read those full URLs, so the section itself is not copied into the pipeline configs; when it is saved with another address, the stream URLs that named the old one follow in the pipeline configs of the host it is saved on, and **Sync to Host** and **Sync from Host** on its form carry the address between two machines' own `config/system_services.yml` and merge the `Streams` entries of the source's pipeline configs into the destination's by name. A store written before this section existed keeps these values under `Gateway`, and is read that way until it is next saved |
| `Sudo (local admin)` | `password` | local sudo password for native service starts (under Credentials, never copied into pipeline configs or to another machine; the one form without a Host selector) |

- When a service starts, the four connection sections (`InfluxDB`, `MongoDB`, `MQTT`, `Redis`) of the `config/system_services.yml` found above its `config.yml` are laid over the ones in that `config.yml` field by field (a field the file does not have keeps the pipeline's value), except for sections the pipeline lists under `SystemServicesOverride` and sections that still hold a `<...>` placeholder, which the pipeline keeps as they are. On the console machine the two always agree. On a remote host the file exists once a console was run there or its Connections forms were saved or synced from another console; pick that host in the [System Settings](tui.md#system-settings) forms to see and edit what it will use.
- Keys named `token`, `password`, `api_key`, `secret` or `subscription_key` are stored as `ENC(...)`: encrypted with the master key of the machine the file is on, `~/.openmmla/master.key`, and decrypted by the services there at startup. Every machine has a key of its own: see [One master key per machine](#one-master-key-per-machine).
- Do not edit the synced sections in the pipeline configs by hand; the TUI overwrites them and `config/system_services.yml` is what counts at runtime. A pipeline that must keep its own value can list the section under `SystemServicesOverride:` in its `config.yml`.
- Host names: `<name>.local` needs mDNS and only works on one LAN; across a Tailscale network use the MagicDNS name or the tailnet IP. See the [FAQ](faq.md#server-name-not-known) when a name does not resolve.

### One master key per machine

`ENC(...)` is a secret (a token, a password, an API key) as a config file keeps it: encrypted with the master key of the machine the file is on, so it never stands in the file in plain text.

Every machine has its own key, in `~/.openmmla/master.key`, readable by its owner alone. The console makes it the first time it writes something encrypted to that machine (`openmmla crypto init` makes one by hand). A colleague's computer running a console of its own has its own key, like any other machine, and works with the same hosts.

- Never copy one machine's `master.key` over another's, and never replace or delete it: every `ENC(...)` value stored on that machine would stop opening there. Back it up with the machine.
- The console re-encrypts secrets for the machine it writes to. Whatever it writes to a machine (a Save of a form with that machine picked, Sync to Host, Sync from Host, the settings a Start pushes there, an SSH profile) has each secret encrypted with that machine's key: a value encrypted with another machine's key is opened with that key and encrypted again; a value the destination's key opens is left as it is, so a copy between two machines with the same key changes nothing. The rest of a file stays byte for byte, comments included.
- To do that, the console reads a host's `~/.openmmla/master.key` over SSH when a write needs it, and makes one there when it has none (never over one that is there). It keeps the keys in memory while it runs and never shows, logs or writes them anywhere; the secrets themselves are in plain text only in its memory and inside its SSH connection. Anyone who can log in over SSH as that user can read the key anyway, so this needs no access the console does not have already.
- A value none of the machines involved can open (encrypted on a machine whose key is gone) is not copied. A whole file holding one stays where it is, named in the status line (`<file> on <host> holds an encrypted value none of the keys involved can open`); a Connections field such as the InfluxDB token keeps the destination's own value; an SSH profile keeps the destination's password, or a new one arrives without one. Type such a secret again where it is needed.
- When a machine's key could not be read (the host did not answer in time, or its key file cannot be read), that key may well open the value, so nothing that holds one is written, and the status line says whose key could not be read and why (`the master key of <host> could not be read (...)`). The same sync carries it once the host answers.
- Machines that already share one key keep working as they did: every value opens everywhere and nothing is re-encrypted. A host that holds values encrypted with another machine's key, copied there by an earlier version of the console, gets them re-encrypted for itself by the next Save of that form, or the next sync to it, from a console that holds that key.

### A new machine

On a machine without `config/system_services.yml`, every address of the Connections forms reads `<uber-server>`, the placeholder of `config/system_services_template.yml`, unless a pipeline config of that machine already names one (another console may have written it there with a Start or a sync). Earlier versions showed `localhost`, which looks filled in but means whichever machine reads it. Ports, database names, bucket, org and scheme keep their defaults.

Until an address is filled in, it counts as not set, and so does a section with any `<...>` left in it. Such a section is neither copied into the pipeline configs nor carried to another machine, and the services skip it at startup and keep their own. Start refuses a pipeline whose config carries a section with no address (`... has no host yet`). Session Control sends no START or STOP without a Redis host. The Sessions tab and its Export, the Recordings tab and the stream URL completion name the form to fill instead of trying `localhost`. A System Services card whose saved address reads `<uber-server>` says so and, without looking the placeholder up, reports the port of the host it is on; with no address saved at all, the cards behave as for `localhost`.

- **One machine that runs everything**: type `localhost` into each Connections form and Save, or start from the template, `sed 's/<uber-server>/localhost/g' config/system_services_template.yml > config/system_services.yml`, then fill in the InfluxDB token (its form, or **Fetch Token** on the InfluxDB card) and the Sudo password: `<influxdb-token>` and `<local-sudo-password>` are placeholders too, and the InfluxDB section counts as not set while its token is one.
- **Joining a deployment**: nothing to copy by hand; this machine makes its own master key at its first Save, and the secrets that come from the other machines are re-encrypted with it (see [One master key per machine](#one-master-key-per-machine)). Add an SSH profile for a machine that is set up, with the Host selector on `Local`; its other profiles follow with **Sync from Host** on SSH Profiles. Then open each Connections form on `Local`, pick that machine and press **Sync from Host**; Experiments and Tasks come the same way when this console is to run sessions.
- **A machine that saved its forms with an earlier version** keeps what it wrote then: a Save used to write every section, `localhost` included. Look over each Connections form, and replace a `localhost` that should name the server.

Saving one form writes only that section into `config/system_services.yml`, next to the ones already there, so a section nobody saved never overrides what the pipeline configs say. Only the console looks for placeholders: a component started by hand (`mmla asr-base ...`) whose own `config.yml` still says `<uber-server>` tries to reach a machine of that name and fails, and one whose config has no `Gateway.host` at all reaches the AI services through `localhost`.

## Verify from a base station

```bash
curl -sf http://<uber-server>:8086/health
mongosh "mongodb://<uber-server>:27017" --eval 'db.adminCommand({ping:1})'
redis-cli -h <uber-server> ping
mosquitto_sub -h <uber-server> -t '$SYS/broker/version' -C 1
```

## Official links

- InfluxDB: https://docs.influxdata.com/influxdb/v2/install/
- MongoDB: https://www.mongodb.com/docs/manual/installation/
- Redis: https://redis.io/downloads/
- Mosquitto: https://mosquitto.org/download/
- Nginx: https://nginx.org/en/docs/install.html
- MediaMTX: https://github.com/bluenviron/mediamtx
