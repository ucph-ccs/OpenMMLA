# System Services

The services below are shared by every pipeline. They usually run together on one dedicated machine, the **Uber Server**, but any split across hosts works: the addresses are entered once under **System Settings → Connections** in the TUI, stored in `config/system_services.yml`, and synced into every pipeline's `config.yml` from there. The TUI's start/stop cards for them sit under **Launcher → System Services**.

| Service | Port | Required | Role |
|---|---|---|---|
| [InfluxDB 2.x](#influxdb) | 8086 | yes | time-series store for every measurement event (`sensor_events`) |
| [MongoDB](#mongodb) | 27017 | yes | session metadata (start/end time, experiment, group, participants) |
| [Redis](#redis) | 6379 | yes | session start/stop control between bases and synchronizers; Celery broker for the dashboard |
| [Mosquitto](#mosquitto) | 1883 | yes | MQTT broker that carries results between *Bases* and *Synchronizers* |
| [Nginx](#nginx-optional) | 8080 HTTP, 1935 RTMP | optional | load balancer in front of the AI services; RTMP ingest for camera and microphone streams |
| [Dashboard](#dashboard-optional) | 5050 | optional | Flask backend and static frontend for live and post-time views |

InfluxDB and MongoDB can run natively (this page) or as containers from `docker/docker-compose.infra.yml`; see [Docker: Database stack](docker.md#database-stack-influxdb-and-mongodb). Redis, Mosquitto, Nginx and the dashboard run natively. What is stored in the two databases is described in the [Database Reference](database.md).

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

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu / Debian
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

### Mosquitto

```bash
# macOS
brew install mosquitto
brew services start mosquitto

# Ubuntu / Debian
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

### Nginx (optional)

```bash
# macOS
brew install nginx            # or: brew tap denji/nginx && brew install nginx-full --with-rtmp-module

# Ubuntu / Debian
sudo apt install -y nginx     # add libnginx-mod-rtmp for RTMP streaming
```

Configuration (upstreams, RTMP apps) is rendered from `pipelines/uber-server/nginx/config.yml`; see the [Nginx Setup Guide](nginx.md).

### Dashboard (optional)

The dashboard is Python code from this repository, not a package. It needs the `uber-server` conda environment (TUI Environment tab, or `pip install -e '.[uber-server]'`), plus Redis and InfluxDB; see the [Dashboard Setup Guide](dashboard.md).

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

`mmla tui` → **Launcher → System Services** lists one card per service (InfluxDB, MongoDB, Redis, Mosquitto, Nginx, Dashboard (Flask), Dashboard Worker (Celery)). Select the host (local, or an SSH profile) and press **Start**, **Stop** or **Logs**. The card status is a TCP probe of the address configured in System Settings, so it reflects what the pipelines will see. The InfluxDB and MongoDB cards have a **Run mode** dropdown: `docker` (default) drives the compose stack, `native` runs the make targets below. Native starts need `sudo`; store the password under **System Settings → Credentials → Sudo (local admin)** and the console types it when the prompt appears. See the [TUI guide](tui.md#system-services).

### From the shell

The Makefile in `pipelines/uber-server` wraps brew services / systemctl and the tmux sessions of the dashboard:

```bash
cd pipelines/uber-server
make all                                # free the default ports, then (re)start every service
make all without=nginx,flask,celery     # everything except some services
make influxdb mongodb redis mosquitto   # start (and reconfigure) individual services
make flask celery                       # dashboard backend + worker (uber-server conda env)
make stop                               # stop everything
make stop-redis                         # stop one service
make clean-ports 8086 5050              # kill whatever holds those ports
```

`make all` and `make clean-ports` send `kill -9` to whatever listens on the default ports, including the `docker-proxy` of containerized databases. On a host that runs the Docker database stack use `make all without=influxdb,mongodb`.

## Pointing the pipelines at the services

Open **Launcher → System Settings → Connections** in the TUI and fill in the sections below; **Save** writes `config/system_services.yml` and copies the sections into `pipelines/asr-base/config.yml`, `pipelines/vfa-base/config.yml`, `pipelines/ips-base/config.yml` and the dashboard backend config. The file is gitignored, like the pipeline configs, because it names your machines and holds the token and sudo password; `config/system_services_template.yml` shows its layout.

| Section | Fields | Notes |
|---|---|---|
| `InfluxDB` | `url`, `token`, `org`, `bucket` | `http://<host>:8086`, org `admin`, bucket `mmla-data` |
| `MongoDB` | `url`, `db` | `mongodb://<host>:27017`, db `openmmla` |
| `MQTT` | `host`, `port` | Mosquitto, 1883 |
| `Redis` | `host`, `port`, `db` | use a db number other than 0 for the Celery queue, e.g. 1 |
| `Dashboard` | `host`, `port` | where the Flask backend runs, 5050 |
| `Gateway` | `host`, `http_port`, `rtmp_port`, `scheme` | the Nginx entry point that bases and streams use |
| `Sudo (local admin)` | `password` | local sudo password for native service starts (under Credentials, never copied into pipeline configs) |

- Keys named `token`, `password`, `api_key`, `secret` or `subscription_key` are encrypted to `ENC(...)` with the Fernet key in `~/.openmmla/master.key` when the file is saved and decrypted by the services at startup. The key is created on first use, and the TUI copies it to a remote host together with any config that contains encrypted values.
- Do not edit the synced sections in the pipeline configs by hand; the TUI overwrites them and `config/system_services.yml` is what counts at runtime. A pipeline that must keep its own value can list the section under `SystemServicesOverride:` in its `config.yml`.
- Host names: `<name>.local` needs mDNS and only works on one LAN; across a Tailscale network use the MagicDNS name or the tailnet IP. See the [FAQ](faq.md#server-name-not-known) when a name does not resolve.

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
