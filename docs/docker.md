# Dockerized OpenMMLA services

Two kinds of stacks live in this directory:

- **AI service stacks** (`docker-compose.asr.yml`, `docker-compose.vfa.yml`): one image per ASR/VFA service, so every service keeps its own Python environment and can be upgraded without touching the others. The ports match the Nginx gateway upstreams, so the gateway config needs no change.
- **Database stack** (`docker-compose.infra.yml`): InfluxDB and MongoDB for the uber server, as an alternative to installing them with brew/apt. See [Database stack: InfluxDB and MongoDB](#database-stack-influxdb-and-mongodb).

| Service | Image | Port | GPU | Stack |
|---|---|---|---|---|
| AudioInferer (wespeaker) | `openmmla/asr-audio-inferer-wespeaker` | 5001 | ✅ | torch 2.4.1 + wespeaker |
| AudioInferer (nemo, optional) | `openmmla/asr-audio-inferer-nemo` | 5001 | ✅ | nemo-toolkit ≤1.23 |
| AudioResampler | `openmmla/asr-audio-resampler` | 5002 | — | librosa (CPU) |
| SpeechEnhancer | `openmmla/asr-speech-enhancer` | 5003 | ✅ | torch 2.4.1 + denoiser |
| SpeechSeparator | `openmmla/asr-speech-separator` | 5004 | ✅ | torch 2.4.1 + modelscope |
| SpeechTranscriber | `openmmla/asr-speech-transcriber` | 5005 | ✅ | **whisperx 3.8.6 + torch 2.8 + ct2 ≥4.5 (cuDNN 9)** |
| VoiceActivityDetector | `openmmla/asr-voice-activity-detector` | 5006 | — | silero-vad (CPU torch) |
| VLLMFrameAnalyzer | `openmmla/vfa-frame-analyzer` | 5007 | ✅ | torch 2.7 + tf-keras/retina-face |
| vLLM VLM backend (optional) | `vllm/vllm-openai` | 8000 | ✅ | official image, profile `mllm` |

## Host requirements

For the AI service stacks (a GPU base server):

- NVIDIA driver
- Docker Engine + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `~/.openmmla/master.key`, mounted read-only into the containers to decrypt the `ENC(...)` values in the config
- the current user in the `docker` group, so `docker` runs without `sudo` (the TUI runs plain `docker compose`)

The database stack only needs Docker Engine: no GPU, no nvidia-container-toolkit, no master key.

## AI service stacks: ASR and VFA

Run from the repository root:

```bash
# all ASR services (build + start)
docker compose -f docker/docker-compose.asr.yml up -d --build

# VFA frame analyzer
docker compose -f docker/docker-compose.vfa.yml up -d --build

# status / logs
docker compose -f docker/docker-compose.asr.yml ps
docker compose -f docker/docker-compose.asr.yml logs -f speech-transcriber

# stop
docker compose -f docker/docker-compose.asr.yml down
```

The service configs stay on the host: `pipelines/asr-server/config.yml` and `pipelines/vfa-server/config.yml` are bind-mounted into the containers as `/project/config.yml`, so edit them exactly as before (the TUI's Config tab on the ASR Server / VFA Server cards writes the same files).

### Switching the inferer backend: wespeaker or nemo

Both listen on 5001, so only one can run at a time:

```bash
docker compose -f docker/docker-compose.asr.yml stop audio-inferer
docker compose -f docker/docker-compose.asr.yml --profile nemo up -d audio-inferer-nemo
```

Set `AudioInferer.backend` to `nemo` in `pipelines/asr-server/config.yml` at the same time. The TUI picks the container from that setting automatically.

### Local vLLM VLM backend

```bash
VLLM_VLM_MODEL=openbmb/MiniCPM-V-2_6 \
docker compose -f docker/docker-compose.vfa.yml --profile mllm up -d
```

The frame analyzer container maps `host.docker.internal` to the host, so a `vlm_base_url` of `http://localhost:8000/v1` in the config has to become `http://host.docker.internal:8000/v1` (or the compose service name, `http://vllm-vlm:8000/v1`).

This is separate from the TUI's **MLLM Server** card, which runs vLLM natively in the `vfa-vllm` conda environment from `config/mllm_server.yml`.

## Database stack: InfluxDB and MongoDB

`docker-compose.infra.yml` runs the two uber-server databases as containers instead of bare-metal brew/apt installs. The default ports match `config/system_services.yml`, so the pipelines only need the host name changed; the host ports can be overridden with `INFLUXDB_PORT` / `MONGODB_PORT` (see [Sharing a host with another project](#sharing-a-host-with-another-project)).

| Service | Image | Port | GPU | Stack |
|---|---|---|---|---|
| InfluxDB | `influxdb:2.7.12` | 8086 | — | official image, v2 API (org/bucket/token + Flux) |
| MongoDB | `mongo:7.0.40-jammy` | 27017 | — | official image, no authentication by default |

The image tags are pinned on purpose; do not switch them to `latest`. From 2026-09-15 `influxdb:latest` points at InfluxDB 3 Core, which has no org/bucket/token semantics and breaks `influxdb-client==1.44.0` outright, and `mongo:latest` drifts across major versions. Both pinned tags are published for linux/amd64 and linux/arm64.

### Setup

Run the commands on the **machine that will hold the databases** (called `server-01` below), from the repository root. That machine only needs a clone of the repository, or just these two files from `docker/`: the stack has no build context.

First stop any bare-metal service that may hold 8086 / 27017, otherwise `up -d` fails with `Bind for 0.0.0.0:8086 failed: port is already allocated`. If the port is held by **another project's container**, do not stop it; change our host port instead, see [Sharing a host with another project](#sharing-a-host-with-another-project).

```bash
# macOS
brew services stop influxdb mongodb-community
# Ubuntu / Debian
sudo systemctl disable --now influxdb mongod
```

Then fill in the secrets and start:

```bash
cp docker/.env.example docker/.env && chmod 600 docker/.env
# edit docker/.env: at least INFLUXDB_INIT_ADMIN_TOKEN and INFLUXDB_INIT_PASSWORD
#   openssl rand -hex 32      -> INFLUXDB_INIT_ADMIN_TOKEN
#   openssl rand -base64 24   -> INFLUXDB_INIT_PASSWORD

docker compose -f docker/docker-compose.infra.yml up -d

# status / logs / stop (down keeps the volumes, down -v deletes all data)
docker compose -f docker/docker-compose.infra.yml ps
docker compose -f docker/docker-compose.infra.yml logs -f influxdb
docker compose -f docker/docker-compose.infra.yml down
```

Keep the secrets in `docker/.env` (gitignored) rather than `export`ing them: compose reads `.env` for **every** subcommand, so `ps`, `logs` and `down` behave the same from any shell and after a reboot, while an exported value only lives in that one shell and ends up in `~/.bash_history`. The same goes for `INFRA_BIND_ADDRESS`: only a value in `.env` guarantees that every later `up -d` uses the same bind address instead of silently falling back to `0.0.0.0`.

### Migrating from the bare-metal databases (do this first)

Both containers start as **brand-new, empty databases**. If you just repoint the URLs, the TUI's session list, the dashboard's session history and every historical `sensor_events` point will appear to vanish. The data is still on the old machine; nothing connects to it any more.

To keep the history, migrate before changing the URLs. On the **old** uber server export:

```bash
mongodump --uri "mongodb://localhost:27017" --db openmmla --archive=openmmla.archive
influx backup ./influx-backup -t "<old admin token>"
```

Copy both to `server-01` and import into the containers:

```bash
docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
  mongorestore --archive --db openmmla < openmmla.archive

docker cp ./influx-backup "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/influx-backup
docker compose -f docker/docker-compose.infra.yml exec influxdb \
  influx restore /tmp/influx-backup --full
```

`influx restore --full` also overwrites the tokens with the old instance's, so afterwards enter the **old** token in the TUI. To keep the new token, restore only the data with `--bucket mmla-data`.

Not migrating is fine too: leave the old machine running, the old and new data just stay apart from then on.

### Backups

The data lives only in named volumes. `docker compose down -v`, `docker volume rm` and a few "just reset it" tutorial commands **delete them permanently**; there is no recycle bin. Run the two exports below regularly and keep the output outside Docker:

```bash
docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
  mongodump --archive --db openmmla > /backup/openmmla-$(date +%F).archive

docker compose -f docker/docker-compose.infra.yml exec influxdb \
  influx backup /tmp/backup -t "<token>"
docker cp "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/backup /backup/influx-$(date +%F)
```

Do not `down` the stack while a collection is running. Both services have `stop_grace_period: 60s`, but the normal order is: end the session, then stop the stack.

### Sharing a host with another project

If the machine already runs another project's InfluxDB / MongoDB containers, even ones bound only to `127.0.0.1:8086`, our `0.0.0.0:8086` will not come up: the kernel does not allow a wildcard bind and a specific bind on the same port. **Do not stop their containers**; change our host ports:

```bash
# docker/.env
INFLUXDB_PORT=8087
MONGODB_PORT=27018
```

Then put the new ports in the System Settings URLs (`http://server-01:8087`, `mongodb://server-01:27018`). The TUI's status probe reads the port from the URL, so nothing else changes. Inside the containers the ports stay 8086 / 27017, so the healthchecks and the data are unaffected.

Do not go the other way and reuse the other project's instance: it is usually bound to loopback only (unreachable from other machines), has authentication enabled (the password would have to go into `MongoDB.url` in plaintext), and if it runs `influxdb:latest`, one `pull` after 2026-09-15 turns it into InfluxDB 3 and takes your data with it.

### Pointing OpenMMLA at the stack

`mmla tui` → Launcher → **System Settings → Connections**; only three fields change:

| Field | Value |
|---|---|
| `InfluxDB.url` | `http://server-01.local:8086` (`org: admin` and `bucket: mmla-data` stay as they are) |
| `InfluxDB.token` | the `INFLUXDB_INIT_ADMIN_TOKEN` from above, replacing the whole `ENC(...)` string with the plain value; or press **Fetch Token** on the InfluxDB card (see below) |
| `MongoDB.url` | `mongodb://server-01.local:27017` (`db: openmmla` stays) |

**How you spell the host name depends on your network.** `.local` is mDNS and only works on the **same LAN**. If your Mac and `server-01` are on different subnets with Tailscale in between (`ssh admin@server-01` works but `ping server-01.local` does not), use the Tailscale MagicDNS name or the tailnet IP: `http://server-01:8086`, `http://100.x.x.x:8086`. That also means **every base station has to be in the tailnet**, or it cannot reach the databases. If you changed the ports, include them: `http://server-01:8087`.

On save the token is re-encrypted to `ENC(...)` and written into every local `pipelines/*/config.yml`. **Do not edit those files by hand**: they get overwritten, and at runtime `config/system_services.yml` wins anyway. The first launch on a remote host syncs the config first and then asks you to launch again; that is by design, not an error.

The new container is a fresh InfluxDB, so the old token is guaranteed to fail authentication; it must be replaced with the new one.

Verify, from the machine that runs the TUI:

```bash
ping -c1 server-01.local
curl -sf http://server-01.local:8086/health
mongosh mongodb://server-01.local:27017 --eval 'db.adminCommand({ping:1})'
```

`server-01.local` relies on mDNS. Ubuntu Server ships neither `avahi-daemon` nor `libnss-mdns` by default (`sudo apt install -y avahi-daemon libnss-mdns` and `sudo hostnamectl set-hostname server-01`), and mDNS does not cross subnets or VLANs. If the name does not resolve, fall back to a fixed IP or `/etc/hosts`.

**These three checks run on the host, but the ASR / VFA containers read the same URLs**, and containers on a bridge network do no mDNS resolution by default: `ping server-01.local` working on the host does not mean it works inside the container. When the containerized AI services are in use as well, put a fixed IP in `config/system_services.yml`, or add `extra_hosts` to the two AI compose files.

### MongoDB authentication (optional, strongly recommended)

Authentication is off by default, matching the bare-metal setup. To enable it, set both variables **before the first start**:

```bash
export MONGO_ROOT_USER=openmmla
export MONGO_ROOT_PASSWORD="<password>"
docker compose -f docker/docker-compose.infra.yml up -d
```

- This only takes effect while the `mongodb-data` volume is empty. Adding the variables once the volume holds data gives you `--auth` with no users at all, and nobody can connect. **Do not delete the volume** in that case: clear both variables and `up -d` again to return to no-auth with the data untouched, or create the user through the container's localhost exception:
  `docker compose -f docker/docker-compose.infra.yml exec mongodb mongosh admin --eval 'db.createUser({user:"openmmla",pwd:"<password>",roles:["root"]})'`
- Setting only one of the two variables makes the container **restart-loop** while `up -d` still reports success. If 27017 never opens, check `docker compose -f docker/docker-compose.infra.yml ps` and `... logs mongodb`.
- With auth on, the URL must be `mongodb://<user>:<pass>@server-01.local:27017/?authSource=admin`; without `authSource=admin` authentication fails.
- `url` is not on the list of encrypted fields (only keys such as token/password/secret are encrypted), and `config/system_services.yml` is **tracked by git**, so a URL with a password is committed in plaintext. Weigh that before enabling auth, at least until `MongoDB` gets separate username/password fields.

## Mounts

- `pipelines/asr-server` / `pipelines/vfa-server` → `/project` in the container: `config.yml`, `temp/` and runtime logs stay on the host, the same as with conda.
- Model caches (HuggingFace / torch hub / ModelScope / wespeaker) are shared named volumes, so re-created containers do not download again.
- `~/.openmmla` is mounted read-only so the containers can decrypt `ENC(...)` secrets.
- The database stack's data lives entirely in named volumes: `influxdb-data` (`/var/lib/influxdb2`, with `influxd.bolt` and the engine), `influxdb-config` (`/etc/influxdb2`, with `influx-configs`, from which the admin token can be recovered), `mongodb-data` (`/data/db`) and `mongodb-config` (`/data/configdb`). Do not bind-mount `/data/db`: WiredTiger needs real file-lock semantics.
- The database containers do **not** mount `~/.openmmla`: the official influxdb / mongo images contain no OpenMMLA code, never read `config.yml`, and have nothing to decrypt.

## How the TUI uses these stacks

**ASR Server / VFA Server** cards (Launcher → Pipelines → ASR → ASR Server, Launcher → Pipelines → VFA → VFA Server): Start / Stop / Logs all go through docker compose; the tmux + gunicorn way has been removed.

- **Start**: `docker compose -f docker/docker-compose.*.yml up -d --build <selected services>` (AudioInferer picks the wespeaker or nemo container from `backend` in the config)
- **Stop**: `docker compose ... down`
- **Logs**: `docker compose ... logs --tail 40`
- **Status**: still probed by port; a running container shows `Running` on the card and the `(R)` marker in the tree

A remote host runs the same commands over SSH in its repository directory, so it needs: the repository cloned (including `docker/`), Docker Engine + nvidia-container-toolkit, and the current user in the `docker` group.

**InfluxDB / MongoDB** cards (Launcher → System Services):

- **Status**: the cards probe the URL configured in System Settings, with a direct TCP connect from the TUI machine to host:port, independent of the Host selector. Only when the URL says `localhost` do they fall back to probing the selected host's own loopback. The Status tab does the same and its port column shows the host:port that was actually probed.
- **Start / Stop / Logs**: each card has a **Run mode** dropdown (`docker` / `native`), **`docker` by default**. Switch a machine that still uses brew / systemctl databases to `native`, otherwise Start brings up a container on that machine and fights the bare-metal instance for the port. In `docker` mode the three buttons run `docker compose -f docker/docker-compose.infra.yml up -d / stop / logs <service>`. Stop uses `stop`, not `down`: both cards share one compose file and `down` would take the other database's container with it.
- Run mode is remembered per host + service, so switching Host or clicking another node and coming back keeps it, but only for this TUI session; a restart returns to `docker`.
- **Fetch Token** (InfluxDB card only, docker mode): reads the docker stack's admin token on the selected Host, first from the running container's `/etc/influxdb2/influx-configs` (which also covers a token influx generated itself), then from `docker/.env`, and stores it encrypted in System Settings as `InfluxDB.token`. The token never appears in the logs; only its first and last 4 characters are shown. The TUI warns when the URL's host and the host the token was read from differ.
- **Status tab, Host column**: for InfluxDB / MongoDB / Redis / Mosquitto / Nginx it shows **the machine the service is configured on** (`server-01`, `ericli.local`), not where the TUI runs; the Port column is the port. View Logs finds the SSH profile for that host (matched by profile name or host), reads locally when the host is this machine, reads the container logs when a container exists, and otherwise falls back to journalctl / brew logs.
- Reachability is judged **from the TUI machine**, which is the path the pipelines actually take. With `InfluxDB.url` set to `http://server-01:8087`, the card and the Status tab connect to `server-01:8087`; which interface `INFRA_BIND_ADDRESS` binds does not matter. The card description states the address being probed. Two consequences: a `localhost` / `127.0.0.1` URL names no particular machine, so the old logic applies (local probes this machine, a remote Host is probed over SSH on its own loopback); and if the TUI machine is outside the tailnet or behind a firewall, the card is grey even when the pipeline machines can connect.
- After the databases move to `server-01`, the two cards on a Mac with `localhost` URLs stay grey, because the local probe hits 127.0.0.1. That is expected, not a connectivity problem. **Do not press Start on the Mac in that state**: in `native` mode it runs `make influxdb` / `make mongodb`, which starts a bare-metal database on local 8086 / 27017, and the card turns green `Running` for an empty local database while the pipelines still use `server-01`. Stop those two local services once the move is done.

## Known caveats

- The nemo image is experimental: nemo-toolkit ≤1.23 has old dependencies and a long build.
- The transcriber image uses whisperx 3.8.6 with the config unchanged (`whisperx/small`, `da`, ...); the first request downloads the model into the hf-cache volume.
- The separator's MossFormer2 model cache is written to `/root/.cache/modelscope` (named volume).
- **`DOCKER_INFLUXDB_INIT_*` only apply on the first start against an empty `influxdb-data` volume.** Once the entrypoint sees `influxd.bolt` it skips setup entirely; changing the token / org / bucket / password afterwards and running `up -d` neither errors nor takes effect. To change the token later use
  `docker compose -f docker/docker-compose.infra.yml exec influxdb influx auth create --org admin --all-access`;
  to recover the initial token use `... exec influxdb cat /etc/influxdb2/influx-configs`.
- **Do not delete `influxd.bolt` to re-run the bootstrap.** The entrypoint only checks whether the file exists; with it gone, setup runs again against a persisted `/etc/influxdb2` that already holds a config, most likely fails, and its failure path runs `rm -rf` on the engine directory. All data gone. Use `influx auth create` from the previous point to change tokens.
- The InfluxDB web UI is published on the same port 8086; anyone who can reach it can log in with `admin` + `INFLUXDB_INIT_PASSWORD`. Do not pad that password to 8 characters; generate it with `openssl rand -base64 24`.
- **Do not run `make -C pipelines/uber-server all` on a machine that runs the containerized databases**: `all` starts with `clean-ports`, which `kill -9`s whatever holds 8086 / 27017, i.e. docker-proxy, and the `influxdb` / `mongodb` targets then start bare-metal services on top. Use `make all without=influxdb,mongodb` there.
- MongoDB runs without authentication by default, and on Linux Docker's NAT rules run before ufw, so `ufw deny 27017` does not block a published port. Use it only on a trusted lab network, restrict publishing to one interface with `INFRA_BIND_ADDRESS=<LAN IP>`, or add iptables rules to the `DOCKER-USER` chain.
- MongoDB 5.0+ requires AVX on x86_64 (ARMv8.2-A or newer on arm64). On an older CPU the container restart-loops with exit code 132 while `up -d` reports success. Check with `grep -m1 -o avx /proc/cpuinfo` on the server before going live.
