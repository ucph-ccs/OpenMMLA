# Docker

OpenMMLA runs two kinds of Docker stacks: the AI services of ASR and VFA, one image per service, on the GPU server; and InfluxDB, MongoDB and the MediaMTX Stream Server on the uber server. The console runs the AI services in Docker only; by hand they also run as gunicorn processes ([ASR](pipelines/asr/run.md#run-the-server-without-docker), [VFA](pipelines/vfa/run.md#run-the-server-without-docker)). The databases and the Stream Server may run natively instead ([System services](system_services.md)).

## What runs in Docker

| Stack | Compose file | Runs on | What it holds |
|---|---|---|---|
| ASR services | `docker/docker-compose.asr.yml` | GPU server | the six ASR services, each with its own Python environment, upgradable without the others |
| VFA service | `docker/docker-compose.vfa.yml` | GPU server | the frame analyzer |
| Databases and streaming | `docker/docker-compose.infra.yml` | uber server | InfluxDB, MongoDB and MediaMTX |

The ports of the AI services match the upstreams of the Gateway (Nginx), so its config needs no change ([Nginx](nginx.md)).

| Service | Image | Port | GPU | Stack |
|---|---|---|---|---|
| AudioInferer (wespeaker) | `openmmla/asr-audio-inferer-wespeaker` | 5001 | yes | torch 2.4.1 + wespeaker |
| AudioInferer (nemo, optional) | `openmmla/asr-audio-inferer-nemo` | 5001 | yes | nemo-toolkit ≤1.23 |
| AudioResampler | `openmmla/asr-audio-resampler` | 5002 | no | librosa (CPU) |
| SpeechEnhancer | `openmmla/asr-speech-enhancer` | 5003 | yes | torch 2.4.1 + denoiser |
| SpeechSeparator | `openmmla/asr-speech-separator` | 5004 | yes | torch 2.4.1 + modelscope |
| SpeechTranscriber | `openmmla/asr-speech-transcriber` | 5005 | yes | whisperx 3.8.6 + torch 2.8 + ct2 ≥4.5 (cuDNN 9) |
| VoiceActivityDetector | `openmmla/asr-voice-activity-detector` | 5006 | no | silero-vad (CPU torch) |
| VLLMFrameAnalyzer | `openmmla/vfa-frame-analyzer` | 5007 | yes | torch 2.7 + tf-keras and retina-face + ultralytics 8.4 (YOLO26 pose) + onnxruntime-gpu 1.22 (the tracker's optional face check) |
| InfluxDB | `influxdb:2.7.12` | 8086 | no | official image, v2 API (org, bucket, token and Flux) |
| MongoDB | `mongo:7.0.40-jammy` | 27017 | no | official image, no authentication by default |
| MediaMTX | `bluenviron/mediamtx:1.21.0-ffmpeg` | 1935, 8554, 8890/udp, 9997, 9996, 8889, 8189/udp and tcp | no | official image with FFmpeg, for the dashboard's live sound ([Streaming → Ports](streaming/index.md#ports)) |

!!! note "Licence of the frame analyzer image"
    `ultralytics` is AGPL-3.0. Serving the frame analyzer image over a network carries the AGPL source-offer obligation for the combined work. For an AGPL-free deployment, leave it out and set `features.enabled: false` ([Skeletons](pipelines/vfa/pose-and-gaze.md#skeletons)).

The database images are pinned on purpose; do not switch them to `latest`. `influxdb:latest` is InfluxDB 3 Core, which has no org, bucket or token and breaks `influxdb-client==1.44.0`, and `mongo:latest` drifts across major versions. Both pinned tags are published for linux/amd64 and linux/arm64.

## What you need { #host-requirements }

| Host | Needs |
|---|---|
| GPU server (AI services) | the NVIDIA driver; Docker Engine and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); the host's own `~/.openmmla/master.key` ([Mounts](#mounts)); the user in the `docker` group, since the console runs plain `docker compose` |
| uber server (databases and streaming) | Docker Engine only: no GPU, no container toolkit, no master key |
| either, run from the console | a clone of the repository, `docker/` included, at the SSH profile's `remote_project_path` |

## Run the AI services

The **ASR Server** and **VFA Server** cards run these commands on their host ([How the console runs the stacks](#how-the-tui-uses-these-stacks)). By hand, from the repository root:

```bash
# all ASR services (build + start)
docker compose -f docker/docker-compose.asr.yml up -d --build

# VFA frame analyzer
docker compose -f docker/docker-compose.vfa.yml up -d --build

# status, logs
docker compose -f docker/docker-compose.asr.yml ps
docker compose -f docker/docker-compose.asr.yml logs -f speech-transcriber

# stop
docker compose -f docker/docker-compose.asr.yml down
```

The service configs stay on the host: `pipelines/asr-server/config.yml` and `pipelines/vfa-server/config.yml` are mounted into the containers as `/project/config.yml`. Edit them on the **Config** tab of the ASR Server and VFA Server cards, which write the same files.

### Switch the inferer backend { #switching-the-inferer-backend-wespeaker-or-nemo }

The wespeaker and nemo inferers both listen on 5001, so only one runs at a time. Set `AudioInferer.backend` to `nemo` in `pipelines/asr-server/config.yml`; the card then starts the nemo container. By hand:

```bash
docker compose -f docker/docker-compose.asr.yml stop audio-inferer
docker compose -f docker/docker-compose.asr.yml --profile nemo up -d audio-inferer-nemo
```

The nemo image is experimental: nemo-toolkit ≤1.23 has old dependencies and a long build.

### A VLM server on the same host

The VLM servers are not containers: the console's **MLLM Server** card runs vLLM natively in the `vfa-vllm` conda environment from `config/mllm_server.yml`, and Ollama is a host install. The frame analyzer container maps `host.docker.internal` to its host, so a `vlm_base_url` of `http://localhost:<port>/v1` becomes `http://host.docker.internal:<port>/v1` when that server runs on the same machine.

## Run the databases and the Stream Server { #database-stack-influxdb-and-mongodb }

`docker/docker-compose.infra.yml` runs InfluxDB, MongoDB and MediaMTX as containers instead of brew or apt installs. Its default ports match `config/system_services.yml`, so the pipelines only need the host name changed.

### Setup

Run these steps on the machine that will hold the databases, `uber-server` below, from the repository root. It needs a clone of the repository: MediaMTX reads `pipelines/uber-server/mediamtx/mediamtx.yml` from it and records into its `artifacts/`. The databases alone need just `docker/docker-compose.infra.yml` and `docker/.env.example`, since the stack has no build context.

1. **Free the ports.** Stop any bare-metal service that holds 8086 or 27017. When another project's container holds them, leave it running and change this stack's ports instead ([Share a host with another project](#sharing-a-host-with-another-project)).

    ```bash
    # macOS
    brew services stop influxdb mongodb-community
    # Ubuntu / Debian
    sudo systemctl disable --now influxdb mongod
    ```

2. **Fill in the secrets** in `docker/.env` ([Environment variables](#environment-variables)). Generate the password rather than padding one to eight characters: the InfluxDB web UI is published on port 8086, and anyone who reaches it can log in as `admin` with it.

    ```bash
    cp docker/.env.example docker/.env && chmod 600 docker/.env
    # edit docker/.env: at least INFLUXDB_INIT_ADMIN_TOKEN and INFLUXDB_INIT_PASSWORD
    #   openssl rand -hex 32      -> INFLUXDB_INIT_ADMIN_TOKEN
    #   openssl rand -base64 24   -> INFLUXDB_INIT_PASSWORD
    ```

3. **Start the stack.**

    ```bash
    docker compose -f docker/docker-compose.infra.yml up -d

    # status, logs, stop (down keeps the volumes, down -v deletes all data)
    docker compose -f docker/docker-compose.infra.yml ps
    docker compose -f docker/docker-compose.infra.yml logs -f influxdb
    docker compose -f docker/docker-compose.infra.yml down
    ```

4. **Point OpenMMLA at the stack** ([below](#pointing-openmmla-at-the-stack)).

!!! warning "The first start sets InfluxDB and MongoDB up"
    The `INFLUXDB_INIT_*` and `MONGO_ROOT_*` values take effect only on the first start against an empty volume. Changing them afterwards neither errors nor takes effect ([Troubleshooting](#known-caveats)).

!!! warning "Do not run `make all` on this machine"
    `make -C pipelines/uber-server all` starts with `clean-ports`, which sends `kill -9` to whatever holds 8086 and 27017, including Docker's proxy, and then starts bare-metal databases on top. Use `make all without=influxdb,mongodb` there.

??? info "Details: why the secrets go in `docker/.env`"
    `docker/.env` is gitignored. Compose reads it for every subcommand, so `ps`, `logs` and `down` behave the same from any shell and after a reboot, while an exported value lives in one shell only and ends up in `~/.bash_history`. Only a value in `.env` makes every later `up -d` use the same `INFRA_BIND_ADDRESS` instead of falling back to `0.0.0.0`. The same file serves the ASR stack on the GPU server, for `HF_TOKEN`.

### Point OpenMMLA at the stack { #pointing-openmmla-at-the-stack }

In `mmla tui`, open `Launcher → System Settings → Connections` and change three fields:

| Field | Value |
|---|---|
| `InfluxDB.url` | `http://uber-server.local:8086`; `org: admin` and `bucket: mmla-data` stay |
| `InfluxDB.token` | the `INFLUXDB_INIT_ADMIN_TOKEN` from `docker/.env`, replacing the whole `ENC(...)` string; or press **Fetch Token** on the InfluxDB card |
| `MongoDB.url` | `mongodb://uber-server.local:27017`; `db: openmmla` stays |

The new container is a fresh InfluxDB, so the old token fails: replace it. **Save** encrypts the token again and writes it into every local `pipelines/*/config.yml` ([Pointing the pipelines at the services](system_services.md#pointing-the-pipelines-at-the-services)).

Spell the host as your network resolves it. `.local` is mDNS and works on the same LAN only. Across subnets with Tailscale in between (`ssh uber-server` works, `ping uber-server.local` does not), use the MagicDNS name or the tailnet IP, `http://uber-server:8086`, and put every base station in the tailnet. Include a changed port: `http://uber-server:8087`.

Verify from the machine that runs the console:

```bash
ping -c1 uber-server.local
curl -sf http://uber-server.local:8086/health
mongosh mongodb://uber-server.local:27017 --eval 'db.adminCommand({ping:1})'
```

??? info "Details: mDNS on Ubuntu Server and inside containers"
    - Ubuntu Server ships neither `avahi-daemon` nor `libnss-mdns`: `sudo apt install -y avahi-daemon libnss-mdns` and `sudo hostnamectl set-hostname uber-server`. mDNS does not cross subnets or VLANs. When the name does not resolve, use a fixed IP or `/etc/hosts`.
    - The ASR and VFA containers read the same URLs, and a container on a bridge network does no mDNS resolution: `ping uber-server.local` working on the host does not mean it works inside a container. With the AI services in Docker, put a fixed IP in `config/system_services.yml`, or add `extra_hosts` to the two AI compose files.

### Move data from bare-metal databases { #migrating-from-the-bare-metal-databases-do-this-first }

The containers start as new, empty databases. Repointing the URLs alone makes the console's session list, the dashboard's history and every `sensor_events` point seem to vanish: the data stays on the old machine. To keep it, migrate before you change the URLs.

1. **Export** on the old uber server:

    ```bash
    mongodump --uri "mongodb://localhost:27017" --db openmmla --archive=openmmla.archive
    influx backup ./influx-backup -t "<old admin token>"
    ```

2. **Import** on `uber-server`, after copying both there:

    ```bash
    docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
      mongorestore --archive --db openmmla < openmmla.archive

    docker cp ./influx-backup "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/influx-backup
    docker compose -f docker/docker-compose.infra.yml exec influxdb \
      influx restore /tmp/influx-backup --full
    ```

3. **Enter the old token** in the console: `influx restore --full` also restores the old instance's tokens. To keep the new token, restore the data alone with `--bucket mmla-data`.

Not migrating is fine too: the old machine keeps the old data, apart from the new.

### Backups

The data lives only in named volumes. `docker compose down -v`, `docker volume rm` and "reset it" commands delete them for good. Run these exports regularly and keep the output outside Docker:

```bash
docker compose -f docker/docker-compose.infra.yml exec -T mongodb \
  mongodump --archive --db openmmla > /backup/openmmla-$(date +%F).archive

docker compose -f docker/docker-compose.infra.yml exec influxdb \
  influx backup /tmp/backup -t "<token>"
docker cp "$(docker compose -f docker/docker-compose.infra.yml ps -q influxdb)":/tmp/backup /backup/influx-$(date +%F)
```

Do not `down` the stack while a collection runs. Both databases get 60 s to stop (`stop_grace_period`), but end the session first, then stop the stack.

### Share a host with another project { #sharing-a-host-with-another-project }

When the machine already runs another project's InfluxDB or MongoDB container, even one bound to `127.0.0.1:8086` only, this stack's `0.0.0.0:8086` does not come up: the kernel refuses a wildcard bind and a specific bind on one port. Leave their containers running and change this stack's host ports:

```bash
# docker/.env
INFLUXDB_PORT=8087
MONGODB_PORT=27018
```

Then put the ports in the System Settings URLs (`http://uber-server:8087`, `mongodb://uber-server:27018`); the console's status probe reads the port from the URL. Inside the containers the ports stay 8086 and 27017, so the healthchecks and the data are unaffected.

!!! warning "Do not reuse the other project's instance"
    It is usually bound to loopback only, out of reach of other machines, and has authentication on, so the password would go into `MongoDB.url` in plain text. If it runs `influxdb:latest`, one `pull` turns it into InfluxDB 3 and takes your data with it.

### Reach the databases from their own host { #when-the-database-host-has-to-reach-its-own-databases }

With `INFRA_BIND_ADDRESS` pinned to one interface, such as a tailnet address, the database host cannot reach its own databases by name. On Debian and Ubuntu the host's own name resolves to `127.0.1.1` on the host itself, nothing is published there, and a local process, such as the dashboard backend, gets `Connection refused` from a URL that works from every other machine. Publish the same ports on that address as well:

1. **Find the address** on the database host:

    ```bash
    getent hosts "$(hostname)"           # 127.0.1.1 on Debian/Ubuntu
    ```

2. **Set it** in `docker/.env` on that host only:

    ```bash
    INFRA_BIND_ADDRESS=<tailnet-ip>      # the tailnet interface
    INFRA_LOOPBACK_ADDRESS=127.0.1.1     # what the host's own name resolves to there
    ```

3. **Recreate** the two databases. `up -d` recreates, `restart` does not; naming the services leaves MediaMTX and any recording it writes alone, and the volumes are not affected:

    ```bash
    docker compose -f docker/docker-compose.infra.yml up -d influxdb mongodb
    ```

4. **Recreate MediaMTX between sessions**, with `up -d mediamtx`. It publishes its API and playback ports on that address too, for the dashboard on this machine, and a recreate closes every stream for a moment.

Every config keeps the host name: `http://uber-server:8087` resolves to the tailnet address on the other machines and to `127.0.1.1` on `uber-server`, and both are published. System Settings syncs one value everywhere, and the dashboard can move to another machine without a config change. A `localhost` URL is right on the database host only, so it would have to be pinned there with `SystemServicesOverride`, which also keeps token rotations from reaching that file.

!!! warning
    Leave `INFRA_LOOPBACK_ADDRESS` empty while `INFRA_BIND_ADDRESS` is `0.0.0.0`: the kernel refuses a wildcard bind and a specific bind on one port, and the containers do not start. Exposure is unchanged by it: a `127.x` address is reachable from the machine itself only.

### MongoDB authentication { #mongodb-authentication-optional-strongly-recommended }

MongoDB runs without authentication by default, as the bare-metal setup does, and on Linux Docker's NAT rules run before ufw, so `ufw deny 27017` does not block a published port. Use it only on a trusted lab network, publish on one interface (`INFRA_BIND_ADDRESS=<LAN IP>`), add iptables rules to the `DOCKER-USER` chain, or switch authentication on. To switch it on, set both variables before the first start:

```bash
export MONGO_ROOT_USER=openmmla
export MONGO_ROOT_PASSWORD="<password>"
docker compose -f docker/docker-compose.infra.yml up -d
```

With authentication on, the URL is `mongodb://<user>:<pass>@uber-server.local:27017/?authSource=admin`; without `authSource=admin` it fails.

!!! warning
    - The variables take effect only while the `mongodb-data` volume is empty. Set once the volume holds data, they give `--auth` with no users, and nobody can connect. **Do not delete the volume** then: clear both variables and `up -d` again to return to no authentication with the data untouched, or create the user through the container's localhost exception:
      `docker compose -f docker/docker-compose.infra.yml exec mongodb mongosh admin --eval 'db.createUser({user:"openmmla",pwd:"<password>",roles:["root"]})'`
    - `MongoDB.url` is not an encrypted field (only keys such as token, password or secret are), so a URL with a password sits in plain text in `config/system_services.yml` and in every synced `pipelines/*/config.yml`. Those files are gitignored.

## How the console runs the stacks { #how-the-tui-uses-these-stacks }

| Card | Start | Stop | Logs |
|---|---|---|---|
| **ASR Server**, **VFA Server** (`Launcher → Pipelines`) | `docker compose -f docker/docker-compose.asr.yml up -d --build <services>`, or `docker-compose.vfa.yml`; AudioInferer picks its container from `backend` | `docker compose ... down` | `docker compose ... logs --tail 40` |
| **InfluxDB**, **MongoDB**, **Stream Server (MediaMTX)** (`Launcher → System Services`), **Run mode** `docker` | `docker compose -f docker/docker-compose.infra.yml up -d <service>` | `... stop <service>`, not `down`, which would stop the other containers of the shared file | `... logs <service>` |

A remote host runs the same commands over SSH, in its repository directory. The AI server cards show `Running` and `(R)` in the tree while their ports answer. The system service cards probe the address in System Settings from the console's machine, whatever interface `INFRA_BIND_ADDRESS` binds, so a console outside the tailnet or behind a firewall shows them grey even when the pipeline machines connect ([TUI → System Services](tui/launcher/system-services.md#status)).

**Fetch Token**, on the InfluxDB card, reads the stack's admin token from the running container's `/etc/influxdb2/influx-configs`, which also holds a token InfluxDB generated itself, else from `docker/.env`, and warns when the URL's host and the host it read the token from differ ([Card controls](tui/launcher/system-services.md#card-controls)).

## Environment variables

The stacks read these from `docker/.env` ([Setup](#setup)), or from the shell that runs `docker compose`.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | empty | ASR stack: the Hugging Face token of an account that accepted the pyannote pipeline's terms, for `SpeechTranscriber.local.diarize` ([Diarization](pipelines/asr/speakers-and-diarization.md#diarize)); an `hf_token` in `pipelines/asr-server/config.yml` wins; read at every `up` |
| `OPENMMLA_KEY_DIR` | `~/.openmmla` | AI stacks: the folder whose `master.key` is mounted; leave it unset ([Mounts](#mounts)) |
| `INFLUXDB_INIT_ADMIN_TOKEN` | empty: InfluxDB generates one | the admin token, which goes into `InfluxDB.token`; first start only |
| `INFLUXDB_INIT_PASSWORD` | empty: the first start fails | the web UI's `admin` password, at least eight characters; first start only |
| `INFLUXDB_INIT_USERNAME`, `INFLUXDB_INIT_ORG`, `INFLUXDB_INIT_BUCKET` | `admin`, `admin`, `mmla-data` | must match `config/system_services.yml`; first start only |
| `MONGO_ROOT_USER`, `MONGO_ROOT_PASSWORD` | empty | both set switch on authentication on an empty volume ([MongoDB authentication](#mongodb-authentication-optional-strongly-recommended)) |
| `INFRA_BIND_ADDRESS` | `0.0.0.0` | the address the infra stack publishes its ports on; a LAN address keeps them off other interfaces |
| `INFRA_LOOPBACK_ADDRESS` | empty | a second address for processes on this machine: the database ports, and MediaMTX's API and playback ports, are published on it too ([Reach the databases from their own host](#when-the-database-host-has-to-reach-its-own-databases)) |
| `INFLUXDB_PORT`, `MONGODB_PORT` | `8086`, `27017` | host ports; keep the System Settings URLs in step |
| `MEDIAMTX_RTMP_PORT`, `MEDIAMTX_RTSP_PORT`, `MEDIAMTX_SRT_PORT`, `MEDIAMTX_API_PORT`, `MEDIAMTX_PLAYBACK_PORT`, `MEDIAMTX_WEBRTC_PORT` | `1935`, `8554`, `8890`, `9997`, `9996`, `8889` | MediaMTX's host ports; keep the Stream Server's ports in System Settings in step |
| `MEDIAMTX_STREAMS_DIR` | `../artifacts/streams/server` | where the server's recordings land, relative to `docker/` or absolute |
| `MEDIAMTX_WEBRTC_HOSTS` | empty | the addresses or names browsers reach this host at, comma-separated, for the dashboard's live video and sound ([Live video and sound in MediaMTX](dashboard/deploy.md#live-video-and-sound-in-mediamtx)) |

A changed port or address needs a recreate, which `up -d` does and `restart` does not.

## Mounts

| In the container | From the host | What it holds |
|---|---|---|
| `/project` | `pipelines/asr-server`, `pipelines/vfa-server` | `config.yml`, `temp/` and runtime logs; for the frame analyzer `weights/`, the pose weights of the [features endpoint](pipelines/vfa/pose-and-gaze.md#features-endpoint), fetched once at the first start, and with the tracker's [face check](pipelines/vfa/pose-and-gaze.md#appearance-checks) on, its ArcFace model under `weights/face/` |
| `/project/config/vfa`, read-only | `config/vfa` | the frame analyzer's action schema (`action_schemas.yml`), which its config refers to |
| model caches | named volumes `hf-cache`, `torch-cache`, `modelscope-cache`, `wespeaker-cache`, `deepface-cache` | shared, so a recreated container downloads nothing again: the transcriber's Whisper model at its first request and the frame analyzer's PaGE checkpoint in `hf-cache`, a Gaze-LLE checkpoint in `torch-cache`, the separator's MossFormer2 model in `modelscope-cache` (`/root/.cache/modelscope`), RetinaFace's weights in `deepface-cache` |
| `/root/.openmmla`, read-only | `~/.openmmla` of the user who runs `docker compose` (`OPENMMLA_KEY_DIR`) | the host's `master.key`, which decrypts the `ENC(...)` values of `config.yml` at startup |
| `/var/lib/influxdb2`, `/etc/influxdb2` | named volumes `influxdb-data`, `influxdb-config` | InfluxDB's data (`influxd.bolt` and the engine) and its config (`influx-configs`, from which the admin token can be recovered) |
| `/data/db`, `/data/configdb` | named volumes `mongodb-data`, `mongodb-config` | MongoDB's data; do not bind-mount `/data/db`, as WiredTiger needs real file locks |
| `/mediamtx.yml`, read-only | `pipelines/uber-server/mediamtx/mediamtx.yml` | MediaMTX's config, read when the container starts |
| `/streams/server` | `MEDIAMTX_STREAMS_DIR` | the server's recordings, plain fMP4 segments on the host ([Record on the server](streaming/recording.md#on-the-server)) |

The database containers do not mount `~/.openmmla`: the official images hold no OpenMMLA code and have nothing to decrypt.

??? info "Details: the master key of the AI services"
    - The key in `~/.openmmla` is the host's own, the one the console encrypts that host's configs with, so no key is copied from anywhere. Run the stack as the user the console's SSH profile logs in as, as the console does.
    - The console always uses `~/.openmmla/master.key` of that login. Leave `OPENMMLA_KEY_DIR` unset or point it at that folder: a key from anywhere else does not open what the console wrote, and the services get the `ENC(...)` strings themselves.
    - A service encrypts a plain-text secret in `config.yml` with that key and writes it back at startup; with no key there, it uses the secret as it is.
    - A host whose configs hold no encrypted value yet may have no key. The console makes it one with the first encrypted value it writes there, or run `openmmla crypto init` there.

## Troubleshooting { #known-caveats }

**A changed token, org, bucket or password in `docker/.env` does nothing.** The `DOCKER_INFLUXDB_INIT_*` values apply only on the first start against an empty `influxdb-data` volume; once `influxd.bolt` exists, setup is skipped without an error. Make a new token, or read the first one back:

```bash
docker compose -f docker/docker-compose.infra.yml exec influxdb influx auth create --org admin --all-access
docker compose -f docker/docker-compose.infra.yml exec influxdb cat /etc/influxdb2/influx-configs
```

!!! danger "Do not delete `influxd.bolt` to run the setup again"
    Setup then runs against the kept `/etc/influxdb2`, which already holds a config, most likely fails, and its failure path deletes the engine directory with `rm -rf`. All data is gone.

**`Bind for 0.0.0.0:8086 failed: port is already allocated`.** A bare-metal database or another project's container holds the port. Stop the first ([Setup](#setup)), or change this stack's ports for the second ([Share a host with another project](#sharing-a-host-with-another-project)).

**8086 or 27017 never opens, while `up -d` reported success.** The container restart-loops; `docker compose -f docker/docker-compose.infra.yml ps` and `... logs influxdb` (or `mongodb`) say why:

- an empty `INFLUXDB_INIT_PASSWORD` on the first start;
- only one of `MONGO_ROOT_USER` and `MONGO_ROOT_PASSWORD` set;
- exit code 132 from MongoDB: MongoDB 5.0 and later needs AVX on x86_64 (ARMv8.2-A or newer on arm64). Check with `grep -m1 -o avx /proc/cpuinfo` before going live.

**The containers are down after a reboot.** `INFRA_BIND_ADDRESS` names an address that comes up after Docker does, such as a tailnet IP, so a container started at boot cannot publish its ports. Docker's restart policy restarts only a container that ran and exited. Run `docker compose -f docker/docker-compose.infra.yml up -d` once the address is there; on a machine that runs the dashboard, `make autostart` in `pipelines/uber-server` makes every boot start them ([Start after a reboot](dashboard/deploy.md#start-after-a-reboot)).

**The dashboard on the database host cannot reach the databases** (`Connection refused`). See [Reach the databases from their own host](#when-the-database-host-has-to-reach-its-own-databases).

**The InfluxDB and MongoDB cards on a Mac stay grey after the move to `uber-server`.** The Mac's URLs still say `localhost`, so the cards probe the Mac's own ports. Do not press **Start** there: in `native` mode it runs `make influxdb` and `make mongodb`, which start empty local databases, and the cards turn green while the pipelines use `uber-server`. Stop those local services once the move is done.

**A service gets `ENC(...)` strings instead of secrets.** Its key folder holds no key the console wrote with ([Details: the master key](#mounts)). A stack started before `~/.openmmla` existed made that folder itself, owned by root, and nothing can put a key in it: run `sudo chown -R "$USER": ~/.openmmla`, then start the stack again.

**MediaMTX runs with an old `mediamtx.yml`, or records nowhere.** The container reads its config, and opens its record folder, when it starts. Recreate it after a change, or after its record folder was renamed or removed ([Streaming troubleshooting](streaming/troubleshooting.md#the-stream-server)).
