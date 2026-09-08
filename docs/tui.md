# Management Console (TUI)

`mmla tui` opens a terminal console that configures, launches and monitors every OpenMMLA component, on this machine or on remote machines over SSH. It is the primary entry point: the old per-pipeline shell scripts are gone, and everything the console launches is also an `mmla` command you can run by hand (see the pipeline guides).

## Install and start

Install the [prerequisites](prerequisites.md) first. The console runs from a clone of the repository, because it edits the configs under `pipelines/` and `config/` and runs the compose files under `docker/`.

```bash
git clone https://github.com/ucph-ccs/OpenMMLA.git
cd OpenMMLA
conda create -n tui python=3.10 -y
conda activate tui
pip install -e '.[tui]'
mmla tui
```

- With an editable install the console finds the repository root from the installed package, so `mmla tui` works from any directory. The package is also on PyPI (`pip install "openmmla[tui]"`), but a non-editable install has to be started from the repository root, because the console reads `pipelines/`, `config/` and `docker/` from the current directory in that case.
- Press `q` to quit. Four tabs sit at the top: **Environment**, **Launcher**, **Sessions**, **Status**.
- The first time a config is saved, a Fernet key is created at `~/.openmmla/master.key` (`0600`). Every value whose key is named `token`, `password`, `api_key`, `secret`, `secret_key` or `subscription_key` is stored as `ENC(...)` and decrypted by the services at startup. When the console copies a config with encrypted values to a remote host it copies the key there too. Back the key up; without it the encrypted values cannot be read.

## Hosts

Both the Environment tab and the Launcher have a **Host** selector: `Local`, or one of the SSH profiles defined under **System Settings → Hosts → SSH Profiles**. Every action then runs on that host, and every path is resolved as `<remote_project_path>/<same path relative to the repository root>`, so the remote machine needs a clone at the path the profile names. The `↻` button next to the selector tests every profile; a host that fails the test is shown as `(offline ✗)` and cannot be selected until it answers again. Profiles are re-tested every 30 seconds.

Once a session has been started, a remote host needs the same things a local one does for the components you launch there:

| You launch | The remote host needs |
|---|---|
| ASR / IPS / VFA bases, camera tools, MLLM Server | conda with the pipeline environment (create it from the Environment tab), tmux, the repository at `remote_project_path` |
| ASR Server / VFA Server, InfluxDB / MongoDB in `docker` mode | Docker Engine with compose, the NVIDIA container toolkit for the AI stacks, the user in the `docker` group, the repository (for `docker/`) |
| System services in `native` mode | the service installed with brew or apt, `make`, and `sudo` |
| Streams | `ffmpeg` and `tmux` |
| Collection Session | only `python3` and `ffmpeg`; the recorder code is pushed to `~/.openmmla/collection-runtime` automatically |

## Environment tab

A table of the conda environments OpenMMLA uses, on the selected host, with columns `Conda Env`, `Dep Group`, `Python`, `Status`, `Description`:

| Conda env | Extra in `pyproject.toml` | Python | Used by |
|---|---|---|---|
| `asr-base` | `asr-base` | 3.10 | ASR base station |
| `vfa-base` | `vfa-base` | 3.10 | VFA base station |
| `vfa-vllm` | `vfa-vllm-runtime` | 3.12 | local vLLM runtime for the MLLM Server card |
| `ips-base` | `ips-base` | 3.10 | IPS base station |
| `uber-base` | `uber-base` | 3.10 | analytics and session exports |
| `uber-server` | `uber-server` | 3.10 | dashboard, Celery worker, Nginx render step |
| `tui` | `tui` | 3.10 | this console |

The ASR and VFA server services have no conda environment any more; they run as Docker images (see the [Docker guide](docker.md)).

`Status` is computed live from the `[project.optional-dependencies]` groups in `pyproject.toml`: `Missing` (no such env), `Partial: pkg, pkg +N` (env exists, packages missing), `Ready`. The buttons run these commands, locally or over SSH:

| Button | Command |
|---|---|
| **Create Env** | `conda create -n <env> python=<version> -y` |
| **Install Deps** | `conda run -n <env> pip install -e '.[<group>]'` in the repository |
| **Delete Env** | `conda env remove -n <env> -y`, after a second press to confirm |
| **Git Clone** | remote hosts only: `git clone <origin url> <remote_project_path>` |
| **Git Pull** | remote hosts only: `cd <remote_project_path> && git pull` |
| **Connect** / **Refresh** | test the host and re-read `conda env list` / `conda list` |

A command console at the bottom shows the output and accepts ad-hoc shell commands on the selected host.

## Launcher tab

The sidebar is a tree; selecting a leaf shows its form or service card on the right. Leaves that launch something carry markers explained by the legend above the tree, `[E] env  [C] config  (R) running`:

- `[E]`: the conda environment on the selected host. Green `Ready`, yellow `Partial`, red `Missing`. Not shown for services that run in Docker or as native system services.
- `[C]`: whether the pipeline's `config.yml` exists on this machine. Green present, red missing.
- `(R)`: the service is running, according to the last status probe.

```
OpenMMLA
├── System Settings
│   ├── Hosts:        SSH Profiles
│   ├── Study:        Experiments, Tasks
│   ├── Connections:  MongoDB, InfluxDB, MQTT, Redis, Gateway (Nginx), Dashboard (Flask)
│   └── Credentials:  Sudo (local admin)
├── System Services
│   ├── InfluxDB, MongoDB, Redis, Mosquitto, Nginx
│   ├── Dashboard (Flask)
│   └── Dashboard Worker (Celery)
├── Collection
│   └── Collection Session
├── Pipelines
│   ├── ASR:  ASR Base, ASR Server
│   ├── IPS:  IPS Base, IPS Camera Calibration, IPS Camera Sync
│   └── VFA:  VFA Base, VFA Server, MLLM Server
└── Session Control
```

### Service cards

Every launchable leaf opens a card with the service name, its conda env and launch type, a status line, its parameters, and the buttons **Start**, **Stop**, **Logs** and **Refresh**. Bases and camera tools open in terminal windows and therefore show `Interactive (runs in its own terminal)` and no Stop button: close them from their own window. Cards for a pipeline also have a **Config** tab that edits `pipelines/<pipeline>/config.yml` on the selected host, with **Save** writing it back (and, on a remote host, copying it there). Output of every action goes to the command console at the bottom of the tab.

Before a Start the console checks that `config.yml` exists on the target host, that the sections managed by System Settings are up to date, and that the conda environment exists. On a remote host whose config is out of date, the first Start only pushes the current System Settings and prints `Relaunch <service> once the sync above completes.`; press Start again. This is by design.

### System Settings

These forms always edit this machine's project; the Host selector is disabled while one is open.

**SSH Profiles**: one entry per remote machine, saved to `config/ssh_profiles.yml` (gitignored; `config/ssh_profiles_template.yml` is the tracked example). Fields: `Profile Name`, `Host`, `User`, `Port` (22), `Password` (needs `sshpass` on this machine; leave empty for key auth), `Key Path` (for example `~/.ssh/id_ed25519`), `Remote Project Path` (default `~/OpenMMLA`, must be the repository root on that machine). **Test Connection** checks the login. Passwords are stored encrypted.

**Experiments**: the study registry in `config/experiments.yaml` (gitignored, template `config/experiments_template.yaml`). An experiment has an id, a title, a status and a task type; each participant has a `group_id`, the `tag_id` of the AprilTag they wear and a short appearance `description`. The active experiments and their groups populate the **Experiment Group** dropdown on the base and collection cards, and the descriptions feed the VFA prompts.

**Tasks**: the task definitions in `config/tasks/*.yaml`, edited as raw YAML.

**Connections**: `MongoDB`, `InfluxDB`, `MQTT`, `Redis`, `Gateway (Nginx)` and `Dashboard (Flask)`. Saving writes `config/system_services.yml` and copies the section into every pipeline `config.yml` that carries it; the pipeline Config tabs then show those fields read-only with a `managed in System Settings` note. **Sync to Remote** under the form pushes the section into the configs on one remote host. A pipeline that must keep its own value lists the section under `SystemServicesOverride:` in its `config.yml`, or presses `Override here` at the bottom of that section in its Config tab. The fields and defaults are listed in [System Services](system_services.md#pointing-the-pipelines-at-the-services).

**Sudo (local admin)**: the sudo password of this machine, stored encrypted. Native Start/Stop of the system services run `make` with `sudo`, and the console types this password when the prompt appears (at most three times per command). Remote sudo prompts use the SSH profile's password instead.

### System Services

One card per system service: **InfluxDB**, **MongoDB**, **Redis**, **Mosquitto**, **Nginx**, **Dashboard (Flask)** and **Dashboard Worker (Celery)**. Installation is described in [System Services](system_services.md).

- **Status** is a TCP probe from this machine to the address configured under System Settings, which is the path the pipelines take. The card description names the probed address. A `localhost` address names no particular machine, so it is probed on the selected host instead (over SSH for a remote host). The Celery worker is detected by its tmux session.
- **Start / Stop** on Redis, Mosquitto and Nginx run `make -C pipelines/uber-server <service>` / `stop-<service>` on the selected host; the make targets also rewrite the service's listener config to bind on all interfaces. Dashboard (Flask) runs `make flask DASHBOARD_PORT=<port>`, which opens a tmux session named `flask`; the worker opens one named `celery`.
- **Run mode** (InfluxDB and MongoDB only): `docker`, the default, runs `docker compose -f docker/docker-compose.infra.yml up -d | stop | logs <service>`; `native` runs the make targets like the other cards. Choose `native` on a machine that still runs brew or systemd databases, otherwise Start brings up a container next to them. The choice is remembered per host for the current console session.
- **Fetch Token** (InfluxDB, docker mode): reads the admin token of the compose stack on the selected host, from the running container or from `docker/.env`, and stores it encrypted as `InfluxDB.token`. Only the first and last four characters are shown.
- **Logs**: compose logs in docker mode; brew or journald logs for native services; the tmux pane for the dashboard.
- Nginx and the dashboard need the `uber-server` conda environment on the host, because the Makefile renders the Nginx config with it and runs gunicorn and Celery inside it.

### Collection

**Collection Session** records raw audio and video with FFmpeg for later replay through the pipelines with `source: file`. The card has an **Audio** and a **Video** tab, each with a recorder count, **Start Audio** / **Start Video**, **Stop**, **Stop All Hosts**, **Logs**, **Refresh**, **Download** and **Delete Remote**, plus these fields:

- **Session**: an existing session id, or `Create MongoDB Session` to mint one from the selected **Experiment Group** (`<experiment>/<group>`).
- **Output Root** and **Host Label**: stored per host. The default root, `artifacts` locally and `~/artifacts` remotely, puts files under `artifacts/<session>/collection/<host label>/{audio,video}/`; any other root puts them under `<root>/<session>/{audio,video}/`.

Devices and formats are chosen interactively in the recorder terminal; the encoding defaults follow the host platform (`avfoundation` on macOS, `alsa` and `v4l2` on Linux). A recorder opens as a terminal window per instance and writes `manifest.yml` and `manifest.json` next to the files, with the shared `initial_sync_time` and ready-made `file_dir` values for the pipelines.

One session is usually recorded by several machines: set the tab, the recorder count and the session once, then switch **Host** and press Start on each machine. The session-scoped fields follow you across hosts, so every machine records into the same session. **Stop All Hosts** stops every recorder of the session on every host, then marks the session as ended in MongoDB. **Download** copies a remote host's recordings into the local `artifacts/<session>/collection/<host label>/` and merges the manifests; **Delete Remote** removes them from the remote machine after a second confirming press.

### Pipelines

The ASR, IPS and VFA base cards share one shape:

- **Launch** tab: how many bases and synchronizers (and IPS visualizers) to start, the **Session** (an existing id or `Create MongoDB Session` from the **Experiment Group**), the mode for ASR and VFA (`live`, `capture`, `analyze`) and the pipeline's toggles. Start opens one terminal window per instance; each base asks which entry of the `Bases` list it is, then waits for the START signal.
- **Config** tab: the pipeline's `config.yml`. `+ Add Stream` adds a `Streams` entry and, for ASR, `+ Add Base` adds a device type under `Base`. `Bases` entries are edited with dropdowns filled from the config (calibrated cameras, base types, sources, files in `file_dir`).
- **Streams** tab: the `Streams` entries of the config, with **Start**, **Stop**, **Logs**, **Probe**, **Start All** and **Stop All**. A stream with an `ssh_profile` is started as an FFmpeg process inside a tmux session named `mmla-stream-<name>` on that host; a stream without one is shown as `External` and only pulled from.
- **Transform Matrix** tab (IPS only): the `transformation_matrices*.json` files produced by camera sync, editable as JSON, with **Sync to Remote** to copy them to a base station.

The server cards are different:

- **ASR Server** and **VFA Server** run the AI services with docker compose on the selected host (`docker/docker-compose.asr.yml`, `docker/docker-compose.vfa.yml`). The Launch tab lists the sub-services from the server config with a `true`/`false` toggle each; Start runs `docker compose up -d --build` for the selected ones, Stop runs `down`, Logs tails the containers. The VFA Server card also has a **Prompts** tab (the templates under `pipelines/vfa-server/prompts/`) and an **Action Schema** tab (`config/vfa/action_schemas.yml`).
- **MLLM Server** starts `vllm serve` from `config/mllm_server.yml` in the `vfa-vllm` environment, inside a tmux session named `mllm-server`, for VFA setups that use a local vision-language model.
- **IPS Camera Calibration** and **IPS Camera Sync** run the interactive calibration and multi-camera synchronisation tools; the calibration card also lists the captured calibration images per camera.

The pipeline guides walk through each one end to end: [ASR](pipelines/asr.md), [IPS](pipelines/ips.md), [VFA](pipelines/vfa.md).

### Session Control

Bases and synchronizers block after start-up until they receive a START signal for their session. This leaf lists the sessions, lets you tick the pipelines, and sends **START** or **STOP** over Redis. STOP also marks the session as ended in MongoDB. The Redis and MongoDB addresses come from System Settings.

A typical run is therefore: start the AI servers and the system services, start the bases and synchronizers on every base station (each picks its `Bases` entry), then send START from here once every window reports it is waiting, and STOP when the session is over.

## Sessions tab

A table of sessions with `Session ID`, `Experiment`, `Group`, `Status`, `Started` and `Source`, merged from the MongoDB `sessions` collection and, on the local host, from the `artifacts/<id>/` and `collection/<id>/` directories and their manifests. The addresses come from `config/system_services.yml`; with a remote host selected, its pipeline config is read over SSH instead.

- **Export Measurements** writes one JSON file per event type (speaker recognition and transcription, IPS translation, rotation and relation, VFA actions) from InfluxDB into `artifacts/<session>/measurements/`, plus a text transcript.
- **Export Visualizations** and **Export All** additionally render the ASR diarization and speaking-interaction plots and the IPS trajectories, heatmap and physical-interaction network into `artifacts/<session>/analysis/visualizations/`. VFA has no visualizations yet.
- **Delete Session** removes the InfluxDB measurements and the MongoDB document, keeping local artifacts. **Delete Artifacts** removes the local directory, keeping the database records. Both need a second press.

The exports need the `uber-base` extra (analytics) in the console's environment.

## Status tab

One row per known service with `Service`, `Host`, `Status`, `Port`, `Session` and `Started`, refreshed every five seconds while the tab is open (**Refresh** also probes the remote hosts). For InfluxDB, MongoDB, Redis, Mosquitto, Nginx and the dashboard the Host column is the machine they are configured on in System Settings and the status is a TCP probe of that address. The Flask dashboard, the Celery worker and any other tmux session on this machine are listed by tmux session name. **View Logs** shows the selected row's logs: container or brew/journald logs for system services, the tmux pane otherwise, over SSH when the row's host matches an SSH profile.

The six ASR service rows still expect tmux sessions named after the services. With the dockerized ASR stack they read `Stopped` even while the containers answer on their ports; use the ASR Server card for the real state.

## Files the console writes

| Path | Content |
|---|---|
| `config/system_services.yml` | System Settings connections and the sudo password (secrets encrypted); tracked by git |
| `config/ssh_profiles.yml` | SSH profiles (gitignored) |
| `config/experiments.yaml` | experiments and participants (gitignored) |
| `config/tasks/*.yaml`, `config/vfa/action_schemas.yml`, `config/mllm_server.yml` | task definitions, VFA action schema, MLLM Server settings |
| `pipelines/*/config.yml` | pipeline configs, with the shared sections mirrored from System Settings (gitignored) |
| `pipelines/ips-base/camera_calib/`, `camera_sync/` | calibration images and transform matrices |
| `artifacts/<session>/` | recordings, exported measurements and visualizations, manifests |
| `~/.openmmla/master.key` | encryption key; `~/.openmmla/streams/` holds stream start times, `~/.openmmla/collection-runtime/` the pushed recorder code on remote hosts |
