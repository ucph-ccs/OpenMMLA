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

In the Launcher every leaf of the tree has **its own host**, because the parts of a deployment rarely share one machine: the AI servers sit on a GPU server, the bases on the base stations, the databases somewhere else again.

- A **pipeline, collection or MLLM Server** card opens on the host it was last pointed at, `Local` the first time. The choice is kept per card in `config/launcher_hosts.yml`, across restarts. If that host is offline or its profile is gone, the card opens on `Local` and says so in the log; the remembered host is kept for next time.
- A **System Services** card opens on the machine System Settings put the service on: the address (`InfluxDB.url`, `Gateway.host`, ...) is matched against this machine (its names and its addresses) and against the saved SSH profiles. This is a default, not a lock: pick another host to start or stop the service somewhere else for once. The card then says that the pipelines are configured for another machine and reports the chosen host's own port, and it is back on the configured machine the next time it is opened; to move a service for good, change its address in System Settings. An address that matches no profile, or a profile that is offline, opens the card on `Local` with the reason in the log. A `localhost` address names no machine, so that card remembers its host like a pipeline card.
- The **Connections** forms of System Settings have a selector of their own, shared between them: it picks whose settings the forms show and save, and starts on `Local` with every new console. The other System Settings (SSH profiles, experiments, tasks, sudo) only mean something on the machine the console runs on, and **Session Control** has no host at all; they show a note with the reason instead of the selector.

The `[E]` and `(R)` markers in the tree follow the same rule: each leaf is checked on its own host, and a system service at the address the pipelines use, so the sidebar describes the deployment and agrees with every card that sits where it opened.

A remote host has to offer a **POSIX shell** over SSH: Linux, macOS or a Raspberry Pi. Everything the console does there goes through `bash -lc`, tmux, conda and the usual `test` / `cat` / `mkdir -p`, and the recorders use POSIX file locks and signals. A Windows machine answering with `cmd.exe` or PowerShell (Windows' own OpenSSH server) is recognised the first time it is picked: the selector goes back to where it was, the log says why, and the host is labelled `(Windows: not supported ✗)` from then on. With WSL2 as that machine's SSH shell it answers as Linux and works for server-side services (the AI stacks, the databases); cameras and microphones are not visible inside WSL, so it cannot record a collection session.

Once a session has been started, a remote host needs the same things a local one does for the components you launch there:

| You launch | The remote host needs |
|---|---|
| ASR / IPS / VFA bases, camera tools, MLLM Server | conda with the pipeline environment (create it from the Environment tab), tmux, the repository at `remote_project_path` |
| ASR Server / VFA Server, InfluxDB / MongoDB in `docker` mode | Docker Engine with compose, the NVIDIA container toolkit for the AI stacks, the user in the `docker` group, the repository (for `docker/`) |
| System services in `native` mode | `make` and `sudo`; Redis, Mosquitto and Nginx are installed by Start when the host has none (brew on macOS, apt on Debian and Ubuntu), InfluxDB and MongoDB natively need their vendor repositories (see [System Services](system_services.md)) |
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

- `[E]`: the conda environment on that leaf's host. Green `Ready`, yellow `Partial`, red `Missing`. Not shown for services that run in Docker or as native system services. A host's envs are read once and kept; the markers catch up when the Environment tab creates, removes or fills an env on that host, and when the Launcher comes back into view a minute or more later.
- `[C]`: whether the config file the leaf needs exists on that leaf's host: the pipeline's `config.yml`, and for the Gateway, Dashboard and Stream Server cards `nginx/config.yml`, `dashboard/flask-backend/config.yml` and `mediamtx/mediamtx.yml` under `pipelines/uber-server`. Green present, red missing. A remote host's files are looked up over SSH with the status probe (when the Launcher opens, when a host is picked or comes back online, after a Start or Stop), and a Save or Sync to Remote that copies one there turns its marker green at once; until a host has answered, its leaves show no `[C]`.
- `(R)`: the service is running, according to the last status probe.

```
OpenMMLA
├── System Settings
│   ├── Hosts:        SSH Profiles
│   ├── Study:        Experiments, Tasks
│   ├── Connections:  InfluxDB, MongoDB, Redis, MQTT (Mosquitto),
│   │                 Gateway (Nginx), Stream Server (MediaMTX), Dashboard (Flask)
│   └── Credentials:  Sudo (local admin)
├── System Services
│   ├── InfluxDB, MongoDB, Redis, MQTT (Mosquitto)
│   ├── Gateway (Nginx), Stream Server (MediaMTX)
│   └── Dashboard (Flask), Dashboard (Celery)
├── Collection
│   └── Collection Session
├── Pipelines
│   ├── ASR:  ASR Base, ASR Server
│   ├── IPS:  IPS Base, IPS Camera Calibration, IPS Camera Sync
│   └── VFA:  VFA Base, VFA Server, MLLM Server
└── Session Control
```

A system service has one name everywhere the console shows it (its card, the sidebar, the Status tab, the log): `Role (Product)`, where the role is the **Connections** form that holds its address. `Gateway (Nginx)` and `Stream Server (MediaMTX)` each have a form of their own, since the load balancer and the stream server need not share a machine; `Dashboard (Flask)` and its worker `Dashboard (Celery)` take their host from `Dashboard (Flask)`, and the broker behind the `MQTT` section is `MQTT (Mosquitto)`. The two groups list them in the same order. The rest of this page uses the product name alone where the text is about the program (`make nginx`, the MediaMTX ports).

### Service cards

Every launchable leaf opens a card with the service name, its conda env and launch type, a status line, its parameters, and the buttons **Start**, **Stop**, **Logs** and **Refresh**. Bases and camera tools open in terminal windows and therefore show `Interactive (runs in its own terminal)` and no Stop button: close them from their own window. Cards for a pipeline also have a **Config** tab that edits `pipelines/<pipeline>/config.yml` on the selected host, with **Save** writing it back (and, on a remote host, copying it there).

Output of every action goes to the command console at the bottom of the tab. It is one transcript for the whole tab, since a build or a remote stop keeps printing after you move to another card; a divider such as `── Stream Server (MediaMTX) · Local ──` is drawn before the first line that belongs to another card or host.

Before a Start the console checks that `config.yml` exists on the target host, that the sections managed by System Settings are up to date, and that the conda environment exists. The Gateway (Nginx), Dashboard (Flask) and Dashboard (Celery) cards need a `config.yml` too (`pipelines/uber-server/nginx/` and `pipelines/uber-server/dashboard/flask-backend/`, gitignored like every config, so a freshly pulled host has none): Start stops with a note when the card's host lacks it. Write it there with **Save** on that card's Config tab, or copy this machine's with Host = `Local` and **Sync to Remote**. On a remote host whose config is out of date, the first Start only pushes the current System Settings and prints `Relaunch <service> once the sync above completes.`; press Start again. This is by design.

### System Settings

Every machine's services read **that machine's** settings: its pipeline `config.yml` files and, on top of them, its `config/system_services.yml` when it has one (see [System Services](system_services.md#pointing-the-pipelines-at-the-services)). The **Connections** forms therefore have a Host selector. On `Local` they edit this machine's project. On another host they read that machine's settings over SSH and show where the values come from: its own `config/system_services.yml`, else its pipeline configs, else the defaults. **Save** then writes the section on that machine, into its settings file (created if it has none) and into its pipeline configs. Flip the selector to check what a host will really connect to before a session.

A Start on a remote host keeps its pipeline configs in step first: with that host's own settings where it has them, section by section, and with this machine's everywhere else. When a host's own settings differ from this machine's, the first Start there says which sections do. Write addresses that are true from every machine (a host name, not `localhost`) whenever more than one machine is involved.

The remaining forms are this machine's alone, and say why in the Host bar: SSH profiles are the machines this console can reach, a session takes its participants to every host through its MongoDB document, and the sudo password is this machine's (a remote host uses the password of its SSH profile).

**SSH Profiles**: one entry per remote machine, saved to `config/ssh_profiles.yml` (gitignored; `config/ssh_profiles_template.yml` is the tracked example). Fields: `Profile Name`, `Host`, `User`, `Port` (22), `Password` (needs `sshpass` on this machine; leave empty for key auth), `Key Path` (for example `~/.ssh/id_ed25519`), `Remote Project Path` (default `~/OpenMMLA`, must be the repository root on that machine). **Test Connection** checks the login and, when a password is stored, the password on its own: a key in the host's `authorized_keys` logs in whatever the password says, while `sudo` on that host asks for the account's real one, so a placeholder left there passes the login and fails the first Start that needs `sudo`; the result says which. Passwords are stored encrypted.

**Experiments**: the study registry in `config/experiments.yaml` (gitignored, template `config/experiments_template.yaml`). An experiment has an id, a title, a status and a task type; each participant has a `group_id`, the `tag_id` of the AprilTag they wear and a short appearance `description`. The active experiments and their groups populate the **Experiment Group** dropdown on the base and collection cards, and the descriptions feed the VFA prompts.

**Tasks**: the task definitions in `config/tasks/*.yaml`, edited as raw YAML.

**Connections**: `InfluxDB`, `MongoDB`, `Redis`, `MQTT (Mosquitto)`, `Gateway (Nginx)`, `Stream Server (MediaMTX)` and `Dashboard (Flask)`. The Stream Server form is read by consoles only (it places and probes the MediaMTX card, and completes a stream written as a path, `ips/cam-1`, into the URLs of its `Streams` entry, which is what streams and bases use), so it has no Host selector. Saving it with another address moves the stream URLs of the local pipeline configs that named the old one. Its **Sync to Remote** gives another machine the address (its own `config/system_services.yml`, created if it has none, which a console there reads) and the `Streams` entries of the local pipeline configs (the URLs it completed, which the bases there pull), and leaves the rest of that machine's configs alone. Saving writes `config/system_services.yml` and copies the section into every pipeline `config.yml` that carries it; the pipeline Config tabs then show those fields read-only with a `managed in System Settings` note. With the selector on `Local`, **Sync to Remote** under the form is the shortcut for "give that host this machine's values": it copies the section into the pipeline configs of one remote host that carry it, and into its own `config/system_services.yml`, created when it has none, without a trip through that host's form; its services and a console there then say what this one says. A section no pipeline config carries (Dashboard: the console places and probes the Dashboard cards with it and passes the port to `make flask`) goes into that file alone. A `localhost` value is copied as it is and then means the remote machine itself: machines that share one service need its real host name here. A pipeline that must keep its own value lists the section under `SystemServicesOverride:` in its `config.yml`, or presses `Override here` at the bottom of that section in its Config tab. The fields and defaults are listed in [System Services](system_services.md#pointing-the-pipelines-at-the-services).

**Sudo (local admin)**: the sudo password of this machine, stored encrypted and never copied to another host (there is no Sync to Remote on this form). Native Start/Stop of the system services run `make` with `sudo`, and the console types this password when the prompt appears (at most three times per command). Remote sudo prompts use the SSH profile's password instead.

### System Services

One card per system service: **InfluxDB**, **MongoDB**, **Redis**, **MQTT (Mosquitto)**, **Gateway (Nginx)**, **Stream Server (MediaMTX)**, **Dashboard (Flask)** and its worker **Dashboard (Celery)**. Installation is described in [System Services](system_services.md).

- **Host**: opens on the machine the service's address in System Settings names (`InfluxDB.url`, `MongoDB.url`, `Redis.host`, `MQTT.host`, `Gateway.host` for Nginx, `StreamServer.host` for MediaMTX, `Dashboard.host` for the dashboard and its worker), see [Hosts](#hosts). To run a service somewhere else for good, change its address in System Settings; the card follows.
- **Status** is a TCP probe from this machine to the address configured under System Settings, which is the path the pipelines take. The card description names the probed address. A `localhost` address names no particular machine, so it is probed on the card's selected host instead (over SSH for a remote host), and so is a card that was moved off the configured machine: it reports the host it is on, while the sidebar marker keeps following the configured address. The Celery worker is detected by its tmux session. MediaMTX counts as running only when its RTMP **and** RTSP ports answer: an Nginx built with the RTMP module, the gateway of earlier versions, holds 1935 too, and would otherwise read as a MediaMTX that Stop can never find. When only one of the two answers, Refresh, Start and Stop say so in the log.
- **Start / Stop** on Redis, Mosquitto and Nginx run `make -C pipelines/uber-server <service>` / `stop-<service>` on the card's host; the make target installs the package first when that host has none (brew on macOS, apt on Debian and Ubuntu; `make` runs on the card's host, so it sees that host's platform) and rewrites the service's listener config to bind on all interfaces. Dashboard (Flask) runs `make flask DASHBOARD_PORT=<port>`, which opens a tmux session named `flask`; the worker opens one named `celery`. On a remote host the same targets run in the command session below the card, as on Local, where the shell has conda; a `sudo` prompt there is answered with the SSH profile's password (the login password is what `sudo` asks for), on Local with the Sudo password of System Settings.
- **Run mode** (InfluxDB, MongoDB and MediaMTX): `docker`, the default, runs `docker compose -f docker/docker-compose.infra.yml up -d | stop | logs <service>`; `native` runs the make targets like the other cards. Choose `native` on a machine that still runs brew or systemd databases, otherwise Start brings up a container next to them. The choice is remembered per host for the current console session. MediaMTX in `native` mode runs `make mediamtx`, a tmux session named `mediamtx` around the installed binary.
- **Fetch Token** (InfluxDB, docker mode): reads the admin token of the compose stack on the selected host, from the running container or from `docker/.env`, and stores it encrypted as `InfluxDB.token`. Only the first and last four characters are shown.
- **Config** tab (Stream Server): the `mediamtx.yml` of the card's host as text, comments included, with a **Server-side recording** switch for `pathDefaults.record` and a **Keep recordings for** choice for `pathDefaults.recordDeleteAfter` on top (three days as shipped; MediaMTX deletes a segment that long after it began, so a session's footage has to be exported before then, and `for ever` keeps everything). Both change the text in the editor, and Save writes it. MediaMTX reloads the file when it changes; Stop and Start the card if a change does not show. Its address and ports for the console are under System Settings → Stream Server (MediaMTX).
- **Recordings** tab (Stream Server): what the server holds, path by path: the number of ten-minute segments, the first and the last, and their size on disk (read over a shell on the card's host, SSH for a remote one), with the free space of that disk and the retention in force above the table. **Delete Path** removes every segment of the selected path, **Delete Older Than** every segment of every path that began before the chosen age; both go through the server's API, so they work for a docker and a native run alike, and both need a second press. Exporting a session's footage stays under Sessions: this tab is the server's inventory, not a session's.
- **Logs**: compose logs in docker mode; brew or journald logs for native services; the tmux pane for the dashboard.
- Nginx and the dashboard need the `uber-server` conda environment on the host, because the Makefile renders the Nginx config with it and runs gunicorn and Celery inside it. MediaMTX, the databases and the brokers need none (their cards say `env: none needed`).

### Collection

**Collection Session** records raw audio and video with FFmpeg for later replay through the pipelines with `source: file`. The card has an **Audio** and a **Video** tab, each with a recorder count, **Start Audio** / **Start Video**, **Stop**, **Stop All Hosts**, **Logs**, **Refresh**, **Download** and **Delete Remote**, plus these fields:

- **Session**: an existing session id, or `Create MongoDB Session` to mint one from the selected **Experiment Group** (`<experiment>/<group>`). A recording always belongs to a session MongoDB knows: an id that MongoDB does not have (it was deleted, or it is only known from artifacts on disk) is registered again under the selected Experiment Group when you press Start, and a session deleted in the Sessions tab is no longer offered here.
- **Output Root** and **Host Label**: stored per host. The default root, `artifacts` locally and `~/artifacts` remotely, puts files under `artifacts/<session>/collection/<host label>/{audio,video}/`; any other root puts them under `<root>/<session>/{audio,video}/`.

Each tab starts at one recorder; step the count down to zero to skip that role on this host. Devices and formats are chosen interactively in the recorder terminal; the encoding defaults follow the host platform (`avfoundation` on macOS, `alsa` and `v4l2` on Linux). A recorder opens as a terminal window per instance and writes `manifest.yml` and `manifest.json` next to the files, with the shared `initial_sync_time` and ready-made `file_dir` values for the pipelines. The card reads `Running` for as long as a recorder process is alive on the selected host, whichever session it records; **Logs** lists the live recorders (role, session, pid), since each of them prints into its own terminal window.

One session is usually recorded by several machines: set the tab, the recorder count and the session once, then switch **Host** and press Start on each machine. The session-scoped fields (Session ID, Experiment Group, recorder counts) follow you across hosts, so every machine records into the same session: the first Start with `Create MongoDB Session` mints the id, and every host after it shows that id already selected. Output Root and Host Label are kept per host. What to know about it:

- The id is remembered by this console while it runs. After a restart, or from a second console, the card opens on `Create MongoDB Session` again; pick the running session from the list to join it. Creating a session for an Experiment Group that still has an active one says so in the log.
- A host that is recording wins over the id that follows you: its card opens on the session its own recorders belong to, and the log says so. This is what two groups recording at the same time on different hosts need (each group is a session of its own): coming back from the other group's host, Stop and Download act on this host's session and not on the one that was started last. A host that records nothing is still offered the id that travels, to join that take. A session picked by hand while the host records (to download an older one) is left alone until the recording on that host changes.
- **Stop** on one host leaves the session active while another host this console started it on is still recording; only the last Stop, or **Stop All Hosts**, marks it ended in MongoDB.
- When the session is over (the last Stop, or Stop All Hosts), the card goes back to `Create MongoDB Session` on every host, so the next Start is a new take. Stop, Download and Delete Remote do not need the id on the card: they fall back to the session last recorded on their host. Picking an ended session by hand and pressing Start is held back once, with the choice spelled out: `Create MongoDB Session` for a new take, or Start again to record into it all the same (it becomes active again). **Stop All Hosts** stops every recorder of the session on every host (this machine, the hosts it was started on, and every other reachable profile; Windows hosts are left out), then marks the session as ended in MongoDB. **Delete Remote** removes a host's recordings from the remote machine after a second confirming press.

#### Downloading a session

**Download** copies a remote host's recordings into the local `artifacts/<session>/collection/<host label>/` and merges the manifests. A progress bar appears under the command log while the transfer runs, showing the transferred and total bytes, the current rate and an estimate of the time left; **Cancel** next to it stops the transfer.

Files are staged under `artifacts/<session>/.staging/` and are merged into the artifact tree only once every file has arrived at its full remote size, so an interrupted download can never leave a truncated recording in `artifacts/`. If a download is interrupted — the network drops, you press Cancel, or you quit the console — the staged data is kept: press **Download** again and the transfer picks up where it stopped. Resume is byte-exact when `rsync` is installed on both machines; without it the console falls back to scp, which resumes file by file — the recordings that already arrived in full are skipped, and only a partly-written one is fetched again from the start. The log names the transport that was used. A second **Download** for a session and host that is already downloading is ignored.

A session that is still recording downloads as far as it has been written: the files that are still growing are named in the log and kept in staging rather than merged, because FFmpeg only finalizes a container when the recorder stops. Stop the recorders and download again.

### Pipelines

The ASR, IPS and VFA base cards share one shape:

- **Launch** tab: how many bases and synchronizers (and IPS visualizers) to start, the **Session** (an existing id or `Create MongoDB Session` from the **Experiment Group**), the mode for ASR and VFA (`live`, `capture`, `analyze`) and the pipeline's toggles. Start opens one terminal window per instance; each base asks which entry of the `Bases` list it is, then waits for the START signal.
- **Config** tab: the pipeline's `config.yml`. The sections that System Settings manage carry the same names as there (`Gateway (Nginx)`, `MQTT (Mosquitto)`), and the ones that only make sense together say how: `Gateway` is the Nginx load balancer, and each entry of `Server` either goes through it (a bare name such as `infer`, shown resolved: `through the Gateway: http://<host>:8080/infer`) or connects to a server directly (a full URL). `+ Add Stream` adds a `Streams` entry and, for ASR, `+ Add Base` adds a device type under `Base`. `Bases` entries are edited with dropdowns filled from the config (calibrated cameras, base types, sources, files in `file_dir`).
- **Streams** tab: the `Streams` entries of the config (they are defined on the Config tab, where `target` takes the path alone, `ips/cam-1`, and Save completes it with the Stream Server of System Settings; they belong to the pipeline because its bases read them from its config), with **Start**, **Stop**, **Logs**, **Probe**, **Record on/off**, **Start All** and **Stop All**, and above them a **Recordings** row with **Download**. Recording has two independent switches: **Record on/off** sets the selected stream's `record` (on the capture device, next to the push), and the Stream Server records on its side whatever reaches it (its card, Config tab). A stream with an `ssh_profile` is started as an FFmpeg process inside a tmux session named `mmla-stream-<name>` on that host (H.264 to MediaMTX for `rtmp://`, `rtsp://` and `srt://` targets, raw PCM for the `udp://` and `tcp://` targets of ASR bases); a stream without one is shown as `External` and only pulled from. **Probe** decodes two seconds of the URL the bases pull (`read_target`, else `target`). A stream with `record: true` also writes a raw recording on the capture host under `<record_root>/streams-<date>/collection/<host label>/`: filed by day and not under a session, because a stream is shared by the sessions that pull it (one after another, or several groups at once); Stop waits for FFmpeg to finalize the file and names it. **Download** in the Recordings row works in two ways. With a session chosen, it takes that session's start and end from MongoDB and cuts that part out of the recordings of every stream of the card, on the capture host and without re-encoding (a video cut is moved back onto a keyframe, at most a second, and named after that frame), then copies only the cuts into `artifacts/<session>/collection/<host label>/`; sessions that share a camera each get their own part. With `Everything ...` it copies the whole files of the selected stream's capture host, every day it holds, into `artifacts/streams-<date>/collection/<host label>/`. An external stream shows `n/a` under Record: the console does not run its FFmpeg, so nothing records it on a capture host (the Stream Server still does: **Sessions → Export Recordings**). The Collection card's own Download is for the sessions of its recorders. See the [Streaming guide](rtmp_streaming.md#recording).
- **Transform Matrix** tab (IPS only): the `transformation_matrices*.json` files produced by camera sync, editable as JSON, with **Sync to Remote** to copy them to a base station.

The server cards are different:

- **ASR Server** and **VFA Server** run the AI services with docker compose on the selected host (`docker/docker-compose.asr.yml`, `docker/docker-compose.vfa.yml`). The Launch tab lists the sub-services from the server config with a `true`/`false` toggle each; Start runs `docker compose up -d --build` for the selected ones, Stop runs `down`, Logs tails the containers. The VFA Server card also has a **Prompts** tab (the templates under `pipelines/vfa-server/prompts/`) and an **Action Schema** tab (`config/vfa/action_schemas.yml`).
- **MLLM Server** starts `vllm serve` from `config/mllm_server.yml` in the `vfa-vllm` environment, inside a tmux session named `mllm-server`, for VFA setups that use a local vision-language model.
- **IPS Camera Calibration** and **IPS Camera Sync** run the interactive calibration and multi-camera synchronisation tools; the calibration card also lists the captured calibration images per camera.

The pipeline guides walk through each one end to end: [ASR](pipelines/asr.md), [IPS](pipelines/ips.md), [VFA](pipelines/vfa/index.md).

### Session Control

Bases and synchronizers block after start-up until they receive a START signal for their session. This leaf lists the sessions, lets you tick the pipelines, and sends **START** or **STOP** over Redis. STOP also marks the session as ended in MongoDB. The Redis and MongoDB addresses come from System Settings and are shown on the panel.

There is no host to pick here: the signal is published from this machine to that Redis, and every base subscribed to the same Redis hears it, whichever machine it runs on. What matters is therefore that all machines of a session use **one** Redis. With `Redis.host` left at `localhost`, only bases on this machine hear the signal, because a base on another machine reads `localhost` as itself; the panel warns about it. Put the host name of the machine that runs Redis into **System Settings → Redis** for a session that spans machines.

A typical run is therefore: start the AI servers and the system services, start the bases and synchronizers on every base station (each picks its `Bases` entry), then send START from here once every window reports it is waiting, and STOP when the session is over.

## Sessions tab

A table of sessions with `Session ID`, `Experiment`, `Group`, `Status`, `Started` and `Source`, merged from the MongoDB `sessions` collection and, on the local host, from the `artifacts/<id>/` and `collection/<id>/` directories and their manifests. The addresses come from `config/system_services.yml`; with a remote host selected, its pipeline config is read over SSH instead.

The table is read again every time the tab comes into view, so a session created or recorded in the Launcher is there without a press on **Refresh** (which also reconnects to the databases). The Host selector opens on the Launcher's host; while that one is Local it opens on the machine the MongoDB address in System Settings names, as long as a saved SSH profile matches that address and answers — a database that lives on another machine is therefore listed without switching by hand. A host picked here stays until the Launcher's host changes again.

- **Export Measurements** writes one JSON file per event type (speaker recognition and transcription, IPS translation, rotation and relation, VFA actions) from InfluxDB into `artifacts/<session>/measurements/`, plus a text transcript.
- **Export Visualizations** and **Export All** additionally render the ASR diarization and speaking-interaction plots and the IPS trajectories, heatmap and physical-interaction network into `artifacts/<session>/analysis/visualizations/`. VFA has no visualizations yet.
- **Export Recordings** fetches the session's footage from the Stream Server: every path MediaMTX recorded between the session's start and end (MongoDB; up to now while it still runs), each cut to that window, into `artifacts/<session>/recordings/<app>/<name>_<start>.mp4`. A stream is shared by the sessions that pull it and the server records it whether or not one runs, so nothing on the server belongs to a session: the time range selects the footage, and two sessions that overlap each get their own cut of the same camera. The field next to the button narrows it to some paths (`ips/*, vfa/front`), for instance to leave out the microphones of another group; empty takes every path. The file name carries the start of the cut, which is what the `file` source reads, so `Base.file_dir` on one of these folders replays it. The server's address and its API and playback ports are under **System Settings → Stream Server**; this talks HTTP to the server and needs no SSH. The **Recordings until** column says when the server begins to delete a session's footage (its start plus the retention set on the Stream Server card's Config tab; `kept` when nothing is ever deleted, `gone` once it has passed), and the summary line names the retention in force.
- **Delete Session** removes the InfluxDB measurements and the MongoDB document, keeping local artifacts. **Delete Artifacts** removes the local directory, keeping the database records. Both need a second press.

The measurement and visualization exports need the `uber-base` extra (analytics) in the console's environment; Export Recordings needs nothing beyond the console.

## Status tab

What is running, and where: `Service`, `Host`, `Status`, `Port` and `tmux`, refreshed every five seconds while the tab is open. The `tmux` column names the tmux session a service runs in and since when (`flask · since 09-17 13:40`): the dashboard and its worker, the MLLM server, a native MediaMTX, streams and recorders. Databases and brokers are brew, systemd or docker processes and show `-`. It has nothing to do with the recording sessions of the Sessions tab. The rows come from the Launcher's own picture of the deployment, so the two tabs cannot disagree: every service is listed on **the host its card is on** (the machine System Settings name for a system service, else the host the card was last pointed at) and probed the way its sidebar marker is. The ASR and VFA servers show how many of their containers answer (`Running 4/6`). Bases and camera tools run in their own terminals and are not listed; any other tmux session on this machine is, by name.

- A service that is **not running** gets a row only when System Settings put it on a named machine: it is part of the deployment, so its being down is worth seeing. Everything else that is stopped is counted in the summary line, and **Show all** lists it, with the host of its card.
- The five-second refresh only asks what can be asked from here (this machine, and addresses that name a machine). A service whose host has to be logged into shows `? (Refresh)` until **Refresh** runs the full pass over SSH; Refresh also looks on every reachable host for system services whose address is `localhost`. Windows hosts are skipped.
- **View Logs** shows the selected row's logs: container or brew/journald logs for system services, the tmux pane for sessions, over SSH when the row's host matches an SSH profile. For the AI servers, use **Logs** on their Launcher card.

## Files the console writes

| Path | Content |
|---|---|
| `config/system_services.yml` | System Settings connections and the sudo password (secrets encrypted); gitignored, template `config/system_services_template.yml` |
| `config/ssh_profiles.yml` | SSH profiles (gitignored) |
| `config/launcher_hosts.yml` | the host each Launcher card was last pointed at (gitignored) |
| `config/experiments.yaml` | experiments and participants (gitignored) |
| `config/tasks/*.yaml`, `config/vfa/action_schemas.yml`, `config/mllm_server.yml` | task definitions, VFA action schema, MLLM Server settings |
| `pipelines/*/config.yml` | pipeline configs, with the shared sections mirrored from System Settings (gitignored) |
| `pipelines/ips-base/camera_calib/`, `camera_sync/` | calibration images and transform matrices |
| `artifacts/<session>/` | recordings, exported measurements and visualizations, manifests |
| `artifacts/<session>/.staging/` | partly-downloaded remote data and its resume ledger; removed when a download completes, and swept after 14 days |
| `~/.openmmla/master.key` | encryption key; `~/.openmmla/streams/` holds stream start times, `~/.openmmla/collection-runtime/` the pushed recorder code on remote hosts |
