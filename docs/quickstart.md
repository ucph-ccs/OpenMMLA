# Quickstart

This page takes a new deployment from nothing to a collected, exported and archived session, through the management console, `mmla tui`. Follow it once from top to bottom; the [Management console (TUI)](tui/index.md) page describes every panel, and the pipeline guides give each pipeline's details.

A session is collected in one of [two ways](#two-ways-to-collect-a-session), and the [one-time setup](#one-time-setup) is the same for both:

- **Collection mode** records raw audio and video only. The pipelines analyze the recordings later.
- **Pipeline mode** runs the ASR, IPS and VFA pipelines during the session and writes their measurements as it goes.

## The machines

A deployment is split into roles. One machine can hold several roles, and for a first try one machine can hold all of them. Only the AI servers need a GPU.

| Role | What runs there | What it needs |
|---|---|---|
| Console | `mmla tui`; the terminal windows of the bases and recorders it starts open on its screen, whichever machine they run on | a desktop (macOS, Ubuntu with GNOME Terminal, or Raspberry Pi OS), conda, git, a clone of the repository, the `tui` env |
| Uber server | the system services: InfluxDB, MongoDB, Redis, MQTT (Mosquitto), and optionally the Gateway (Nginx), the Stream Server (MediaMTX) and the dashboard; the archive of the sessions | a clone, Docker Engine, `make`, `sudo`, tmux, the `uber-server` env |
| GPU server | the AI services: ASR Server and VFA Server in Docker, optionally the MLLM Server | an NVIDIA GPU and driver, Docker Engine with the NVIDIA container toolkit, a clone; the `vfa-vllm` env for the MLLM Server |
| Base station | the ASR, IPS and VFA bases and synchronizers | a clone, conda, the env of each pipeline it runs (`asr-base`, `ips-base`, `vfa-base`) |
| Capture device, such as a Raspberry Pi | FFmpeg: a camera or microphone pushed to the Stream Server, or a Collection recorder | SSH and FFmpeg; tmux for a stream, `python3` for a recorder; no clone and no conda |

Every machine other than the console is reached over SSH and needs a POSIX shell there: Linux, macOS or Raspberry Pi OS ([Hosts](tui/index.md#hosts)). The examples on this page call the uber server `uber-server` and the GPU server `gpu-server`.

## One-time setup

Do these steps once per deployment, in this order. A machine added later goes through [Add the other machines](#add-the-other-machines) and [Prepare every machine](#prepare-every-machine) on its own.

### Install the console

1. Install the [prerequisites](prerequisites.md) on the console machine: conda, git, tmux and FFmpeg.
2. Clone the repository, make the `tui` env and start the console:

    ```bash
    git clone https://github.com/ucph-ccs/OpenMMLA.git
    cd OpenMMLA
    conda create -n tui python=3.10 -y
    conda activate tui
    pip install -e '.[tui]'
    mmla tui
    ```

The console opens on four tabs; `q` quits.

| Tab | What it holds |
|---|---|
| **Environment** | the conda envs and the repository on each machine |
| **Launcher** | the settings, the system services, the Collection card and the pipelines |
| **Sessions** | export, archive, end and delete sessions |
| **Status** | what runs where |

??? info "Details: why the console runs from a clone"
    The console edits the configs of its own clone (`pipelines/`, `config/`) and runs the compose files under `docker/`. With the editable install above, `mmla tui` works from any directory ([Install and start](tui/index.md#install-and-start)).

### Add the other machines

Every remote action picks a machine by its SSH profile, so add the profiles first.

1. Open `Launcher → System Settings → Hosts → SSH Profiles` with the **Host** selector at the top on `Local`.
2. Fill in one profile per machine (fields below) and press **Test Connection**.
3. Press **Add Profile**. It saves the profile to `config/ssh_profiles.yml`, which is gitignored, with the password encrypted.
4. Check that the **Host** selectors of the Environment and Launcher tabs list the profile. The `↻` button beside them tests every profile.

| Field | Example | What it is |
|---|---|---|
| **Profile Name** | `gpu-server` | the name the console shows for the machine everywhere |
| **Host** | `gpu-server.lan` | the name or address this machine reaches it by |
| **User**, **Port** | `22` | the SSH login |
| **Password** | | the account's password; leave it empty for a key login, except on a machine where the console runs `sudo` |
| **Key Path** | `~/.ssh/id_ed25519` | the private key of a key login |
| **Remote Project Path** | `~/OpenMMLA` | where the clone is, or will be, on that machine |

![SSH Profiles form under System Settings → Hosts: the saved profiles with Test, Edit and Delete, and below them a profile being edited, with the Save Profile, Test Connection and Clear Form buttons over its Profile Name, Host and User fields](img/tui/ssh-profiles.png)

??? info "Details: passwords, sshpass and offline machines"
    - The console runs `sudo` for **Install Tools**, for the native system services, and on a Pi that gets FFmpeg installed by a stream's Start. `sudo` there is answered with the profile's password, so such a machine needs it even with a key login.
    - **Test Connection** checks a given password on its own, since a key logs in whatever the password says while `sudo` asks for the real one.
    - A password login needs `sshpass` on the console machine, and **Test Connection** fails for it until then. Once a profile with a password is saved, **Refresh** on the Environment tab names `sshpass` in the `System` column of the `tui` row, and **Install Tools** there installs it.
    - **Edit** on a saved profile loads it into the form, where the button reads **Save Profile**.
    - The form shows the profiles of the machine the **Host** selector names; this console uses its own.
    - A machine that does not answer reads `(offline ✗)` in the selectors and cannot be picked until it does.

### Prepare every machine

Each machine gets the repository and the conda env of its role from the **Environment** tab.

| Role | Env on the Environment tab | Installed by hand |
|---|---|---|
| Console | `tui`, made in [Install the console](#install-the-console) | |
| Uber server | `uber-server`, for the dashboard, its worker and Nginx | Docker Engine, with the SSH user in the `docker` group |
| GPU server | `vfa-vllm`, only for the MLLM Server; the ASR and VFA Servers run in Docker | NVIDIA driver, Docker Engine and the NVIDIA container toolkit, with the SSH user in the `docker` group ([Host requirements](docker.md#host-requirements)) |
| Base station | `asr-base`, `ips-base`, `vfa-base`: one per pipeline it runs | |
| Capture device | none | FFmpeg; see [Capture devices](#capture-devices) below |

For each machine with an env in that table:

1. Install conda on it ([Conda](prerequisites.md#conda)); the console does not.
2. On the Environment tab, set **Host** to the machine. The table lists its envs with their `Status`, and under `System` what the machine lacks for each.
3. On a remote machine without a clone, press **Git Clone**. It clones this clone's origin into the profile's Remote Project Path.
4. Select the env's row and press **Create Env**.
5. When `System` names something, press **Install Tools**. For `asr-base`, do it before the next step, since PyAudio builds against PortAudio.
6. Press **Install Deps**, which installs the env's extra of `pyproject.toml`. The row then reads `Ready`.

![Environment tab: the Host selector, the table of conda envs with Status and System columns, and the Connect, Refresh, Git Clone, Git Pull, Git Pull All, Create Env, Install Deps, Install Tools and Delete Env buttons](img/tui/environment.png)

??? info "Details: Install Tools, Git Clone and the command console"
    - `Status` reads `Missing`, `Partial: ...` or `Ready`; `System` reads like `lacks ffmpeg, portaudio`.
    - **Install Tools** installs with `apt-get` through `sudo` on Debian, Ubuntu and Raspberry Pi OS, and with `brew` on a Mac. `sudo` is answered with the password of the machine's SSH profile, or on this machine with the one saved under `Launcher → System Settings → Credentials → Sudo (local admin)`.
    - **Git Clone** needs git on the machine and access to the origin URL.
    - The console runs plain `docker compose` on the uber server and the GPU server, so their SSH user must be in the `docker` group ([Docker guide](docker.md)).
    - The command console at the bottom of the tab runs a typed command on the machine **Host** names, in its clone.

#### Capture devices

A capture device that only streams or records needs no env: an SSH profile, a user in the `video` and `audio` groups, and FFmpeg ([Raspberry Pi](raspi_config.md)).

??? info "Details: FFmpeg on a Pi, and Macs that record"
    - **Start** on a Streams tab installs FFmpeg and tmux on the device when they are missing.
    - The Collection card copies its recorder code to the device but installs nothing. On a Pi that only records for it, run `sudo apt-get install -y ffmpeg` in a shell on the Pi.
    - A Mac that records its own camera or microphone needs someone logged in on its screen, with Terminal allowed under **Privacy & Security → Camera** and **Microphone** ([Recording on a Mac](tui/collection.md#recording-on-a-mac)).

### Point everything at the system services

The pipelines, the dashboard and the console find the system services through the forms under **System Settings → Connections**. Fill them in before starting any service: each System Services card opens on the machine its address names.

1. Open `Launcher → System Settings → Connections` with the **Host** selector on `Local`. On a new machine every address reads `<uber-server>`, a placeholder that counts as not set ([A new machine](system_services.md#a-new-machine)).
2. In each form, replace `<uber-server>` with the name every machine reaches the uber server by, as the **Host** field of its SSH profile has it. Set the values below.
3. Press **Save** on each form. It writes the section into `config/system_services.yml` and into every pipeline config of this machine that carries it.

| Form | Field | Value |
|---|---|---|
| **InfluxDB** | `url`, `org`, `bucket` | `http://uber-server:8086`, `admin` (the form starts empty), `mmla-data`; the `token` comes from **Fetch Token** in the [next section](#start-the-system-services) |
| **MongoDB** | `url` | `mongodb://uber-server:27017` |
| **Redis** | `host`, `db` | `uber-server`, `1` (the form starts at `0`) |
| **MQTT (Mosquitto)** | `host` | `uber-server` |
| **Gateway (Nginx)** | `host` | `uber-server`, also when Nginx will not run |
| **Stream Server (MediaMTX)** | `host` | `uber-server`, for streamed cameras and microphones |
| **Dashboard (Flask)** | `host` | the machine the dashboard runs on |

!!! warning "Name a machine, never `localhost`"
    Every machine reads `localhost` as itself, so a base station would look for Redis on itself and never hear the START sent from Session Control. Use `localhost` only when one machine runs everything.

![System Settings → Connections → InfluxDB form: the url, token, org and bucket fields with Save and Reset to Defaults, and the host picker with Sync to Host and Sync from Host](img/tui/connections.png)

??? info "Details: why these values"
    - The console matches each address against the SSH profiles, and opens the System Services cards on the profile it matches.
    - `org` `admin` and `bucket` `mmla-data` are what the Docker stack creates. Redis `db` `1` keeps the dashboard's job queue out of the default database.
    - Every pipeline config carries a `Gateway` section, and **Start** refuses a pipeline whose sections name no machine, so the Gateway form needs a host even without Nginx.
    - Once saved, the pipeline Config tabs show these fields read-only. Every field is in [Pointing the pipelines at the services](system_services.md#pointing-the-pipelines-at-the-services).

??? info "Details: which other machines need the settings"
    - **Base stations**: nothing to do. The first **Start** of a base card on a remote machine copies the settings into its pipeline config and says `Relaunch <card> once the sync above completes.`; press **Start** again.
    - **The dashboard's machine**, when it is not this one: the dashboard reads the `config/system_services.yml` of the machine it runs on. On the InfluxDB, MongoDB, Redis and Stream Server (MediaMTX) forms, pick that machine in the host picker beside **Sync to Host** and press **Sync to Host**. Do the InfluxDB form after **Fetch Token** has stored the token.
    - **A second console** takes the settings with **Sync from Host** on each form ([A new machine](system_services.md#a-new-machine)).

### Start the system services

Which services a session needs:

| Card under `Launcher → System Services` | Collection mode | Pipeline mode |
|---|---|---|
| **InfluxDB** | for the replay | required: every measurement |
| **MongoDB** | required: the session records | required |
| **Redis** | for the replay | required: START and STOP |
| **MQTT (Mosquitto)** | for the replay | required: bases to synchronizers |
| **Stream Server (MediaMTX)** | not used | streamed cameras and microphones, the server-side recording of a session, the dashboard's live video |
| **Gateway (Nginx)** | optional, for the replay | optional: a load balancer in front of the AI services |
| **Dashboard (Flask)**, **Dashboard (Celery)** | optional | optional: live view, replay and analysis report; the worker takes the report jobs off the web process |

1. **Sudo password.** When a service runs natively on this machine, as Redis, Mosquitto and Nginx always do (through `make` and `sudo`), save its sudo password under `Launcher → System Settings → Credentials → Sudo (local admin)`.
2. **Docker secrets.** InfluxDB in Docker needs `docker/.env` on its machine before its first start. Type this into the Environment tab's command console with **Host** on the uber server; it fills both secrets with random values:

    ```bash
    test ! -e docker/.env && cp docker/.env.example docker/.env && chmod 600 docker/.env && sed -i.bak -e "s|^INFLUXDB_INIT_ADMIN_TOKEN=.*|INFLUXDB_INIT_ADMIN_TOKEN=$(openssl rand -hex 32)|" -e "s|^INFLUXDB_INIT_PASSWORD=.*|INFLUXDB_INIT_PASSWORD=$(openssl rand -base64 24)|" docker/.env && rm docker/.env.bak
    ```

3. **Start the services.** Open each card under `Launcher → System Services` and press **Start** on InfluxDB, MongoDB, Redis and MQTT (Mosquitto), then on Stream Server (MediaMTX) when cameras or microphones are streamed. The sidebar heading says which machine the cards run on (`System Services @ uber-server`).

    ![Launcher tab: the sidebar tree with System Settings, System Services, Collection and Pipelines, and the InfluxDB card on the right with its status, Run mode, Start, Stop, Logs, Refresh and Fetch Token](img/tui/launcher.png)

4. **InfluxDB token.** Press **Fetch Token** on the InfluxDB card. It stores the Docker stack's admin token, encrypted, as `InfluxDB.token` in this machine's settings.
5. **Gateway and dashboard.** With each card's **Host** on its machine, press **Save** on the **Config** tab of Gateway (Nginx) and of Dashboard (Flask). Then **Start** Gateway (Nginx), Dashboard (Flask) and Dashboard (Celery).
6. **Check.** Each card's status line, the `(R)` marker after its name in the sidebar, and the **Status** tab show what answers.
7. **Open the dashboard** at `http://uber-server:5050`. The header of its Sessions page says whether InfluxDB, MongoDB, the report worker and the Stream Server answer ([Sessions page](dashboard/index.md#sessions)).

    ![Dashboard Sessions page: the header with the state of InfluxDB, MongoDB, the report worker and the Stream Server, the search, Speech and Sort filters, and the sessions grouped by month with their data, speech setup and Replay and Analysis buttons](img/dashboard/sessions.png)

!!! warning "The dashboard has no login"
    It serves the archived recordings of every session to anyone who reaches its port. Keep it on a trusted network, or turn the recordings off ([Keep the dashboard private](dashboard/deploy.md#keep-the-dashboard-private)).

??? info "Details: sudo and `docker/.env`"
    - A password saved for **Install Tools** already serves here. On a remote machine the console answers `sudo` with the password of its SSH profile.
    - `INFLUXDB_INIT_PASSWORD` needs 8 characters or more; without it the container restarts in a loop.
    - The command above does nothing where `docker/.env` exists already. The values are read once, when the stack starts against an empty volume ([Setup](docker.md#setup)).
    - For the dashboard's live video and sound from a Stream Server in Docker, also set `MEDIAMTX_WEBRTC_HOSTS` in that file to the name or address browsers reach the uber server at ([Live video and sound in MediaMTX](dashboard/deploy.md#live-video-and-sound-in-mediamtx)).

??? info "Details: what Start runs"
    - **InfluxDB**, **MongoDB** and **Stream Server (MediaMTX)** run `docker compose -f docker/docker-compose.infra.yml up -d <service>` on the card's machine, unless their **Run mode** is `native`, for a machine that already runs them as brew or systemd services ([From the console](system_services.md#from-the-tui)).
    - **Redis**, **MQTT (Mosquitto)** and **Gateway (Nginx)** run `make -C pipelines/uber-server <service>`, which installs the package when the machine has none and makes it listen on every interface ([Listener configuration](system_services.md#listener-configuration)).
    - The **Status** tab lists every service with its host, status and port.

??? info "Details: the token, the Gateway and the dashboard"
    - **Fetch Token** reads the admin token of the Docker stack on the card's machine; the log shows only its first and last four characters. When the dashboard runs on another machine, press **Sync to Host** on the InfluxDB form for it now.
    - A `native` InfluxDB has no such token: type the token of its initial setup into the InfluxDB form ([InfluxDB](system_services.md#influxdb)).
    - Gateway (Nginx) and Dashboard (Flask) read a `config.yml` on their machine, which a fresh clone does not have; **Save** on the card's **Config** tab writes it. Both also need the `uber-server` env there, in which Nginx's config is rendered and the dashboard runs.
    - Start the Gateway only when the bases are to reach the AI services through it. Dashboard (Flask) runs in a tmux session named `flask`, and Dashboard (Celery) in one named `celery`.

### Describe the study

The cards create every session in an experiment group, so describe the study before the first session.

1. **Task.** `Launcher → System Settings → Study → Tasks` lists the task definitions in `config/tasks/`. To add one, type a name under **New task name**, press **Create Task**, edit its YAML and press **Save**.
2. **Experiment.** In `Launcher → System Settings → Study → Experiments`, type an id such as `exp_20261001_lego_building` under **New experiment ID** and press **Create Experiment**.
3. **Title and task type.** Set **Title** and **Task Type**, and leave **Status** on `active`.
4. **Participants.** Under **Add Participant**, add one entry per participant (fields below) with **Add**, then press **Save & Back**.

| Field | Example | What it is for |
|---|---|---|
| **Name** | | the participant's name; the dashboard never shows it |
| **Group ID** | `group_01` | the group they are in |
| **Tag ID** | `3` | the AprilTag they wear, by which IPS and VFA tell them apart; printable tags are in `pipelines/ips-base/apriltag/` |
| **Description** | `grey hoodie, left of the table` | a short appearance note, which goes into the VFA prompts |

The **Experiment Group** dropdown of the Collection and base cards now lists `<experiment>/<group>` for every group of every active experiment.

![System Settings → Study → Experiments: one experiment with its ID, Title, Task Type and Status, its participants with group and tag, and the Add Participant fields](img/tui/experiments.png)

!!! warning "Participant names are personal data"
    `config/experiments.yaml` is gitignored, but the session documents in MongoDB carry the names too, and **Sync to Host** on the Experiments form copies the whole file. Sync it only to machines that should hold it. The dashboard shows participants by their tag (`Tag 0`).

??? info "Details: experiment ids, tasks and groups"
    - The experiment id starts the id of every session of the experiment, `<experiment>_<group>_<YYMMDDTHHMMZ>`, and with it the names of the session's folders. Once a session carries the id, it cannot change.
    - An id has up to 64 ASCII letters, digits, `_`, `-` and `.`, and starts with a letter or digit. The dashboard reads a session's date and task from the form `exp_<YYYYMMDD>_<task type>`.
    - An experiment's **Task Type** picks from the tasks, so create the task first.
    - Only `active` experiments are offered on the cards; a new experiment starts `active`.
    - A session created from a card belongs to the group picked there, and its MongoDB document carries that group's participants.

## Two ways to collect a session

| | Collection mode | Pipeline mode |
|---|---|---|
| What runs during the session | one FFmpeg recorder per camera and microphone, on the machine it is plugged into, started from the Collection Session card | the bases and synchronizers on the base stations, the AI servers on the GPU server, usually streams through the Stream Server |
| AI servers | not during the session; needed for the replay | ASR Server for ASR, VFA Server for VFA (IPS needs none) |
| Streams | none: each recorder opens its device | cameras and microphones pushed to the Stream Server and pulled by the bases, or devices of the base station itself |
| Session Control | not used: **Start Audio**, **Start Video** and **Stop** on the card | **Send START** and **Send STOP** |
| What you get | raw recordings per machine, with manifests; measurements once they are replayed | measurements in InfluxDB as the session runs, live on the dashboard |
| When to choose it | no GPU server at hand, a first try on one machine, analysis settings chosen afterwards | results during the session, a live view |

!!! note "A pipeline session keeps its raw recordings too"
    When its devices are streamed, the Stream Server records the streams of a running session from START to STOP, and a stream with **Record** `yes` on its Streams tab also records on its capture device ([Recording](streaming/recording.md)). **Sessions → Export** gathers both. A device is recorded by the Collection card or streamed, not both at once.

## Collection mode

The Collection card registers each session in MongoDB, in an experiment group from [Describe the study](#describe-the-study). [Collection](tui/collection.md) lists every field.

### Plan the recorders

1. Open `Launcher → Collection → Collection Session`.
2. Set **Session ID** to `Create MongoDB Session` and **Experiment Group** to the group being recorded.
3. Leave **Output Root** on `artifacts`: each machine writes its recordings into `artifacts/<session>/collection/<host>/` on its own disk.
4. On the **Audio** tab, set **Num Audio** to the number of microphones and fill in one row per recorder (columns below). Click a cell, or press Enter on it, to open its dropdown.
5. On the **Video** tab, set **Num Video** to the number of cameras and fill in each row the same way.

| Column | Tab | What it holds |
|---|---|---|
| **Host** | both | the machine the device is plugged into: `Local` or an SSH profile |
| **Device** | both | a device that machine reports, with its channel count where the machine says it |
| **Channel** | Audio | `mix`, every channel averaged into one mono file; `each`, a file per channel; or one channel (`ch0`) |
| **Device Label** | both | the name the recording's files give the device (`jabra-1`), offered from the `Streams` entries of the pipeline configs; **type another…** takes any other |
| **Participant** | Audio | whose voice it records: a participant of the group (`<name> (tag <id>)`), **Group (room microphone)**, or **bind later** |
| **Rotate** | Video | `180°` for a camera mounted upside down; FFmpeg turns the picture as it records |

![Collection Session card, Audio tab: Num Audio, the Start Audio, Stop, Stop All Hosts, Logs, Refresh, Download and Delete Remote buttons, the Session ID, Experiment Group and Output Root fields, and the recorder table with Host, Device, Channel, Device Label and Participant](img/tui/collection.png)

??? info "Details: the recorder table"
    - Each recorder runs on the machine its row names, so the card's **Host** selector plays no part and reads `Per recorder`.
    - On a remote machine, `artifacts` is `~/artifacts`.
    - `+` and `-` change **Num Audio** and **Num Video**; 0 records none.
    - A machine's devices are asked over SSH the first time its **Device** dropdown opens.
    - `each` suits a receiver of several worn microphones. A recorder that writes several channels gets a row per channel, each with its own **Participant**.
    - A **Device Label** whose `Streams` entry names the machine that captures it fills **Host** and **Device** too.
    - **Start** refuses two recorders on one device of one machine, and one tag picked on two rows.

### Record

1. Press **Start Audio** on the **Audio** tab. It creates the session in the picked group and opens one terminal window per recorder on this screen.
2. Press **Start Video** on the **Video** tab. The card now shows the session Start Audio created, so the cameras record into it.
3. Check that the card reads `Running` and that each window shows its recorder's output. **Logs** lists the live recorders of each machine.
4. At the end, press **Stop**. It stops every recorder of the session on every machine of the card, on both tabs, and marks the session ended in MongoDB.

??? info "Details: Start and Stop"
    - Start copies the recorder code to each remote machine of the card; a recording machine needs no clone.
    - A recorder with a Device starts at once and asks nothing.
    - A machine that cannot be reached is named in the log and left out, and the others record.
    - **Stop All Hosts** also reaches every other SSH profile, for a recorder started from another console.
    - After Stop the card goes back to `Create MongoDB Session`, so the next Start is a new take.

### Collect the files

The recordings of a remote machine stay on it until they are fetched. For a finished session, **Sessions → Export** fetches them with the rest of the session: see [Export and archive](#export-and-archive).

??? info "Details: Download and Delete Remote on the card"
    - **Download** copies the recordings of every remote machine of the card into this console's `artifacts/<session>/collection/<host>/`. It resumes where it stopped, shows a progress bar with **Cancel** under the log, and a second press fetches only what is new ([Download a session](tui/collection.md#downloading-a-session)).
    - A session that is still recording downloads as far as it has been written, and the files still growing stay staged: stop the recorders, then download again.
    - Once the recordings are here and archived, **Delete Remote** removes them from the card's remote machines. The first press names each machine and the session; the second deletes.

### Analyze the recordings later

A collected session is analyzed by replaying its recordings through the pipelines, whose bases then read files instead of devices or streams. The replay needs the system services of Pipeline mode, and the AI servers of ASR and VFA.

1. Do the [one-time part of each pipeline](#set-up-the-pipelines) to be replayed.
2. [Export](#export-and-archive) the session, so its recordings are on this console, where the bases of the replay run. This machine needs the env of each pipeline it replays.
3. On the **Config** tab of each base card, with **Host** `Local`, give the `Bases` entries `source: file` and the full path of a recording as `source_index` (**Browse…** picks it).
4. Run it as in [Run a session](#run-a-session), without the streams and with the base cards on `Local`: pick the recorded session under **Session**, press **Send START**, and **Send STOP** once the bases have replayed their files.

Each pipeline guide says which files its bases replay: [ASR](pipelines/asr/run.md#post-time-processing), [IPS](pipelines/ips/run.md#post-time-processing), [VFA](pipelines/vfa/run.md#post-time-processing).

??? info "Details: replaying into an ended session"
    - The bases take the start time from the file names.
    - A Start into a session that has ended is held back once; press **Start** again to go on in it.

## Pipeline mode

### Set up the pipelines

Each pipeline has a part that is done once per deployment, and again only when something it sets changes. The AI servers are started there and keep running from session to session.

| Pipeline | What is set up once | Guide |
|---|---|---|
| ASR | the ASR Server's config and its Start on the GPU server; the ASR Base config (one `Bases` entry per microphone, the `Streams` they pull, the service endpoints); speaker profiles where bases verify speakers | [ASR → Once per deployment](pipelines/asr/run.md#once-per-deployment) |
| IPS | the cameras' intrinsics, their streams and `Bases` entries, the camera sync, the transform matrices on every base station | [IPS → Once per deployment](pipelines/ips/run.md#once-per-deployment) |
| VFA | the VFA Server's config and its Start on the GPU server (and the MLLM Server for a local model), the VFA Base config | [VFA → Once per deployment](pipelines/vfa/run.md#once-per-deployment) |

### Run a session

The three pipelines run a session the same way. Each guide lists its own card's fields: [ASR → Every session](pipelines/asr/run.md#every-session), [IPS → Every session](pipelines/ips/run.md#every-session), [VFA → Every session](pipelines/vfa/run.md#every-session).

1. **Check what runs.** The **Status** tab lists the system services and the AI servers (`Running 4/6`: four of a server's six containers answer).
2. **Start the streams.** On each base card's **Streams** tab, press **Start All**, and wait until each row's **Stream Server** column reads `● live`.
3. **Start the bases.** On each base card (ASR Base, IPS Base, VFA Base), with **Host** on the base station that runs those bases, set the fields below and press **Start**. It opens one terminal window per base and synchronizer, and each waits for START.

    | Base card field | Set it to |
    |---|---|
    | **Session** | on the first card, `Create MongoDB Session` with the **Experiment Group**; every other base card, on any machine, then opens on that session: keep it |
    | **Num Synchronizers** | `1` on one card per pipeline; `0` on the cards of that pipeline's other base stations |
    | **Sync Waits For** (ASR, VFA) | on the card that runs the synchronizer, the number of the session's bases of that pipeline on every machine |

4. **Send START.** Open `Launcher → Pipelines → Session Control`, keep the pipelines that run ticked (**ASR**, **IPS**, **VFA**), and press **Send START** once every window reports that it is waiting.

    ![Session Control: the session list with its refresh button, the ASR, IPS and VFA checkboxes, the Redis and MongoDB addresses, and the Send START and Send STOP buttons](img/tui/session-control.png)

5. **Watch it** on the dashboard at `http://uber-server:5050`. The session appears in the **Live now** band of the Sessions page, and its Live page follows it as it is written ([Live page](dashboard/index.md#live)).

    ![Dashboard Live page, here replaying an ended session: the replay controls, the health strip, the indicators, the room plan, the transcript and the activity timeline](img/dashboard/live.png)

6. **Send STOP** at the end. Every base and synchronizer of the session ends its run and exits, and the session is marked ended in MongoDB.
7. **Stop the streams** once no further session pulls them: **Stop All** on the card's **Streams** tab.

Then [export and archive](#export-and-archive) the session.

??? info "Details: streams"
    - Streams come before bases. A base opens its stream as it starts (an ASR base at START), waits up to 30 s for one that is not up, and then stops ([Bases: pulling a stream](streaming/index.md#bases-pulling-a-stream)).
    - **Start All** starts every stream of the card that the console runs; **Start** starts the selected row alone.
    - A stream needs no restart between sessions, so leave it running for the next take.
    - **Stop All** on the **Streams** tab of `Launcher → System Services → Stream Server (MediaMTX)` stops the streams of every card, and asks for a second press.

??? info "Details: what a base card's Start checks"
    - Start asks the Stream Server whether the streams the bases pull are live, and holds back once when one is not. Start it, or press **Start** again to go ahead.
    - On a remote machine whose settings are behind, the first Start only brings them up to date and says `Relaunch <card> once the sync above completes.`; press **Start** again.
    - Nothing is asked in the windows; each component starts and waits for START.

??? info "Details: START, STOP and the AI servers"
    - Session Control's list opens on the session the base cards were last started into.
    - START also switches on the Stream Server's recording of the session's streams, and STOP switches it off.
    - The bases and synchronizers exit on STOP also when STOP comes before START. STOP puts the base cards back on `Create MongoDB Session` for the next take.
    - The AI servers keep running until **Stop** on their cards, which runs `docker compose down`.

## Export and archive

Both modes end here. The [Sessions tab](tui/sessions.md) page describes every part.

1. Open the **Sessions** tab and select the session.
2. Press **Export**. It gathers the session onto this console, into `artifacts/<session>/` (parts below).
3. Press **Archive**. It sends the session's raw files from this console to the archive host, into `artifacts/<session>/` of its clone, and checks each file there by its sha256.

| Export part | What it holds |
|---|---|
| Measurements | the session's events from InfluxDB as JSON, a text transcript, and `<session>_parameters.json`, what every base and synchronizer ran with |
| Collection recordings | the recordings of every machine the session recorded on |
| Streams | the session's part of each stream its bases took, from the Stream Server over HTTP, and from the capture device when it recorded the stream too |
| Base files | what the bases and synchronizers wrote on their machines: logs, the config they ran with, what they stored |

![Sessions tab: the session table with Session ID, Experiment, Group, Status, Started, Recordings until, Files here and Source, and the Refresh, Export, Archive, End Session, Delete Session and Delete Files buttons](img/tui/sessions.png)

!!! warning "Export a pipeline session within three days"
    The Stream Server deletes each ten-minute segment of its recordings three days after it began. The **Recordings until** column says when that starts for each session.

The archived recordings are what the dashboard's Analysis page lists for download and playback.

![Dashboard Analysis page of an ended session: the Overview indicators, the session timeline and the participants table, under the Overview, Speech, Space, Attention and Data and exports tabs](img/dashboard/analysis.png)

??? info "Details: Export"
    - The table lists the sessions MongoDB knows and the session folders of the machine its **Host** selector picks (`Local` by default).
    - A progress row names what is being fetched from where, and **Cancel** stops it. **Export** again resumes and fetches only what is missing; the log ends with a summary per part.
    - `mmla ses-export <session>` does the same from a shell.
    - The Stream Server's retention is `recordDeleteAfter: 72h` in `pipelines/uber-server/mediamtx/mediamtx.yml`, **Keep recordings for** on the Stream Server card's **Config** tab.
    - A capture device keeps its stream recordings as long as each stream's `record_keep_days` says (`0`, the default, keeps them). A Collection recorder's files stay on their machine until **Delete Remote**.

??? info "Details: Archive"
    - The archive host is the machine of `System Settings → Connections → Dashboard (Flask)`, else of the Stream Server. It needs an SSH profile, unless it is this machine, and `python3`.
    - Each file is moved into place only once its sha256 there matches. The session's MongoDB document notes the archive as `complete`, or `partial` while something is missing; **Archive** again completes it.
    - Nothing is deleted on either side, and speaker profiles are never sent.
    - Export falls back on the archive for a file its machine no longer holds.
    - `mmla ses-archive <session>` does the same from a shell.

??? info "Details: ending and deleting sessions"
    - **End Session** is for a session left `active` because its console closed or its bases went down before STOP. The first press says the end it would write; the second writes it.
    - **Delete Files** frees space: it deletes the session's folders on the machine the **Host** selector picks, and nowhere else. The first press names each folder with its size; the second deletes.
    - The Collection card's **Delete Remote** removes its recorders' files from their machines, and **Manage** on a Streams tab deletes the stream recordings of the capture devices.
    - **Delete Session** is not a cleanup: it deletes the session everywhere central (its archive, its InfluxDB events and its MongoDB document) and cannot be undone ([Deleting a session](database.md#deleting-a-session)).

## Keeping up to date

1. Quit the console with `q`, run `git pull` in its clone, and start `mmla tui` again.
2. Press **Git Pull All** on the Environment tab, which pulls on every SSH profile.
3. When the pull changed `pyproject.toml`, press **Install Deps** for each env on each machine, and run `pip install -e '.[tui]'` in the console's `tui` env.
4. Press **Start** on the ASR Server and VFA Server cards. It runs `docker compose up -d --build`, which rebuilds the images and recreates the containers whose image changed.
5. **Stop** and **Start** Dashboard (Flask) and Dashboard (Celery).
6. When the pull changed `mediamtx.yml`, **Stop** and **Start** the Stream Server card between sessions.

??? info "Details: what a pull leaves behind"
    - A running console, and a running dashboard, keep the code they started with.
    - **Git Pull All** skips offline machines, and its last line says which machines pulled, were up to date or failed.
    - A machine where a **Sync to Host** changed a file that git tracks (a prompt, a task, `mediamtx.yml`) stops on that file when the pull changes it, until the change is committed or undone there ([Sync safety](tui/launcher.md#sync-safety)).
    - An env that lacks a package reads `Partial: ...`, and its `[E]` marker in the Launcher turns yellow.
    - A Stream Server in Docker reads `mediamtx.yml` only when its container starts, and the restart closes every stream for a moment.
