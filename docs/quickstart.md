# Quickstart

This page takes a new deployment from nothing to a collected, exported and archived session, through the management console, `mmla tui`. It gives the steps in the order they are done, with what to press and what to check; the [Management Console (TUI)](tui.md) page describes every panel in full, and the steps link there and to the pipeline guides for the details.

A session is collected in one of two ways, compared under [Two ways to collect a session](#two-ways-to-collect-a-session):

- **Collection mode** records raw audio and video only. The recordings are analyzed later, by replaying them through the pipelines.
- **Pipeline mode** runs the ASR, IPS and VFA pipelines during the session, and writes their measurements as it goes.

The [one-time setup](#one-time-setup) is the same for both.

## The machines

A deployment is split into roles. One machine can hold several of them, and for a first try one machine can hold all of them; a GPU is needed only for the AI servers, which Pipeline mode uses during a session and Collection mode when its recordings are replayed.

| Role | What runs there | What it needs |
|---|---|---|
| Console | `mmla tui`. The terminal windows of the bases and recorders it starts open on its screen, whichever machine they run on | a desktop (macOS, Ubuntu with GNOME Terminal, or Raspberry Pi OS), conda, git, a clone of the repository, the `tui` env |
| Uber server | the system services: InfluxDB, MongoDB, Redis, MQTT (Mosquitto), and optionally the Gateway (Nginx), the Stream Server (MediaMTX) and the dashboard; the archive of the sessions | a clone, Docker Engine (InfluxDB, MongoDB and MediaMTX run in Docker by default), `make`, `sudo` and tmux, the `uber-server` env for the dashboard and Nginx |
| GPU server | the AI services: ASR Server and VFA Server in Docker, optionally the MLLM Server | an NVIDIA GPU and driver, Docker Engine with the NVIDIA container toolkit, a clone; the `vfa-vllm` env for the MLLM Server |
| Base station | the ASR, IPS and VFA bases and synchronizers | a clone, conda, the env of each pipeline it runs (`asr-base`, `ips-base`, `vfa-base`) |
| Capture device, such as a Raspberry Pi | FFmpeg: a camera or microphone pushed to the Stream Server, or a Collection recorder | an SSH login and FFmpeg; tmux for a stream, `python3` for a Collection recorder. No clone and no conda |

Every machine other than the console is reached over SSH and has to offer a POSIX shell there: Linux, macOS or Raspberry Pi OS (see [Hosts](tui.md#hosts)). The examples below call the uber server `uber-server` and the GPU server `gpu-server`.

## One-time setup

These steps are done once per deployment, in this order. A machine added later goes through [Add the other machines](#add-the-other-machines) and [Prepare every machine](#prepare-every-machine) on its own.

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

3. The console opens with four tabs: **Environment** (the conda envs and the repository on each machine), **Launcher** (the settings, the system services, the Collection card and the pipelines), **Sessions** (export, archive, end and delete sessions) and **Status** (what runs where). `q` quits.

The console edits the configs of its own clone (`pipelines/`, `config/`) and runs the compose files under `docker/`, so it runs from a clone. With the editable install above, `mmla tui` works from any directory (see [Install and start](tui.md#install-and-start)).

### Add the other machines

Every remote action in the steps below picks a machine by its SSH profile, so the profiles come first.

1. Open `Launcher → System Settings → Hosts → SSH Profiles` with the **Host** selector at the top on `Local`. The form shows the profiles of the machine the selector names, and this console uses its own.
2. Fill in one profile per machine:
    - **Profile Name**: the name the console shows for the machine everywhere (`gpu-server`).
    - **Host**: the name or address this machine reaches it by.
    - **User** and **Port** (22).
    - **Password**: the account's password. Leave it empty for a key login, except on a machine where the console runs `sudo` (**Install Tools**, the native system services, a Pi that gets FFmpeg installed by a stream's Start): `sudo` there is answered with this password. A password login needs `sshpass` on this machine, and **Test Connection** fails for it until then: once a profile with a password is saved, **Refresh** on the Environment tab names `sshpass` in the `System` column of the `tui` row, and **Install Tools** there installs it (see [Prepare every machine](#prepare-every-machine)).
    - **Key Path**: the private key of a key login (`~/.ssh/id_ed25519`).
    - **Remote Project Path**: where the clone is, or will be, on that machine (`~/OpenMMLA`).
3. **Test Connection** logs in with what the form holds, and checks a given password on its own, since a key logs in whatever the password says while `sudo` asks for the real one. **Add Profile** saves the profile to `config/ssh_profiles.yml` (gitignored; the password is stored encrypted). **Edit** on a saved profile loads it into the form, where the button reads **Save Profile**.
4. The **Host** selectors of the Environment and Launcher tabs now list the profile. The `↻` button next to them tests every profile; a machine that does not answer reads `(offline ✗)` and cannot be picked until it does.

![SSH Profiles form under System Settings → Hosts: the saved profiles with Test, Edit and Delete, and below them a profile being edited, with the Save Profile, Test Connection and Clear Form buttons over its Profile Name, Host and User fields](img/tui/ssh-profiles.png)

### Prepare every machine

Each machine gets the repository and the conda env of its role from the **Environment** tab:

| Role | Env row on the Environment tab | Installed by hand |
|---|---|---|
| Console | `tui`, made in [Install the console](#install-the-console) | |
| Uber server | `uber-server`, for the dashboard, its worker and Nginx | Docker Engine, the user in the `docker` group |
| GPU server | none for the ASR and VFA Servers, which run in Docker; `vfa-vllm` for the MLLM Server | NVIDIA driver, Docker Engine and the NVIDIA container toolkit, the user in the `docker` group ([Host requirements](docker.md#host-requirements)) |
| Base station | `asr-base`, `ips-base`, `vfa-base`: one per pipeline it runs | |
| Capture device | none | see below |

For each machine that has a row in that table:

1. Install conda on it ([Conda](prerequisites.md#conda)); the console does not.
2. On the Environment tab set **Host** to the machine. The table lists the envs there with their `Status` (`Missing`, `Partial: ...`, `Ready`) and `System`, what the machine lacks for that env (`lacks ffmpeg, portaudio`).
3. On a remote machine without a clone, **Git Clone** clones this clone's origin into the profile's Remote Project Path. The machine needs git, and access to that URL.
4. Select the row of the env and press **Create Env**.
5. When `System` names something, press **Install Tools**: it installs it with `apt-get` through `sudo` on Debian, Ubuntu and Raspberry Pi OS, and with `brew` on a Mac. `sudo` is answered with the password of the machine's SSH profile, or, on this machine, with the one saved under `Launcher → System Settings → Credentials → Sudo (local admin)`. For `asr-base` do this before the next step, since pip builds PyAudio against PortAudio.
6. **Install Deps** installs the env's extra of `pyproject.toml` into it. The row then reads `Ready`.

The command console at the bottom of the tab runs a typed command on the machine **Host** names, in its clone. Docker on the uber server and the GPU server is installed by hand ([Docker guide](docker.md)); the console runs plain `docker compose` there, so the SSH user has to be in the `docker` group.

A capture device that only streams or records needs no env. A Raspberry Pi needs an SSH profile, a user in the `video` and `audio` groups, and FFmpeg: **Start** on a Streams tab installs FFmpeg and tmux there when they are missing, and the Collection card copies its recorder code there by itself but installs nothing, so a Pi that only records for the Collection card gets FFmpeg by hand, in a shell on the Pi (`sudo apt-get install -y ffmpeg`; see [Raspberry Pi Setup](raspi_config.md)). A Mac that records its own camera or microphone needs someone logged in on its screen, with Terminal allowed under **Privacy & Security → Camera** and **Microphone** (see [Collection](tui.md#collection)).

![Environment tab: the Host selector, the table of conda envs with Status and System columns, and the Connect, Refresh, Git Clone, Git Pull, Git Pull All, Create Env, Install Deps, Install Tools and Delete Env buttons](img/tui/environment.png)

### Point everything at the system services

The pipelines, the dashboard and the console find the system services through the forms under **System Settings → Connections**. Fill them in before starting any service: a System Services card opens on the machine its address names, so the address decides where **Start** runs it.

1. Open `Launcher → System Settings → Connections` with the **Host** selector on `Local`. On a new machine every address reads `<uber-server>`, a placeholder that counts as not set ([A new machine](system_services.md#a-new-machine)).
2. In each form, replace `<uber-server>` with the name or address by which every machine reaches the uber server. Use the value the **Host** field of the uber server's SSH profile holds: the console matches each address against the profiles, and opens the cards on the one it matches. The fields:
    - **InfluxDB**: `url` (`http://uber-server:8086`), `org` `admin` (the form shows it empty) and `bucket` `mmla-data`, what the Docker stack creates. The `token` is filled in by **Fetch Token** in the [next section](#start-the-system-services).
    - **MongoDB**: `url` (`mongodb://uber-server:27017`).
    - **Redis**: `host`; `db` `1` (the form shows `0`), which keeps the dashboard's job queue out of the default database.
    - **MQTT (Mosquitto)**: `host`.
    - **Gateway (Nginx)**: `host`, also when Nginx will not run: every pipeline config carries a `Gateway` section, and **Start** refuses a pipeline whose sections name no machine.
    - **Stream Server (MediaMTX)**: `host`, for streamed cameras and microphones.
    - **Dashboard (Flask)**: `host`, the machine the dashboard runs on.
3. Press **Save** on each form. It writes the section into `config/system_services.yml` and into every pipeline config of this machine that carries it; the pipeline Config tabs then show those fields read-only. Every field is listed in [Pointing the pipelines at the services](system_services.md#pointing-the-pipelines-at-the-services).

Whenever more than one machine is involved, the addresses name a machine, never `localhost`: every machine reads `localhost` as itself, so a base station would look for Redis on itself and never hear the START sent from Session Control. For a first try on one machine, `localhost` in every form is right.

Which machines need these settings before their first Start:

- **Base stations**: nothing to do. The first **Start** of a base card on a remote machine copies the settings into its pipeline config and says `Relaunch <card> once the sync above completes.`; press **Start** again.
- **The dashboard's machine**, when it is not this one: the dashboard reads the `config/system_services.yml` of the machine it runs on. On the InfluxDB, MongoDB, Redis and Stream Server (MediaMTX) forms, pick that machine in the host picker beside **Sync to Host** and press **Sync to Host**. Do the InfluxDB form after **Fetch Token** has stored the token.
- **A second console**: it takes the settings with **Sync from Host** on each form (see [A new machine](system_services.md#a-new-machine)).

![System Settings → Connections → InfluxDB form: the url, token, org and bucket fields with Save and Reset to Defaults, and the host picker with Sync to Host and Sync from Host](img/tui/connections.png)

### Start the system services

Which services a session needs:

| Card under `Launcher → System Services` | Collection mode | Pipeline mode |
|---|---|---|
| **InfluxDB** | for the replay | required: every measurement |
| **MongoDB** | required: the session records | required |
| **Redis** | for the replay | required: START and STOP |
| **MQTT (Mosquitto)** | for the replay | required: bases to synchronizers |
| **Stream Server (MediaMTX)** | not used | for cameras and microphones streamed over the network, the server-side recording of a session and the dashboard's live video |
| **Gateway (Nginx)** | optional, for the replay | optional: a load balancer in front of the AI services |
| **Dashboard (Flask)**, **Dashboard (Celery)** | optional | optional: live view, replay and analysis report; the worker takes the report jobs off the web process |

1. **Sudo**: Redis, Mosquitto and Nginx always run natively, through `make` and `sudo`. For a native service on this machine, enter its sudo password under `Launcher → System Settings → Credentials → Sudo (local admin)` and **Save**, unless it was saved for **Install Tools** already; on a remote machine the console answers `sudo` with the password of its SSH profile.
2. **Docker secrets**: InfluxDB in Docker needs `docker/.env` on its machine before its first start, with an `INFLUXDB_INIT_PASSWORD` of 8 characters or more (without it the container restarts in a loop). Type this into the command console of the Environment tab with **Host** on the uber server; it fills both secrets with random values, and does nothing where `docker/.env` exists already:

    ```bash
    test ! -e docker/.env && cp docker/.env.example docker/.env && chmod 600 docker/.env && sed -i.bak -e "s|^INFLUXDB_INIT_ADMIN_TOKEN=.*|INFLUXDB_INIT_ADMIN_TOKEN=$(openssl rand -hex 32)|" -e "s|^INFLUXDB_INIT_PASSWORD=.*|INFLUXDB_INIT_PASSWORD=$(openssl rand -base64 24)|" docker/.env && rm docker/.env.bak
    ```

    The values are read once, when the stack starts against an empty volume (see [Setup](docker.md#setup)). For the dashboard's live video and sound from a Stream Server in Docker, also set `MEDIAMTX_WEBRTC_HOSTS` in that file to the name or address browsers reach the uber server at ([Camera tiles and live video](dashboard.md#camera-tiles-and-live-video)).
3. Open each card under `Launcher → System Services`. It opens on the machine its address in System Settings names, and the sidebar heading says which (`System Services @ uber-server`).
    - **InfluxDB**, **MongoDB** and **Stream Server (MediaMTX)** have a **Run mode**: `docker`, the default, runs `docker compose -f docker/docker-compose.infra.yml up -d <service>` on the card's machine; `native` runs the make target instead, for a machine that already runs these as brew or systemd services.
    - **Redis**, **MQTT (Mosquitto)** and **Gateway (Nginx)**: **Start** runs `make -C pipelines/uber-server <service>`, which installs the package when the machine has none (brew on macOS, apt on Debian and Ubuntu) and makes it listen on every interface ([Listener configuration](system_services.md#listener-configuration)).

    Press **Start** on InfluxDB, MongoDB, Redis and MQTT (Mosquitto), then on Stream Server (MediaMTX) when cameras or microphones are streamed. The Gateway and the dashboard need a config first (step 5).

    ![Launcher tab: the sidebar tree with System Settings, System Services, Collection and Pipelines, and the InfluxDB card on the right with its status, Run mode, Start, Stop, Logs, Refresh and Fetch Token](img/tui/launcher.png)

4. **InfluxDB token**: press **Fetch Token** on the InfluxDB card. It reads the admin token of the Docker stack on the card's machine and stores it encrypted as `InfluxDB.token` in this machine's settings; the log shows only its first and last four characters. When the dashboard runs on another machine, press **Sync to Host** on the InfluxDB form for it now. A `native` InfluxDB has no such token: run its first setup at `http://uber-server:8086` (org `admin`, bucket `mmla-data`) and type its token into the InfluxDB form ([InfluxDB](system_services.md#influxdb)).
5. **Gateway and dashboard**: Gateway (Nginx) and Dashboard (Flask) read a `config.yml` on their machine, which a fresh clone does not have. With the card's **Host** on that machine, **Save** on the card's **Config** tab writes it. Both also need the `uber-server` env there, in which Nginx's config is rendered and the dashboard runs ([Prepare every machine](#prepare-every-machine)). Then **Start** Gateway (Nginx) when the bases are to reach the AI services through it, Dashboard (Flask), which runs in a tmux session named `flask`, and Dashboard (Celery), named `celery`.
6. **Check**: each card's status line, the `(R)` marker after its name in the sidebar, and the **Status** tab, which lists every service with its host, status and port. A card's status is a connection from this console to the address in System Settings, which is the path the pipelines take.
7. Open the dashboard at `http://uber-server:5050`. The header of its Sessions page says whether InfluxDB, MongoDB, the report worker and the stream server answer ([Dashboard](dashboard.md#sessions)).

    ![Dashboard Sessions page: the header with the state of InfluxDB, MongoDB, the report worker and the stream server, the search, Speech and Sort filters, and the sessions grouped by month with their data, speech setup and Replay and Analysis buttons](img/dashboard/sessions.png)

The dashboard has no login, and it serves the archived recordings of every session to anyone who reaches its port: keep it on a trusted network, or turn the recordings off ([Raw recordings](dashboard.md#raw-recordings)).

### Describe the study

The cards create every session in an experiment group, so the study is described before the first session.

1. **Tasks**: `Launcher → System Settings → Study → Tasks`. The repository ships a few task definitions in `config/tasks/`. **Create Task** with a name under **New task name** adds another, edited as YAML and written with **Save**. An experiment's **Task Type** picks from the tasks, so create the task first.
2. **Experiments**: `Launcher → System Settings → Study → Experiments`. Type the id under **New experiment ID** and press **Create Experiment**. The id starts the id of every session of the experiment, `<experiment>_<group>_<YYMMDDTHHMMZ>`, and with it the names of the session's folders: up to 64 ASCII letters, digits, `_`, `-` and `.`, starting with a letter or digit. The form `exp_<YYYYMMDD>_<task type>` (`exp_20261001_lego_building`) is what the dashboard reads a session's date and task from. Once a session carries the id, it cannot change.
3. In the experiment, set **Title** and **Task Type**, and leave **Status** on `active`, which a new experiment starts on: only active experiments are offered on the cards.
4. Under **Add Participant**, one entry per participant: **Name**; **Group ID** (`group_01`), the group they are in; **Tag ID**, the AprilTag they wear, by which IPS and VFA tell them apart (printable tags are in `pipelines/ips-base/apriltag/`); **Description**, a short appearance note (`grey hoodie, left of the table`), which goes into the VFA prompts. Press **Add** for each, then **Save & Back**.
5. The **Experiment Group** dropdown of the Collection and base cards now lists `<experiment>/<group>` for every group of every active experiment. A session created from a card belongs to the group picked there, and its MongoDB document carries that group's participants.

![System Settings → Study → Experiments: one experiment with its ID, Title, Task Type and Status, its participants with group and tag, and the Add Participant fields](img/tui/experiments.png)

The names and descriptions are personal data. `config/experiments.yaml` is gitignored, the session documents in MongoDB carry the names too, and **Sync to Host** on the Experiments form copies the whole file: sync it only to machines that should hold it. The dashboard shows participants by their tag (`Tag 0`), never by name.

## Two ways to collect a session

| | Collection mode | Pipeline mode |
|---|---|---|
| What runs during the session | one FFmpeg recorder per camera and microphone, on the machine it is plugged into, started from the Collection Session card | the bases and synchronizers on the base stations, the AI servers on the GPU server, usually streams through the Stream Server |
| AI servers | not during the session; needed for the replay | ASR Server for ASR, VFA Server for VFA (IPS needs none) |
| Streams | none: each recorder opens its device | cameras and microphones pushed to the Stream Server and pulled by the bases, or devices of the base station itself |
| Session Control | not used: **Start Audio**, **Start Video** and **Stop** on the card | **Send START** and **Send STOP** |
| What you get | raw recordings per machine, with manifests; measurements once they are replayed | measurements in InfluxDB as the session runs, live on the dashboard |
| When to choose it | no GPU server at hand, a first try on one machine, analysis settings chosen afterwards | results during the session, a live view |

A pipeline session keeps its raw audio and video too, when its devices are streamed: the Stream Server records the streams of a running session from START to STOP, and a stream with **Record** `yes` on its Streams tab also records on its capture device ([Recording](rtmp_streaming.md#recording)). **Sessions → Export** gathers both. A device is recorded by the Collection card or streamed, not both at once.

## Collection mode

The Collection card needs MongoDB, where each session is registered, and an experiment group from [Describe the study](#describe-the-study). See [Collection](tui.md#collection) for every field.

### Plan the recorders

1. Open `Launcher → Collection → Collection Session`. Each recorder runs on the machine its row names, so the **Host** selector at the top plays no part here and reads `Per recorder`.
2. **Session ID**: `Create MongoDB Session`, and **Experiment Group**: the group being recorded. The first Start creates the session in that group.
3. **Output Root**: `artifacts`, the default, puts every machine's recordings into `artifacts/<session>/collection/<host>/` of its own disk (`~/artifacts` on a remote machine).
4. On the **Audio** tab, set **Num Audio** to the number of microphones (`+` and `-`; 0 records none). Each recorder is a row; click a cell, or press Enter on it, to open its dropdown:
    - **Host**: the machine the microphone is plugged into, `Local` or an SSH profile.
    - **Device**: the microphones that machine reports, asked over SSH the first time, each with its channel count where the machine says it.
    - **Channel**: `mix` (every channel averaged into one mono file), `each` (a file per channel, for a receiver of several worn microphones) or one channel (`ch0`).
    - **Device Label**: the name the recording's files give the device, offered from the `Streams` entries of the pipeline configs (`jabra-1`); **type another…** takes any other. A name whose entry names the machine that captures it fills **Host** and **Device** too.
    - **Participant**: whose voice it records: a participant of the group (`<name> (tag <id>)`), **Group (room microphone)**, or **bind later**. A recorder that writes several channels gets a row per channel, each with its own Participant.
5. On the **Video** tab, set **Num Video** to the number of cameras, and give each row its **Host**, **Device**, **Device Label** and **Rotate** (`180°` for a camera mounted upside down; FFmpeg turns the picture as it records).

**Start** refuses two recorders on one device of one machine, and one tag picked on two rows.

![Collection Session card, Audio tab: Num Audio, the Start Audio, Stop, Stop All Hosts, Logs, Refresh, Download and Delete Remote buttons, the Session ID, Experiment Group and Output Root fields, and the recorder table with Host, Device, Channel, Device Label and Participant](img/tui/collection.png)

### Record

1. Press **Start Audio** on the Audio tab. Start creates the session, copies the recorder code to each remote machine of the card (a recording machine needs no clone), and opens one terminal window per recorder on this screen. A recorder with a Device starts at once and asks nothing; a machine that cannot be reached is named in the log and left out, and the others record.
2. Press **Start Video** on the Video tab. The card now shows the session Start Audio created, so the cameras record into the same session.
3. Check: the card reads `Running` while a recorder runs on one of its machines, each window shows its recorder's output, and **Logs** lists the live recorders of each machine.
4. At the end, press **Stop**. It stops every recorder of the session on every machine of the card, on both tabs, and marks the session ended in MongoDB. **Stop All Hosts** also reaches every other SSH profile, for a recorder started from another console. The card goes back to `Create MongoDB Session`, so the next Start is a new take.

### Collect the files

The recordings of a remote machine stay on it until they are fetched:

- **Download** on the card copies the recordings of every remote machine of the card into this console's `artifacts/<session>/collection/<host>/`. The transfer resumes where it stopped, a progress bar with **Cancel** shows under the log, and a second press fetches only what is new ([Downloading a session](tui.md#downloading-a-session)).
- **Sessions → Export** runs the same transfer for every machine the session recorded on, with the rest of the session, and needs no card. This is the step for a finished session: see [Export and archive](#export-and-archive).

A session that is still recording downloads as far as it has been written, and the files still growing stay staged: stop the recorders, then download again. Once the recordings are here and archived, **Delete Remote** removes them from the remote machines of the card: the first press names each machine and the session, the second deletes.

### Analyze the recordings later

A collected session is analyzed by replaying its recordings through the pipelines: a Pipeline-mode session whose bases read files instead of devices or streams. It needs the system services that Pipeline mode needs, and the AI servers of ASR and VFA.

1. Do the [one-time part of each pipeline](#set-up-the-pipelines) to be replayed.
2. Export the session to this console ([Export and archive](#export-and-archive)), so its recordings are on this machine, where the bases of the replay run. This machine then needs the env of each pipeline it replays ([Prepare every machine](#prepare-every-machine)).
3. On the **Config** tab of each base card (ASR Base, IPS Base, VFA Base), with **Host** `Local`, give the `Bases` entries `source: file` and, as `source_index`, the full path of a recording (**Browse…** picks it). The bases take the start time from the file names.
4. Run it as in [Run a session](#run-a-session), without the streams and with the base cards on `Local`: pick the recorded session under **Session** (a Start into a session that has ended is held back once; press **Start** again to go on in it), then **Send START**, and **Send STOP** once the bases have replayed their files.

Each pipeline guide says which files its bases replay: [ASR](pipelines/asr.md#post-time-processing), [IPS](pipelines/ips.md#post-time-processing), [VFA](pipelines/vfa/index.md#post-time-processing).

## Pipeline mode

### Set up the pipelines

Each pipeline has a part that is done once per deployment, and again only when something it sets changes:

- **ASR**: the ASR Server's config and its Start on the GPU server, the ASR Base config (one `Bases` entry per microphone, the `Streams` they pull, the service endpoints), and the speaker profiles where bases verify speakers. See [ASR → Once per deployment](pipelines/asr.md#once-per-deployment).
- **IPS**: the cameras' intrinsics, their streams and `Bases` entries, the camera sync, and the transform matrices on every base station. See [IPS → Once per deployment](pipelines/ips.md#once-per-deployment).
- **VFA**: the VFA Server's config and its Start on the GPU server (and the MLLM Server for a local model), and the VFA Base config. See [VFA → Once per deployment](pipelines/vfa/index.md#once-per-deployment).

The AI servers are started here and keep running from session to session.

### Run a session

The three pipelines run a session in the same order. Each pipeline's own fields are listed in its guide: [ASR → Every session](pipelines/asr.md#every-session), [IPS → Every session](pipelines/ips.md#every-session), [VFA → Every session](pipelines/vfa/index.md#every-session).

1. **Check what runs.** The **Status** tab lists the system services and the AI servers (`Running 4/6` for a server with four of its six containers answering).
2. **Start the streams the bases pull.** On each base card's **Streams** tab, **Start All** starts every stream of the card that the console runs, or **Start** the selected row. Wait until each row's **Stream Server** column reads `● live`. Streams come before bases: a base opens its stream as it starts (an ASR base at START), waits up to 30 s for one that is not up, and then stops ([Bases: pulling a stream](rtmp_streaming.md#bases-pulling-a-stream)).
3. **Start the base cards.** On each base card (ASR Base, IPS Base, VFA Base), with **Host** on the base station that runs those bases:
    - **Session**: on the first card, `Create MongoDB Session` with the **Experiment Group**. Its Start creates the session, and every other base card, on any machine, then opens on that session: keep it, so all bases of the take are in one session.
    - A session has one synchronizer per pipeline. On the cards of that pipeline's other base stations set **Num Synchronizers** to 0, and on the card that runs it set **Sync Waits For** (ASR and VFA) to the number of the session's bases on every machine.
    - **Start** opens one terminal window per base and synchronizer. Nothing is asked in them; each starts and waits for START. Before that, Start asks the Stream Server whether the streams the bases pull are live, and holds back once when one is not: start it, or press **Start** again to go ahead. On a remote machine whose settings are behind, the first Start only brings them up to date and says `Relaunch <card> once the sync above completes.`: press **Start** again.
4. **Send START.** Open `Launcher → Pipelines → Session Control`. Its list opens on the session the base cards were last started into. Keep the pipelines that run ticked (**ASR**, **IPS**, **VFA**), and once every window reports that it is waiting, press **Send START**. START also switches on the Stream Server's recording of the session's streams.

    ![Session Control: the session list with its refresh button, the ASR, IPS and VFA checkboxes, the Redis and MongoDB addresses, and the Send START and Send STOP buttons](img/tui/session-control.png)

5. **Watch it.** On the dashboard, `http://uber-server:5050`, the session appears in the **Live now** band of the Sessions page; its Live page follows the session as it is written: the transcript, the room plan of the badges, the camera tiles and the indicators ([Live](dashboard.md#live)).

    ![Dashboard Live page, here replaying an ended session: the replay controls, the health strip, the indicators, the room plan, the transcript and the activity timeline](img/dashboard/live.png)

6. **Send STOP** at the end. Every base and synchronizer of the session ends its run and exits by itself, also when STOP comes before START. STOP marks the session ended in MongoDB, stops the Stream Server's recording of its streams, and puts the base cards back on `Create MongoDB Session` for the next take.
7. **Stop the streams** once no further session pulls them: **Stop All** on the card's Streams tab, or **Stop All** on the **Streams** tab of `Launcher → System Services → Stream Server (MediaMTX)`, which stops the streams of every card and asks for a second press. A stream needs no restart between sessions, so leave it running for the next take.
8. The AI servers keep running until **Stop** on their cards, which runs `docker compose down`.

Then [export and archive](#export-and-archive) the session.

## Export and archive

Both modes end here. See [Sessions tab](tui.md#sessions-tab) for every part.

1. Open the **Sessions** tab. The table lists the sessions MongoDB knows and the session folders of the machine its **Host** selector picks (`Local` by default). Select the session.
2. Press **Export**. It gathers everything of the session onto this console, into `artifacts/<session>/`:
    - **Measurements**: the session's events from InfluxDB as JSON, a text transcript, and `<session>_parameters.json`, what every base and synchronizer ran with.
    - **Collection recordings**: the recordings of every machine the session recorded on.
    - **Streams**: the session's part of each stream its bases took, from the Stream Server over HTTP, and from the capture device when it recorded the stream too.
    - **Base files**: what the bases and synchronizers wrote on their machines: logs, the config they ran with, what they stored.

    A progress row names what is being fetched from where, and **Cancel** stops it. Pressing **Export** again resumes, and fetches only what is missing; the log ends with a summary per part. `mmla ses-export <session>` does the same from a shell.

    ![Sessions tab: the session table with Session ID, Experiment, Group, Status, Started, Recordings until, Files here and Source, and the Refresh, Export, Archive, End Session, Delete Session and Delete Files buttons](img/tui/sessions.png)

3. Export a pipeline session within three days. The Stream Server deletes each ten-minute segment of its recordings three days after it began (`recordDeleteAfter: 72h` in `pipelines/uber-server/mediamtx/mediamtx.yml`, **Keep recordings for** on the Stream Server card's **Config** tab), and the **Recordings until** column says when that starts for each session. A capture device keeps its stream recordings for as long as each stream's `record_keep_days` says (`0`, the default, keeps them), and a Collection recorder's files stay on their machine until **Delete Remote**.
4. Press **Archive**. It sends the session's raw files from this console to the archive host, the machine of **System Settings → Dashboard (Flask)** (else of the Stream Server), into `artifacts/<session>/` of its clone. Every file is checked there by its sha256 before it is moved into place, and the session's MongoDB document notes the archive as `complete`, or `partial` while something is missing; pressing **Archive** again completes it. Nothing is deleted on either side, and speaker profiles are never sent. The archive host needs an SSH profile (unless it is this machine) and `python3`. The archived recordings are what the dashboard's Analysis page lists for download and playback, and what Export falls back on for a file its machine no longer holds. `mmla ses-archive <session>` does the same from a shell.

    ![Dashboard Analysis page of an ended session: the Overview indicators, the session timeline and the participants table, under the Overview, Speech, Space, Attention and Data and exports tabs](img/dashboard/analysis.png)

5. **End Session** is for a session left `active`, because its console closed or its bases went down before STOP. The first press says the end it would write; the second writes it.
6. To free space, **Delete Files** deletes the session's folders on the machine the **Host** selector picks, and nowhere else; the first press names each folder with its size, the second deletes. The Collection card's **Delete Remote** removes its recorders' files from their machines, and **Manage** on a Streams tab deletes the stream recordings of the capture devices. **Delete Session** is not a cleanup: it deletes the session everywhere central, its archive, its InfluxDB events and its MongoDB document, and cannot be undone.

## Keeping up to date

1. Quit the console with `q`, run `git pull` in its clone, and start `mmla tui` again: a running console keeps the code it started with.
2. **Git Pull All** on the Environment tab pulls on every SSH profile at once. Offline machines are skipped, and the last line says which machines pulled, were up to date or failed. A machine where a **Sync to Host** changed a file that git tracks (a prompt, a task, `mediamtx.yml`) stops on that file when the pull changes it, until the change is committed or undone there ([Service cards](tui.md#service-cards)).
3. When the pull changed `pyproject.toml`, press **Install Deps** again for each env on each machine; an env that lacks a package reads `Partial: ...`, and its `[E]` marker in the Launcher turns yellow. For the console, run `pip install -e '.[tui]'` in its `tui` env.
4. **Start** on the ASR Server and VFA Server cards runs `docker compose up -d --build`, which rebuilds the images from the pulled code and recreates the containers whose image changed.
5. **Stop** and **Start** Dashboard (Flask) and Dashboard (Celery): a running dashboard keeps the code it started with.
6. A Stream Server in Docker reads `mediamtx.yml` only when its container starts: after a pull that changes it, **Stop** and **Start** its card between sessions, since that closes every stream for a moment.
