# Management console (TUI)

`mmla tui` opens a terminal console that configures, launches and monitors every OpenMMLA component, on this machine or on other machines over SSH. Everything it launches is also an `mmla` command you can run by hand; the pipeline guides list them.

The [Quickstart](../quickstart.md) gives the order of a first setup and of a session. These pages describe each panel of the console.

## Install and start

Install the [prerequisites](../prerequisites.md) first. The console runs from a clone of the repository, because it edits the configs under `pipelines/` and `config/` and runs the compose files under `docker/`.

```bash
git clone https://github.com/ucph-ccs/OpenMMLA.git
cd OpenMMLA
conda create -n tui python=3.10 -y
conda activate tui
pip install -e '.[tui]'
mmla tui
```

With this editable install, `mmla tui` finds the repository from the installed package and works from any directory.

??? info "Details: a non-editable install"
    The package is also on PyPI (`pip install "openmmla[tui]"`). A non-editable install reads `pipelines/`, `config/` and `docker/` from the current directory, so start it from the root of a clone.

## The four tabs

| Tab | What it does |
|---|---|
| [Environment](environment.md) | the conda environments on a host, the tools they need, and git on remote hosts |
| [Launcher](launcher/index.md) | System Settings, the system services, Collection and the pipelines, one card each |
| [Sessions](sessions.md) | the sessions, with Export, Archive, End Session and the deletes |
| [Status](status.md) | what runs where |

## Keys

| Key | What it does |
|---|---|
| `q` | quit |
| **Ctrl+C** | copy the selection, in a field or text dragged over with the mouse |
| **Ctrl+V** | paste |

On a Mac both go through the Mac's clipboard, so text moves to and from other apps. **Cmd+C** does not copy there, because Terminal keeps that key for itself.

## Hosts

The Environment tab and the Launcher have a **Host** selector: `Local`, or one of the SSH profiles under `System Settings → Hosts → SSH Profiles` ([SSH Profiles](launcher/system-settings.md#ssh-profiles)). Every action then runs on that host.

- A path is resolved on a remote host as `<remote_project_path>/<the same path relative to the repository root>`, so the host needs a clone at the path its profile names.
- The `↻` button beside the selector tests every profile, and the profiles are tested again every 30 seconds. A host that fails reads `(offline ✗)` and cannot be picked until it answers.
- The selector always lists this console's own profiles, also while the SSH Profiles form shows another host's.

### How each card picks its host

The parts of a deployment rarely share one machine, so every leaf of the Launcher has its own host.

| Where | Opens on | To change it |
|---|---|---|
| Pipeline and MLLM Server cards | the host the card was last pointed at, `Local` the first time; kept per card in `config/launcher_hosts.yml` | pick another host |
| Collection card | each recorder has its own **Host** in the recorder table; the selector reads `Per recorder` | pick a row's Host ([Collection](launcher/collection/index.md#recorder-table)) |
| System Services cards | the machine whose address System Settings gives the service (`InfluxDB.url`, `Redis.host`, ...), matched against this machine's names and addresses and the SSH profiles | pick another host for a one-off; change the address in System Settings to move it for good |
| System Settings forms, except Sudo | one selector shared by the forms, on `Local` in every new console | pick another host |
| Sudo form, Session Control | this machine only; a note says why | — |

??? info "Details: when a card cannot open where it would"
    - A pipeline or MLLM Server card whose host is offline, or whose profile is gone, opens on `Local` and says so in the log. The remembered host is kept for next time.
    - A System Services card moved to another host says that the pipelines are configured for another machine and reports the chosen host's own port. It is back on the configured machine the next time it opens.
    - A system service address that matches no profile, or names an offline host, opens the card on `Local` with the reason in the log.
    - A `localhost` address names no machine, so that card remembers its host like a pipeline card.
    - An address that still reads `<uber-server>` (not filled in yet, see [A new machine](../system_services.md#a-new-machine)) is never looked up. The card stays on the host it was last pointed at, says `System Settings → Connections → Redis has no host yet` (with its own form's name), and reports the port of the host it is on.

## What a remote host needs

A remote host has to offer a POSIX shell over SSH: Linux, macOS or Raspberry Pi OS. The console runs everything there through `bash -lc`, tmux, conda and the usual `test`, `cat` and `mkdir -p`, and the recorders use POSIX file locks and signals.

| You launch | The remote host needs |
|---|---|
| ASR, IPS or VFA bases, camera tools | conda with the pipeline environment and the repository at `remote_project_path`; an ASR base also FFmpeg and PortAudio. The Environment tab creates the environment and installs what the host lacks (**Install Tools**) |
| MLLM Server | conda with the `vfa-vllm` environment, tmux, the repository at `remote_project_path` |
| ASR Server, VFA Server, InfluxDB or MongoDB in `docker` mode | Docker Engine with compose, the NVIDIA container toolkit for the AI stacks, the user in the `docker` group, the repository (for `docker/`) |
| System services in `native` mode | `make` and `sudo`, and tmux for the dashboard, its worker and a native MediaMTX. Start installs Redis, Mosquitto and Nginx when the host has none (Homebrew on macOS, apt on Debian and Ubuntu); InfluxDB and MongoDB need their vendor repositories ([System services](../system_services.md)) |
| Streams | `ffmpeg` and `tmux`; Start installs them when the host lacks them ([Streams tab](launcher/pipelines/streams.md#tools-start-installs)) |
| Collection Session | `python3` and `ffmpeg`; the recorder code is copied to `~/.openmmla/collection-runtime` by itself. A Mac also needs someone logged in on its screen, with Terminal allowed to use the camera and the microphone ([Recording on a Mac](launcher/collection/index.md#recording-on-a-mac)) |

??? info "Details: Windows hosts"
    A Windows machine that answers with `cmd.exe` or PowerShell (Windows' own OpenSSH server) is recognised the first time it is picked. The selector goes back to where it was, the log says why, and the host reads `(Windows: not supported ✗)` from then on.

    With WSL2 as the machine's SSH shell it answers as Linux and runs server-side services (the AI stacks, the databases). Cameras and microphones are not visible inside WSL, so it cannot record a collection session.

## Secrets and the master key

The secrets in the config files (tokens, passwords, API keys) are stored as `ENC(...)`, encrypted with the master key of the machine the file is on, `~/.openmmla/master.key`. Every machine has its own key, a colleague's computer running its own console too. When the console writes a file to another machine, it encrypts the secrets again with that machine's key, and it never copies or replaces a key.

[One master key per machine](../system_services.md#one-master-key-per-machine) lists the keys that are encrypted and how the console handles the master keys.

!!! warning "Back up the master key"
    Back the key up with its machine. Without it, the encrypted values stored there cannot be read.

## Files the console writes

| Path | What it holds |
|---|---|
| `config/system_services.yml` | the System Settings connections and the sudo password, secrets encrypted; gitignored, template `config/system_services_template.yml` |
| `config/ssh_profiles.yml` | the SSH profiles; gitignored, template `config/ssh_profiles_template.yml` |
| `config/launcher_hosts.yml` | the host each Launcher card was last pointed at; gitignored |
| `config/experiments.yaml` | the experiments and their participants; gitignored, template `config/experiments_template.yaml` |
| `config/tasks/*.yaml`, `config/vfa/action_schemas.yml`, `config/mllm_server.yml` | the task definitions, the VFA action schema, the MLLM Server settings; tracked in git |
| `pipelines/*/config.yml` | the pipeline configs, with the shared sections copied from System Settings; gitignored |
| `pipelines/ips-base/camera_calib/`, `pipelines/ips-base/camera_sync/` | calibration images and transform matrices |
| `artifacts/<session>/` | a session's recordings, exported measurements and manifests; `streams/server/` and `streams/capture/` hold its part of its streams, `pipelines/<pipeline>/<host>/` what its bases wrote ([Export](sessions.md#export)) |
| `artifacts/streams/capture/<day>/<host label>/` | the recordings of the streams captured on this machine with Record on, filed by day and tied to no session ([Recordings and Manage](launcher/pipelines/streams.md#recordings-and-manage)) |
| `artifacts/<session>/.staging/` | partly downloaded remote data and its resume ledger; removed when a download completes, swept after 14 days |
| `~/.openmmla/master.key` | this machine's encryption key, never copied to another machine |
| `~/.openmmla/streams/` | stream start times |
| `~/.openmmla/collection-runtime/` | the recorder code copied to a remote host |

## Pages in this guide

- [Environment tab](environment.md): conda environments, system tools, git.
- [Launcher tab](launcher/index.md): the tree and its markers, service cards, Sync to Host and Sync from Host.
    - [System Settings](launcher/system-settings.md): SSH profiles, experiments, tasks, connections, sudo.
    - [System Services](launcher/system-services.md): the service cards, and the Stream Server's Streams and Recordings tabs.
    - [Collection](launcher/collection/index.md): record raw audio and video, and download it.
        - [Bringing recordings in](launcher/collection/session-tools.md): `mmla ses-import`, `ses-tidy` and `ses-align`.
    - [Pipelines](launcher/pipelines/index.md): base cards, server cards, the MLLM Server, IPS calibration, Session Control.
        - [Streams tab](launcher/pipelines/streams.md): start, stop and record the streams of a pipeline.
- [Sessions tab](sessions.md): export, archive, end and delete sessions.
- [Status tab](status.md): what runs where.
