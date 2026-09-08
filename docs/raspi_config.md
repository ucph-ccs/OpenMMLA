# Raspberry Pi Setup

A Raspberry Pi can play two roles in OpenMMLA:

- **Streaming device**: it pushes a camera or microphone stream to the RTMP gateway (or straight to an ASR base over UDP) with FFmpeg. The TUI starts and stops that FFmpeg process over SSH, so the Pi only needs FFmpeg and an SSH login. See [RTMP Streaming](rtmp_streaming.md) and the `Streams` section of the pipeline configs.
- **Base station**: it runs an IPS/VFA/ASR base itself, which needs the full [prerequisites](prerequisites.md), a clone of the repository and a conda environment (the TUI's Environment tab can create it remotely).

## Headless setup

1. Flash Raspberry Pi OS onto the microSD card with Raspberry Pi Imager and enable SSH (and set the hostname, user and Wi-Fi) in the imager's settings.

2. If a previous Pi used the same hostname, drop its old host key:

    ```sh
    ssh-keygen -R raspi-01.local
    ```

3. Connect (example hostname `raspi-01`, user `admin`):

    ```sh
    ssh admin@raspi-01.local
    ```

4. Optional: enable the VNC server for a remote desktop:

    ```sh
    sudo raspi-config
    # 3 Interface Options -> I2 VNC
    ```

5. Install the base tools:

    ```sh
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y git tmux ffmpeg v4l-utils alsa-utils
    ```

    `v4l2-ctl --list-devices` and `arecord -l` then show the camera and microphone device names to put in the `Streams` config.

## Streaming device only

Nothing else is needed. Add an SSH profile for the Pi in the TUI (**Launcher → System Settings → Hosts → SSH Profiles**) and reference it from a `Streams` entry with `ssh_profile`, `device` and `target`; the Launcher's Streams tab starts and stops FFmpeg on the Pi.

## Base station

6. Give the Pi access to the repository, either with an SSH key added to your GitHub account or with a personal access token:

    ```sh
    # ssh key
    ssh-keygen -t ed25519 -C "your_email@example.com"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub     # add at https://github.com/settings/keys

    # or a personal access token from https://github.com/settings/tokens
    echo "your_personal_access_token" > ~/.github_pat
    chmod 600 ~/.github_pat
    echo 'export GITHUB_PAT=$(cat ~/.github_pat)' >> ~/.bashrc
    source ~/.bashrc
    ```

7. Clone the repository:

    ```sh
    git clone git@github.com:ucph-ccs/OpenMMLA.git
    # or
    git clone https://$GITHUB_PAT@github.com/ucph-ccs/OpenMMLA.git
    ```

8. Install Miniforge (the conda-forge distribution with aarch64 builds):

    ```sh
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

9. Install the remaining [prerequisites](prerequisites.md#ubuntu-debian-and-raspberry-pi-os), then either create the pipeline environment from the TUI (Environment tab, with the Pi selected as target) or by hand:

    ```sh
    conda create -n ips-base python=3.10 -y
    conda activate ips-base
    pip install -e '.[ips-base]'
    ```

The TUI's SSH profile for the Pi should point `remote_project_path` at the clone (default `~/OpenMMLA`); the first remote launch syncs the pipeline config to it.

## Related

- [FAQ: Connect the smraza fisheye camera to a Raspberry Pi](faq.md#connect-smraza-fisheye-camera-to-raspberry-pi)
- [RTMP Streaming](rtmp_streaming.md)
