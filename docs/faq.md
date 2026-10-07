# FAQ

Known problems outside a single pipeline, with their fixes. Problems of one pipeline or service are in the **Troubleshooting** section of its page, such as [System services](system_services.md#troubleshooting) and [Deploy the dashboard](dashboard/deploy.md#troubleshooting).

## Installation

### PyAudio does not build { #pyaudio-installation-errors }

PyAudio compiles against PortAudio. Install `build-essential` and `portaudio19-dev` (apt) or `portaudio` (Homebrew) first ([Prerequisites](prerequisites.md#git-tmux-portaudio-and-ffmpeg)), or press **Install Tools** on the Environment tab, then **Install Deps** again.

### pydub cannot find FFmpeg { #pydub-installation-errors }

pydub runs FFmpeg. Install FFmpeg and make sure it is on the `PATH` of the env's shell ([Prerequisites](prerequisites.md#git-tmux-portaudio-and-ffmpeg)).

### librosa fails to load audio { #librosa-installation-errors }

librosa reads audio through soundfile, which needs `libsndfile1` on Debian, Ubuntu and Raspberry Pi OS, and through FFmpeg for other formats. Install both ([Prerequisites](prerequisites.md#ubuntu-debian-and-raspberry-pi-os)).

### Speech separation gives poor results { #speech-separation-doesnt-perform-well }

The cause is usually a PyTorch and modelscope pair that do not match. Run the speech separator in Docker, whose image pins a working pair (`torch==2.4.1`, `modelscope[framework]==1.16.1`); see the [Docker guide](docker.md).

### Speech separation does not load { #speech-separation-couldnt-load }

With `ImportError: cannot import name '_datasets_server' from 'datasets.utils'`, install an older `datasets` in the separator's env:

```bash
pip install datasets==2.18.0
```

## Network

### Server name not known

A `<name>.local` name needs mDNS and works on one LAN only. Across networks, use a name every machine resolves, such as a Tailscale MagicDNS name, or an address. When a `.local` name stops resolving on the LAN, restart mDNS on the server:

=== "macOS"

    1. Check that the firewall lets InfluxDB, `redis-server` and Mosquitto accept connections, and that **System Settings → General → Sharing → Remote Management** is on.
    2. Restart mDNSResponder and flush the DNS cache:

        ```bash
        sudo killall -HUP mDNSResponder
        # or: sudo killall -STOP mDNSResponder && sudo killall -CONT mDNSResponder
        sudo dscacheutil -flushcache
        ```

    3. Check with `ps aux | grep mDNSResponder` that the process has a new PID.

=== "Linux"

    1. In `/etc/avahi/avahi-daemon.conf`, set `publish-workstation=yes` and `publish-domain=yes`.
    2. Restart Avahi, enable it at boot, and check it:

        ```bash
        sudo systemctl restart avahi-daemon
        sudo systemctl enable avahi-daemon
        sudo systemctl status avahi-daemon
        ```

    3. Flush the nscd cache, where nscd runs:

        ```bash
        sudo /etc/init.d/nscd restart
        sudo nscd -i hosts
        ```

When that does not help, restart the network, and reboot the router and the machines.

### A port is already in use { #socket-address-already-in-use-when-running-audio-bases-on-mac }

A process from an earlier run still holds the port, often the UDP or TCP port of an audio base. Find it and stop it:

```bash
sudo lsof -i :50004                  # macOS
sudo netstat -tulnp | grep 50004     # Linux
kill -9 $(lsof -ti:50004)
```

`make -C pipelines/uber-server clean-ports 50004` does the same. For Redis, see [System services → Troubleshooting](system_services.md#troubleshooting).

### The AI services cannot download their models { #server-couldnt-download-the-model }

The first start of the AI services downloads the models, which can time out when every service starts at once. Start the failing service alone, with only that service selected on the ASR Server or VFA Server card, or with `docker compose -f docker/docker-compose.asr.yml up -d <service>`, and try again.

## Devices

### A badge does not reconnect to its base { #badge-reconnection-mechanism-not-work-as-expected-in-mac-audio-base }

A `tcp` base waits for its badge to connect before anything else, and hears no STOP while it waits. Free the base's port ([A port is already in use](#socket-address-already-in-use-when-running-audio-bases-on-mac)) and start the base again.

### Update the Wi-Fi firmware of an Arduino badge { #update-arduinos-wi-fi-firmware }

Follow Arduino's guide, [Update Wi-Fi firmware on Portenta H7 boards](https://support.arduino.cc/hc/en-us/articles/4403365234322-Update-Wi-Fi-firmware-on-Portenta-H7-boards).

### Connect a Smraza fisheye camera to a Raspberry Pi { #connect-smraza-fisheye-camera-to-raspberry-pi }

Enable the camera interface in `sudo raspi-config`; this [video tutorial](https://www.youtube.com/watch?v=iyITuOcHCjg) shows the steps.

### A window crashes on macOS with an `objc` fork error { #video-module-visualizer-fails-to-initialize }

The error reads `+[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called`. Turn off the fork safety check of the Objective-C runtime in `~/.zshrc`, and open a new shell:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## GPU servers

### Install the NVIDIA driver { #nvidia-driver-installation }

On Ubuntu, `sudo ubuntu-drivers install` installs the recommended driver (`sudo ubuntu-drivers autoinstall` where `ubuntu-drivers` has no `install` command); [this guide](https://www.murhabazi.com/install-nvidia-driver) walks through it. Docker also needs the NVIDIA container toolkit ([Host requirements](docker.md#host-requirements)).

### CUDA unknown error { #nvidia-cuda-unknown-error }

With `CUDA unknown error - this may be due to an incorrectly set up environment` when a server starts, check that the GPU has power, and reboot.

### NVML driver and library version mismatch { #nvidia-nvml-driverlibrary-version-mismatch }

An `apt upgrade` replaced the NVIDIA driver while the old one is still loaded. Reboot. When the error stays, see [this answer](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch#comment73133147_43022843).
