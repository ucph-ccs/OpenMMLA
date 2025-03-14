# Frequently Asked Questions
- [Frequently Asked Questions](#frequently-asked-questions)
  - [PyAudio Installation Errors](#pyaudio-installation-errors)
  - [Pydub Installation Errors](#pydub-installation-errors)
  - [Librosa Installation Errors](#librosa-installation-errors)
  - [Server name not known](#server-name-not-known)
    - [Mac server](#mac-server)
    - [Linux server](#linux-server)
  - [Speech separation doesn't perform well](#speech-separation-doesnt-perform-well)
  - [Speech separation couldn't load](#speech-separation-couldnt-load)
  - [Server couldn't download the model](#server-couldnt-download-the-model)
  - [Redis address already in use](#redis-address-already-in-use)
  - [Update Arduino's Wi-Fi firmware](#update-arduinos-wi-fi-firmware)
  - [Badge reconnection mechanism not work as expected in Mac audio base](#badge-reconnection-mechanism-not-work-as-expected-in-mac-audio-base)
  - [Socket address already in use when running audio bases on Mac](#socket-address-already-in-use-when-running-audio-bases-on-mac)
  - [Connect smraza fisheye camera to Raspberry Pi](#connect-smraza-fisheye-camera-to-raspberry-pi)
  - [Video module visualizer fails to initialize](#video-module-visualizer-fails-to-initialize)
  - [Nvidia driver installation](#nvidia-driver-installation)
  - [Nvidia CUDA unknown error](#nvidia-cuda-unknown-error)
  - [Nvidia NVML Driver/library version mismatch](#nvidia-nvml-driverlibrary-version-mismatch)

## PyAudio Installation Errors

To resolve the installation failures of PyAudio, you need to use the C++ library for compiling during installation. If your system is Windows, and Python is 3.7, you can download the whl installation package here: [https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

## Pydub Installation Errors

Installation of pydub requires the installation of ffmpeg and adding it to the system path.

## Librosa Installation Errors

For resolving the installation failures of librosa, you can install it from source. Download the source code from [https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/). For Windows users, you can download the zip package which is easy to decompress.

```cmd
pip install pytest-runner
tar xzf librosa-<version number>.tar.gz or unzip librosa-<version number>.tar.gz
cd librosa-<version number>/
python setup.py install 
```

If you encounter an error like `'libsndfile64bit.dll': error 0x7e`, please install version 0.6.3, such as `pip install librosa==0.6.3`.

Download and install ffmpeg.

+ For Windows, check the blog: [http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/). The author downloaded the 64-bit, static version. Then go to the C drive, decompress it, rename the file as `ffmpeg`, store it in the `C:\Program Files\` directory, and add the environment variable `C:\Program Files\ffmpeg\bin`.

    Finally, modify the source code. The path is `C:\Python3.7\Lib\site-packages\audioread\ffdec.py`, and modify the 32nd line of code as follows:

    ```bash
    COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
    ```

+ For Mac,
  + Install the homebrew first
      ```cmd
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      ```
  + Install ffmpeg
     ```cmd
     brew install ffmpeg
     ```
  + Verify ffmpeg
     ```cmd
     ffmpeg -version
     ```
     
## Server name not known

### Mac server
Check the following options:
1. Firewall should be inactive
2. For mac server, go to setting -> Sharing, enable the remote management
3. Allow influxdb, redis-server, and mosquitto server process to accept connections

After checking the options above, if the problem still exists, you can try the following steps:
+ check the mDNSResponder process with `ps aux | grep mDNSResponder`
+ restart the mDNSResponder
  ```sh
  sudo killall -HUP mDNSResponder
  # or 
  sudo killall -STOP mDNSResponder
  sudo killall -CONT mDNSResponder
  ```
+ flush the DNS cache
  ```sh
  sudo dscacheutil -flushcache
  ```
+ check the mDNSResponder process with `ps aux | grep mDNSResponder`
+ compare the mDNSResponder process with the previous one, if the process is different, 
then the mDNSResponder is restarted successfully

**Restart the network, and reboot the router and devices if the above actions doesn't work**

### Linux server
+ Restart the mDNS service
  ```sh
  sudo systemctl restart avahi-daemon
  ```
+ Enable avahi-daemon service on boot
  ```sh
  sudo systemctl enable avahi-daemon
  ```
+ Check the avahi-daemon service
  ```sh
  sudo systemctl status avahi-daemon
  ```
+ Flush the nscd DNS Cache
  ```sh
  sudo /etc/init.d/nscd restart
  sudo nscd -i hosts
  ```
+ Go to `/etc/avahi/avahi-daemon.conf`, ensure that
  ```sh
  publish-workstation=yes
  publish-domain=yes
  ```

**Restart the network, and reboot the router and devices if the above actions doesn't work** 

## Speech separation doesn't perform well
The reason may be that you are not using the latest pytroch version, right now the pytroch
2.1.1 with the speech separation version 1.9.5 performs well, with 945 components indexed.

## Speech separation couldn't load 
If you find `ImportError: cannot import name '_datasets_server' from 'datasets.utils'`, you can
either downgrade the modelscope to 1.12.0 with `pip install modelscope==1.12.0` or dataset dependencies with
`pip install datasets==2.18.0`.

## Server couldn't download the model
If you run the server script to run all services for the first time, you might experience the failure of downloading the model.
Please go to the server scripts and run the service script separately.

## Redis address already in use
Simply shutdown the redis server if you are on Mac and then start it again
```cmd
redis-cli shutdown
```
```cmd
redis-server --protected-mode no
```

## Update Arduino's Wi-Fi firmware
https://support.arduino.cc/hc/en-us/articles/4403365234322-Update-Wi-Fi-firmware-on-Portenta-H7-boards

## Badge reconnection mechanism not work as expected in Mac audio base
Because the TIME_WAIT doesn't allow reuse the same port as in the last connection (the last socket still in TIME_WAIT state).
Also, when the badges are not connected, it won't receive the stop signal from synchronizer.

## Socket address already in use when running audio bases on Mac
You can manually shut down the process to free up the port, e.g. port 50004 is in use
type in terminal. 
```cmd
sudo lsof -i :50004
```
it will list out the process using the port 50004
```
COMMAND    PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.9 3440 ericli    9u  IPv4 0x2bf903e3e96b6203      0t0  UDP *:50004
```
Then, kill the process with PID
```cmd
kill -9 3440
```
Or directly kill the process by 
`kill -9 $(lsof -ti:50004)`

Or you can use _free_port(**[port]**)_ in _/utils/port.py_ to free up the port. 

## Connect smraza fisheye camera to Raspberry Pi
Check [Youtube Tutorial](https://www.youtube.com/watch?v=iyITuOcHCjg)
```
sudo raspi-config
```

## Video module visualizer fails to initialize
Error message:
```sh
objc[19587]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.
```
If you see the above warning, change your safety setting
```sh
sudo nano ~/.zshrc
# Add this line at the end for multiprocess fork
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## Nvidia driver installation
Check the blog: [Install nvidia driver](https://www.murhabazi.com/install-nvidia-driver)

## Nvidia CUDA unknown error
Error message: 
```sh
Error(s):	CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable
CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.
```
If you see the above error message when starting the server, make sure you are connecting the device to power, and try
reboot.

## Nvidia NVML Driver/library version mismatch
This error happens when we have upgraded the nvidia driver by `sudo apt upgrade`,
you might try to reboot your server first, it should work. If not, try the solution listed in [stackoverflow](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch#comment73133147_43022843).
