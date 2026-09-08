# RTMP Streaming

How to run an RTMP server, push camera and microphone streams to it with FFmpeg, and record the streams with OBS or FFmpeg.

In a deployment managed from the TUI you rarely type these FFmpeg commands yourself: a `Streams` entry with an `ssh_profile`, `device` and `target` in the pipeline config makes the Launcher's **Streams** tab start and stop the same kind of FFmpeg process on the remote device over SSH. The commands below are what runs under the hood, and what to use for devices without SSH access.

## Server side

The RTMP server is Nginx with the RTMP module. The full install and the templated configuration are in the [Nginx Setup Guide](nginx.md); the manual configuration below is the minimum for streaming only.

### Installation

```bash
# macOS
brew tap denji/nginx
brew install nginx-full --with-rtmp-module

# Ubuntu
sudo apt update && sudo apt install -y nginx libnginx-mod-rtmp
```

### Configuring RTMP
```bash
# Edit configuration file
# macOS
sudo nano /opt/homebrew/etc/nginx/nginx.conf

# Ubuntu
sudo nano /etc/nginx/nginx.conf


# Add an RTMP configuration block outside the http block:
rtmp {
    server {
        listen 1935;  # Standard port for RTMP
        chunk_size 4096;

        # camera 1
        application stream_01 {
            live on;    # stream option
            record off; # storage option
        }

        # Add more stream applications as needed
    }
}

# Save the file and reload Nginx:
sudo nginx -s reload
```

## Client side

On the devices that stream to the RTMP server (Raspberry Pis or PCs).

### Installation
 ```bash
 sudo apt update
 sudo apt install ffmpeg
 ```

### Stream to RTMP
1. Check your IP address or hostname on your RTMP server
   ```sh
   # macOS
   ifconfig | grep inet
   hostname
   
   # Ubuntu
   ip addr show
   hostname -I
   ```
2. Get your video & audio device details
   ```sh
   # macOS
   ffmpeg -f avfoundation -list_devices true -i ""
     
   # Ubuntu & Debian
   v4l2-ctl --list-devices
   arecord -l
   ```
3. Publish the stream to RTMP server
   ```sh
   # Video Streaming
   ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> \
     -c:v libx264 -preset ultrafast -tune zerolatency \
     -g 30 -keyint_min 30 -sc_threshold 0 \
     -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
     -b:v 1M -maxrate 2M -bufsize 2M \
     -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<App-Name>/<Stream-Name>
   
   # Example on Ubuntu/Debian:
   ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
     -c:v libx264 -preset ultrafast -tune zerolatency \
     -g 30 -keyint_min 30 -sc_threshold 0 \
     -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
     -b:v 1M -maxrate 2M -bufsize 2M \
     -f flv rtmp://mac-01.local/ips/1
   
   # Example on macOS:
   ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "0:none" \
     -c:v libx264 -preset ultrafast -tune zerolatency \
     -f flv rtmp://mac-01.local/ips/m
     
   # Or using hardware encoder (macOS with h264_videotoolbox)
   ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "0:none" \
     -c:v h264_videotoolbox -preset ultrafast -tune zerolatency \
     -b:v 2M \
     -f flv rtmp://mac-01.local/ips/1
   
   # Audio Streaming only
   ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> \
     -c:a aac -b:a 128k \
     -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<App-Name>/<Stream-Name>
     
   e.g.
   ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:3,0 \
     -c:a aac -b:a 128k \
     -f flv rtmp://mac-01.local/ips/1
   
   # Audio + Video Streaming
   ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> \
     -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> \
     -c:v libx264 -preset ultrafast -tune zerolatency \
     -g 30 -keyint_min 30 -sc_threshold 0 \
     -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
     -b:v 1M -maxrate 2M -bufsize 2M \
     -c:a aac -b:a 128k \
     -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<App-Name>/<Stream-Name>
   
   # Example on Ubuntu:
   ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
     -f alsa -ac 2 -ar 44100 -i plughw:2,0 \
     -c:v libx264 -preset ultrafast -tune zerolatency \
     -g 30 -keyint_min 30 -sc_threshold 0 \
     -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
     -b:v 1M -maxrate 2M -bufsize 2M \
     -c:a aac -b:a 128k \
     -f flv rtmp://mac-01.local/ips/1
   ```

## Record the streams with OBS
You can record the streams from RTMP server via the OBS on your devices by following the instructions:

1. Download OBS

2. Configure firewall
    + Go to **System Preferences** -> **Privacy & Security** and ensure that nginx is allowed to receive incoming
      connections.

3. Open OBS
    - Add a **Media Source** for each stream via **Sources** -> **+**.
    - Uncheck local file and enter the RTMP URL: `rtmp://localhost/stream_XX` (replace XX with the stream number).

4. Do for other streams as well

5. Sync Video and Audio
    - Add delay to the video source if needed using a video async filter.
    - Adjust the sync offset for audio sources to match the video.

6. Start recording

## Record the streams with FFMPEG
  + Video recording on Ubuntu
    ```sh
    START_TIME=$(date +"%s.%3N")
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
      -c:v libx264 -preset veryfast -tune zerolatency \
      -g 30 -keyint_min 30 -sc_threshold 0 \
      -x264-params "keyint=30:min-keyint=30:no-scenecut=1:repeat-headers=1" \
      -b:v 3M -maxrate 6M -bufsize 10M \
      -f mp4 "record_${START_TIME}.mp4"
    ```
  
  + Video recording on Mac
    ```sh
    START_TIME=$(python3 -c 'import time; print(f"{time.time():.3f}")')

    ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "0:none" \
      -c:v h264_videotoolbox -realtime true \
      -g 30 -b:v 2M -maxrate 2M -bufsize 4M \
      -movflags +faststart \
      -f mp4 "record_${START_TIME}.mp4"
    ```

## Troubleshooting
1. Check that the stream is being published: open `http://<rtmp-server-host>:8080/stat` (needs the `stat.xsl` from the Nginx guide)
2. Check the stream is in correct format and can be read by ffmpeg and ffplay
   ```sh
   ffmpeg -i rtmp://<host>/<app>/<stream> -f null -
   e.g.
   ffmpeg -i rtmp://mac-01.local/ips/1 -f null -
   
   ffplay -fflags nobuffer -analyzeduration 0 -loglevel verbose rtmp://<host>/<app>/<stream>
   e.g.
   ffplay -fflags nobuffer -analyzeduration 0 -loglevel verbose rtmp://mac-01.local/ips/1
   ```