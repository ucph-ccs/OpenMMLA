# RTMP Streaming
This readme introduces how to setup your RTMP server, and how to stream from clients to your RTMP server, and how to record from the RTMP server via OBS.

## Server side
If you haven't setup your RTMP server, here is a guide for how to setup your RTMP server manually.

### Installation

```bash
# macOS
brew tap denji/nginx
brew install nginx-full --with-rtmp-module

# Ubuntu
sudo apt update && install nginx libnginx-mod-rtmp
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
On your devices which you want to stream data to the RTMP server, could be Raspberry Pi or PCs.

### Installation
 ```bash
 sudo apt update
 sudo apt install ffmpeg
 ```

### Stream to RTMP

```bash
# Check your ip address or hostname on your Mac server
ifconfig | grep inet
hostname

# Get your video & audio device details
# Ubuntu
v4l2-ctl --list-devices
arecord -l
# macOS
ffmpeg -f avfoundation -list_devices true -i ""
 
# Video Streaming
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> \
  -c:v libx264 -b:v 1M -preset ultrafast -tune zerolatency \
  -maxrate 2M -bufsize 2M \
  -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>

e.g.
# Ubuntu
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
  -c:v libx264 -b:v 1M -preset ultrafast -tune zerolatency \
  -maxrate 2M -bufsize 2M \
  -f flv rtmp://ericli.local/stream_01

# macOS
ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:none" \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -maxrate 4000k -bufsize 4000k \
  -f flv rtmp://ericli.local/stream_01

ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "0:none" \
  -c:v h264_videotoolbox -b:v 2000k -preset ultrafast \
  -f flv rtmp://ericli.local/stream_01
  
# Audio Streaming
ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> \
  -c:a aac -b:a 128k \
  -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>

e.g.
ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:3,0 \
  -c:a aac -b:a 128k \
  -f flv rtmp://ericli.local/stream_01


# Both (more laggy since two source into one port)
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> \
  -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> \
  -c:v libx264 -b:v 1M \
  -bufsize 2M -maxrate 2M \
  -preset ultrafast -tune zerolatency \
  -c:a aac -b:a 128k -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>

e.g.
ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 \
  -f alsa -ac 2 -ar 44100 -i plughw:3,0 \
  -c:v libx264 -b:v 1M \
  -bufsize 2M -maxrate 2M \
  -preset ultrafast -tune zerolatency \
  -c:a aac -b:a 128k -f flv rtmp://uber-server.local/stream_01
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
