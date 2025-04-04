# Raspberry Pi RTMP Streaming Setup

## Client side (Raspberry Pi)

1. Install ffmpeg
    ```sh
    sudo apt update
    sudo apt install ffmpeg
    ```

2. Start streaming to RTMP server
    ```sh
    # Check your ip address or hostname on your Mac server
    ifconfig | grep inet
    hostname

    # Get your video and audio device details
    v4l2-ctl --list-devices
    arecord -l

    # Video
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> -c:v libx264 -b:v 1M -bufsize 2M -maxrate 2M -preset ultrafast -tune zerolatency -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>
    
    e.g.
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 -c:v libx264 -b:v 1M -bufsize 2M -maxrate 2M -preset ultrafast -tune zerolatency -f flv rtmp://uber-server.local/stream_01


    # Audio
    ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> -c:a aac -b:a 128k -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>

    e.g.
    ffmpeg -f alsa -ac 2 -ar 44100 -i plughw:3,0 -c:a aac -b:a 128k -f flv rtmp://uber-server.local/stream_01


    # Both (more laggy since two source into one port)
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i <Input_Device> -f alsa -ac 2 -ar 44100 -i plughw:<card_number>,<device_number> -c:v libx264 -b:v 1M -bufsize 2M -maxrate 2M -preset ultrafast -tune zerolatency -c:a aac -b:a 128k -f flv rtmp://<Mac-IP-Address or Mac-Host-Name>/<Stream_ID>

    e.g.
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 -i /dev/video0 -f alsa -ac 2 -ar 44100 -i plughw:3,0 -c:v libx264 -b:v 1M -bufsize 2M -maxrate 2M -preset ultrafast -tune zerolatency -c:a aac -b:a 128k -f flv rtmp://uber-server.local/stream_01
    ```

## Server side

### Installation

#### On macOS (Apple Silicon)

   ```sh
   brew tap denji/nginx
   brew install nginx-full --with-rtmp-module
   ```

#### On Linux (Ubuntu/Debian)

   ```sh
   sudo apt update
   sudo apt install nginx libnginx-mod-rtmp
   ```

### Configuring RTMP

1. Open the Nginx configuration file:
    - On macOS: `/opt/homebrew/etc/nginx/nginx.conf`
    - On Linux: `/etc/nginx/nginx.conf`

2. Add an RTMP configuration block outside the http block:

   ```nginx
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
   ```

3. Save the file and reload Nginx:
   ```sh
   sudo nginx -s reload
   ```

### Record the streams with OBS on Mac

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
