# Nginx Load Balancer and RTMP Documentation

This document outlines the setup and configuration of Nginx as a load balancer for multiple services and as an RTMP
server.

## Part 1: Load Balancer

### Installation

#### On macOS (Apple Silicon)

   ```
   brew install nginx
   ```

The configuration file will be located at `/opt/homebrew/etc/nginx/nginx.conf`.

#### On Linux (Ubuntu/Debian)

   ```
   sudo apt update
   sudo apt install nginx
   ```

The configuration file will be located at `/etc/nginx/nginx.conf`.

### Configuring the Load Balancer

#### Step 1: Edit the nginx.conf Template

1. Locate the `mbox-uber/conf/nginx.conf` template file in your project directory.

2. Define your upstream services in the `http` block. Each upstream should specify the servers that will handle the
   requests. Use the format `$<server-hostname>:PORT` for each server. For example:

   ```nginx
   upstream transcribe_service {
       server $server-01:5000 weight=3;
       server $server-02:5000 weight=3;
       server $server-03:5000 weight=1;
       keepalive 40;
   }
   ```

3. Add a `location` block in the `server` section for each service:

   ```nginx
   location /transcribe {
       proxy_pass http://transcribe_service;
   }
   ```

4. Repeat this process for all your services (separate, infer, enhance, vad, etc.).

#### Step 2: Configure the nginx.sh Script

1. Modify the line in `mbox-uber/bash/nginx.sh` to specify the server hostnames you use in the `nginx.conf` template:
   ```shell
   SERVER_NAMES=("server-01" "server-02" "server-03")
   ```

#### Step 3: Run the nginx.sh Script

   ```
   cd <your-paht-to>/mbox-uber/bash
   ./nginx.sh
   ```

The script will automatically:

- Resolve the IP addresses of the servers in your local network.
- Replace the placeholders (`$<server-hostname>`) with the actual IP addresses.
- Comment out the unreachable servers.
- Copy the template `nginx.conf` to the system's Nginx configuration directory.
- Start or reload Nginx with the new configuration.

## Part 2: RTMP

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

There are two ways to set up RTMP for our system:

#### Method 1: Direct Configuration

1. Open the system-wide Nginx configuration file:
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

#### Method 2: Using mbox-uber Configuration (Recommended)

1. Edit the RTMP configuration in mbox-uber:
   ```
   mbox-uber/conf/nginx.conf
   ```

2. Add or modify the RTMP configuration block in this file, similar to the one shown in Method 1.

3. Run custom script to update the system-wide Nginx configuration:
   ```sh
   cd <your-path-to>/mbox-uber/bash
   ./nginx.sh
   ```

### Additional Setup for macOS

1. Configure firewall:
    - Go to **System Preferences** -> **Privacy & Security** and ensure that Nginx is allowed to receive incoming
      connections.