# Nginx Setup Guide

Nginx is optional. It plays two roles in OpenMMLA:

- **Load balancer**: one HTTP entry point (port 8080 by default) in front of the ASR and VFA AI services, so base stations only need the gateway address and requests can be spread over several servers.
- **RTMP server**: ingest point (port 1935) for camera and microphone streams pushed with FFmpeg from Raspberry Pis or PCs; the bases then pull from `rtmp://<gateway>/<app>/<stream>`. See [RTMP Streaming](rtmp_streaming.md).

The gateway address is set once under **System Settings → Gateway** in the TUI (`host`, `http_port`, `rtmp_port`, `scheme`) and synced into every pipeline config.

## Installation

### Load balancer only

```bash
# macOS (config: /opt/homebrew/etc/nginx/nginx.conf)
brew install nginx

# Ubuntu / Debian (config: /etc/nginx/nginx.conf)
sudo apt update && sudo apt install -y nginx
```

### Load balancer + RTMP

Use this build instead if you need RTMP streaming:

```bash
# macOS
brew tap denji/nginx
brew install nginx-full --with-rtmp-module

# Ubuntu / Debian
sudo apt update && sudo apt install -y nginx libnginx-mod-rtmp

# stat.xsl renders the RTMP statistics page at http://<host>:8080/stat
sudo mkdir -p /usr/local/nginx/html
sudo curl -o /usr/local/nginx/html/stat.xsl https://raw.githubusercontent.com/arut/nginx-rtmp-module/master/stat.xsl
```

## Configuration

The Nginx config is rendered from a Jinja2 template. Three files under `pipelines/uber-server/nginx/` are involved:

1. `config.yml`: your upstreams and RTMP apps (copy `config_template.yml` to start).
2. `nginx.conf.j2`: the template.
3. `nginx.generated.conf`: the rendered file (gitignored) that is copied over the system `nginx.conf`.

Edit `config.yml`:

```yaml
# load balancer: one entry per AI service endpoint, one server line per host that runs it
upstreams:
  transcribe:
    - host: server-01.local
      port: 5005
      weight: 3
    - host: 192.168.1.12
      port: 5005
      weight: 1

# RTMP: one application per stream group (only if you need streaming)
rtmp_apps:
  - ips
  - vfa
```

The endpoint names (`infer`, `resample`, `enhance`, `separate`, `transcribe`, `vad`, `vllm`) must match the service endpoint names; the default service ports are 5001 to 5006 for ASR and 5007 for VFA. The template generates an upstream block and a matching `location` for every service that is defined and reachable, and an `rtmp` block when `rtmp_apps` is not empty.

## Running

The Makefile in `pipelines/uber-server` renders the config, installs it and (re)starts Nginx. It activates the `uber-server` conda environment for the render step, so create that environment first (TUI Environment tab, or `pip install -e '.[uber-server]'`).

```bash
cd pipelines/uber-server
make nginx          # render with an upstream port check, install, reload/restart
make nginx false    # skip the port check
make stop-nginx
```

From the TUI, the same targets run behind **Launcher → System Services → Nginx**.

### macOS firewall

Go to **System Settings → Privacy & Security → Firewall** and make sure Nginx is allowed to accept incoming connections.

## Troubleshooting

1. **Port conflicts**

   ```bash
   make clean-ports 8080 1935
   ```

2. **Error logs**

   ```bash
   tail -f /opt/homebrew/var/log/nginx/error.log   # macOS
   tail -f /var/log/nginx/error.log                # Linux
   ```

3. **Check the installed config** before blaming Nginx: `sudo nginx -t`.

4. **RTMP statistics**: `http://<host>:8080/stat` lists the live publishers and their bitrates.

## Official links

- Nginx: https://nginx.org/en/docs/install.html
- nginx-rtmp-module: https://github.com/arut/nginx-rtmp-module
