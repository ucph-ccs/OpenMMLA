# Nginx Setup Guide

Nginx is optional. It is the **load balancer** in front of the ASR and VFA AI services: one HTTP entry point (port 8080 by default) so base stations only need the gateway address and requests can be spread over several servers.

Streams no longer go through Nginx: cameras and microphones publish to [MediaMTX](rtmp_streaming.md), which has an address of its own and may run on another machine. The Nginx address is set once under **System Settings → Connections → Gateway (Nginx)** in the TUI (`host`, `http_port`, `scheme`) and synced into every pipeline config.

## Installation

```bash
# macOS (config: /opt/homebrew/etc/nginx/nginx.conf)
brew install nginx

# Ubuntu / Debian (config: /etc/nginx/nginx.conf)
sudo apt update && sudo apt install -y nginx
```

No extra module is needed; the RTMP module that earlier versions of this guide asked for can be dropped.

## Configuration

The Nginx config is rendered from a Jinja2 template. Three files under `pipelines/uber-server/nginx/` are involved:

1. `config.yml`: your upstreams (copy `config_template.yml` to start).
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
```

The endpoint names (`infer`, `resample`, `enhance`, `separate`, `transcribe`, `vad`, `vllm`) must match the service endpoint names; the default service ports are 5001 to 5006 for ASR and 5007 for VFA. The template generates an upstream block and a matching `location` for every service that is defined. With the port check (the default) the render tries each server's port. A server that answers is listed, and so is one whose machine answers but runs nothing on that port yet (the connection is refused). A machine that does not answer at all (off, out of reach, or behind a firewall that drops the connection) is left out, as is a name that does not resolve; a service none of whose machines answers gets no `location`, and Nginx answers `404` for it. Nginx does the rest itself: a listed server that fails is skipped for `fail_timeout` (30 s) and then tried again with a real request, so a server that starts after Nginx, or restarts, is used without starting the Gateway again, and one that stops is skipped. Its workers share what they learn of the servers (`zone`). A machine that goes off after the render holds one request every 30 s for up to `proxy_connect_timeout` (3 s) before it goes to the next server: start the Gateway again, or take the machine out of the list, while it stays off. When none of a service's servers answers, Nginx answers `502`, which the bases retry. The render prints what it did with each server. An `rtmp_apps` key left over from the RTMP days is ignored.

## Running

The Makefile in `pipelines/uber-server` renders the config, installs it and (re)starts Nginx. It activates the `uber-server` conda environment for the render step, so create that environment first (TUI Environment tab, or `pip install -e '.[uber-server]'`).

```bash
cd pipelines/uber-server
make nginx          # render with an upstream port check, install, reload/restart
make nginx false    # skip the check: list every server whose name resolves
make stop-nginx
```

From the TUI, the same targets run behind **Launcher → System Services → Gateway (Nginx)**.

### macOS firewall

Go to **System Settings → Privacy & Security → Firewall** and make sure Nginx is allowed to accept incoming connections.

## Troubleshooting

1. **Port conflicts**

   ```bash
   make clean-ports 8080
   ```

2. **Error logs**

   ```bash
   tail -f /opt/homebrew/var/log/nginx/error.log   # macOS
   tail -f /var/log/nginx/error.log                # Linux
   ```

3. **Check the installed config** before blaming Nginx: `sudo nginx -t`.

4. **`404 Not Found (the Gateway has no route to it)`** in a base's log, or a speaker registration that says the Gateway has no route: Nginx was started while none of that service's machines could be reached, so it has no `location` for it. Press **Start** on the Gateway card again once one is on. `502 Bad Gateway (the Gateway has no server for it that answers)` means the route is there and its servers are down: start the ASR or VFA Server card, or read its Logs.

## Official links

- Nginx: https://nginx.org/en/docs/install.html
