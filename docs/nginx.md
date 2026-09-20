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

The endpoint names (`infer`, `resample`, `enhance`, `separate`, `transcribe`, `vad`, `vllm`) must match the service endpoint names; the default service ports are 5001 to 5006 for ASR and 5007 for VFA. The template generates an upstream block and a matching `location` for every service that is defined, with a server line for every machine of it whose name resolves — running or not, on or off. What is up when you render decides nothing, so the Gateway does not have to be started again after the ASR or VFA server: Nginx takes up a server that starts later by itself. A server that fails is skipped for `fail_timeout` (30 s) and then tried again with a real request, one that stops is skipped, and its workers share what they learn of the servers (`zone`). The cost of a machine that is off is one request every 30 s held for up to `proxy_connect_timeout` (2 s), which `proxy_next_upstream` then sends to the next server; take a machine you do not run out of the list to save even that. An upstream of one server is never skipped at all (Nginx ignores `max_fails` for it), so a list that names only the machines you actually run is also the fastest to pick a service up. While none of a service's servers answers, Nginx answers `502`, which the bases retry; a name that does not resolve is left out, as Nginx would not start with it, and a service not one of whose names resolves gets a `location` that answers `503` rather than no route at all. With the port check (the default) the render also tries each server's port, and prints for every one whether something answers there, nothing does yet, or the machine is silent. An `rtmp_apps` key left over from the RTMP days is ignored.

## Running

The Makefile in `pipelines/uber-server` renders the config, installs it and reloads Nginx, starting it when it is not running. A reload, not a restart: the old workers answer the requests they hold, so a session in progress keeps its answers, and it clears what Nginx learned of the servers, so one that has just started is tried at once instead of after `fail_timeout`. The rendered config is tested with `nginx -t` before the reload, and the one that was running comes back if the new one does not pass. The render step runs in the `uber-server` conda environment, so create that environment first (TUI Environment tab, or `pip install -e '.[uber-server]'`).

```bash
cd pipelines/uber-server
make nginx          # render with an upstream port check, install, test, reload
make nginx false    # skip the check: list every server whose name resolves
make stop-nginx
```

From the TUI, the same targets run behind **Launcher → System Services → Gateway (Nginx)**. Starting the **ASR Server** or **VFA Server** card renders and reloads the Gateway too, on the machine `Gateway.host` names, so a machine whose name did not resolve at the last render is routed without pressing Start on the Gateway card; it is skipped, with a line in the log, while the Gateway itself is not running. The **MLLM Server** does not do this: the frame analyzer calls it straight, not through the Gateway.

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

4. **`404 Not Found (the Gateway has no route to it)`** in a base's log, or a speaker registration that says the Gateway has no route: the running config has no `location` for that service, so its endpoint is missing from `upstreams` in `nginx/config.yml`, or the config was rendered before this behaviour existed. Press **Start** on the Gateway card and read the summary the render prints. `502 Bad Gateway (the Gateway has no server for it that answers)` means the route is there and its servers are down: start the ASR or VFA Server card, or read its Logs. `503` means no machine of that service has a name that resolves right now.

## Official links

- Nginx: https://nginx.org/en/docs/install.html
