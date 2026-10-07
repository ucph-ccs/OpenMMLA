# Nginx

Nginx is the optional Gateway in front of the ASR and VFA services: one HTTP entry point, on port 8080, that spreads the requests over the servers that run each service. Use it when several GPU servers share the work, or to give the bases one address for all the services.

## What you need

- **Nginx** on the Gateway's host, with no extra module; **Start** on the Gateway card installs it when the host has none ([System services → Nginx](system_services.md#nginx-optional)).
- **The `uber-server` conda environment** on that host, which renders the config: create it on the console's **Environment** tab, or with `pip install -e '.[uber-server]'`.
- **Its address**, under `System Settings → Connections → Gateway (Nginx)` (`host`, `http_port`, `scheme`), which the console syncs into every pipeline config.
- **On macOS**, Nginx allowed to accept incoming connections under **System Settings → Privacy & Security → Firewall**.

Streams do not go through Nginx: cameras and microphones publish to the Stream Server ([Streaming](streaming/index.md)).

## Configure the upstreams

The config is rendered from a Jinja2 template. Three files under `pipelines/uber-server/nginx/` take part:

| File | What it is |
|---|---|
| `config.yml` | your upstreams; copy it from `config_template.yml`, or **Save** it on the Gateway card's **Config** tab |
| `nginx.conf.j2` | the template |
| `nginx.generated.conf` | the rendered file, gitignored, copied over the system's `nginx.conf` (`/opt/homebrew/etc/nginx/nginx.conf` on macOS, `/etc/nginx/nginx.conf` on Linux) |

List one entry per service endpoint in `config.yml`, with one line per machine that runs it:

```yaml
# load balancer: one entry per AI service endpoint, one server line per host that runs it
upstreams:
  transcribe:
    - host: gpu-server.local
      port: 5005
      weight: 3
    - host: gpu-server-2.local
      port: 5005
      weight: 1
```

| Endpoint | Service | Default port |
|---|---|---|
| `infer` | AudioInferer | 5001 |
| `resample` | AudioResampler | 5002 |
| `enhance` | SpeechEnhancer | 5003 |
| `separate` | SpeechSeparator | 5004 |
| `transcribe` | SpeechTranscriber | 5005 |
| `vad` | VoiceActivityDetector | 5006 |
| `vllm` | VLLMFrameAnalyzer | 5007 |

| Key | Default | What it does |
|---|---|---|
| `host` | required | the server's name or address |
| `port` | required | the port the service listens on |
| `weight` | required | its share of the requests: a server of weight 3 gets three times the requests of one of weight 1 |
| `rtmp_apps` | | ignored: streams go through MediaMTX |

A service of your own, made with `create_app()`, is listed under the endpoint name it gives there. The template writes an upstream and a matching `location` for every service listed, with a server line for every machine whose name resolves, running or not. So the Gateway need not be started again after an ASR or VFA server: Nginx takes up a server that starts later by itself. List only the machines you run: it is also the fastest way to pick a service up.

??? info "Details: how Nginx picks a server"
    - A server that fails is skipped for `fail_timeout` (30 s) and then tried again with a real request; one that stops is skipped. The workers share what they learn of the servers (`zone`).
    - A machine that is off costs one request every 30 s, held for up to `proxy_connect_timeout` (2 s), which `proxy_next_upstream` then sends to the next server. Take a machine you do not run out of the list to save even that.
    - An upstream of one server is never skipped: Nginx ignores `max_fails` for it.
    - While none of a service's servers answers, Nginx answers `502`, which the bases retry.
    - A name that does not resolve is left out, as Nginx would not start with it. A service none of whose names resolves gets a `location` that answers `503`, rather than no route at all.
    - With the port check, the default, the render also tries each server's port and prints for every one whether something answers there, nothing does yet, or the machine is silent.

## Run it { #running }

Press **Start** on `Launcher → System Services → Gateway (Nginx)`. It runs the Makefile target in `pipelines/uber-server`, which renders the config, installs it, tests it with `nginx -t`, and reloads Nginx, starting it when it is not running. By hand:

```bash
cd pipelines/uber-server
make nginx          # render with an upstream port check, install, test, reload
make nginx false    # skip the check: list every server whose name resolves
make stop-nginx
```

**Start** on the **ASR Server** or **VFA Server** card renders and reloads the Gateway too, on the machine `Gateway.host` names, so a server whose name did not resolve at the last render is routed without pressing **Start** on the Gateway card. While the Gateway is not running, this is skipped with a line in the log. The **MLLM Server** does not do this: the frame analyzer calls it directly, not through the Gateway.

??? info "Details: a reload, not a restart"
    The old workers answer the requests they hold, so a session in progress keeps its answers. The reload clears what Nginx learned of the servers, so one that has just started is tried at once instead of after `fail_timeout`. When the new config does not pass `nginx -t`, the one that was running comes back.

## Troubleshooting

**`404 Not Found (the Gateway has no route to it)`** in a base's log, or a speaker registration that says the Gateway has no route. The running config has no `location` for that service: its endpoint is missing from `upstreams` in `nginx/config.yml`. Add it, press **Start** on the Gateway card, and read the summary the render prints.

**`502 Bad Gateway (the Gateway has no server for it that answers)`.** The route is there and its servers are down. Start the ASR or VFA Server card, or read its **Logs**.

**`503`.** No machine of that service has a name that resolves right now.

**Port 8080 is taken.** Free it, in `pipelines/uber-server`:

```bash
make clean-ports 8080
```

**Nginx misbehaves.** Test the installed config with `sudo nginx -t`, and read the error log:

```bash
tail -f /opt/homebrew/var/log/nginx/error.log   # macOS
tail -f /var/log/nginx/error.log                # Linux
```

See also the [Nginx installation guide](https://nginx.org/en/docs/install.html).
