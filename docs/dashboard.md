# Dashboard Setup Guide

The dashboard shows a session live and as a report afterwards. It is a Flask backend served by gunicorn, a Celery worker that renders the post-time visualizations, and a dependency-free static frontend (plain HTML, CSS and JavaScript) served by the same Flask process. No Node.js, npm or build step is involved.

## Prerequisites

- The `uber-server` conda environment: create it from the TUI's Environment tab, or by hand:

    ```bash
    conda create -n uber-server python=3.10 -y
    conda activate uber-server
    pip install -e '.[uber-server]'
    ```

- Redis, as the Celery broker and result store.
- InfluxDB, for the measurements. MongoDB is not needed by the dashboard itself.

Both are described in [System Services](system_services.md).

## Configuration

The backend reads `pipelines/uber-server/dashboard/flask-backend/config.yml`. Its `InfluxDB`, `MongoDB` and `Redis` sections are among the files that the TUI fills from **System Settings → Connections**, so saving the connections there configures the dashboard too. To edit it by hand, copy `config_template.yml` next to it and fill in:

```yaml
InfluxDB:
  url: http://<influxdb-host>:8086
  token: <influxdb-token>
  org: admin
  bucket: mmla-data

Redis:
  host: <redis-host>
  port: 6379
  db: 1          # a database number other than 0, so the Celery queue stays apart from the session control bus
```

The backend listens on port 5050 by default. Change it under **System Settings → Connections → Dashboard (Flask)**; the TUI passes it to `make flask` as `DASHBOARD_PORT` and the Status tab probes that host and port. The frontend needs no configuration: it lives in `pipelines/uber-server/dashboard/frontend/` and talks to the backend over same-origin URLs.

## Running

From the TUI: **Launcher → System Services → Dashboard (Flask)** and **Dashboard Worker (Celery)**, Start on each. From a shell:

```bash
cd pipelines/uber-server
make flask DASHBOARD_PORT=5050   # gunicorn with the gevent worker, in a tmux session named flask
make celery                      # the visualization worker, in a tmux session named celery
```

Then open `http://localhost:5050` on the server, or `http://<dashboard-host>:5050` from any device on the network.

## Pages

- **Session explorer** (`/`): every session found in InfluxDB, each openable live or as a report.
- **Live view** (`/realtime?session=<id>`): the running transcript (who said what, when), speaking-participation bars, the physical proximity map with badge positions and interaction links, and the raw diarization log.
- **Analysis report** (`/posttime?session=<id>`): triggers the visualization job and polls until it is ready, then shows the gallery (interaction networks, heatmap, trajectories, interactive diarization) with zoom and per-image download, plus the measurement logs for download.

## Troubleshooting

- **Port in use**: `make clean-ports 5050`.
- **Stop**: `make stop-flask`, `make stop-celery`, or the Stop buttons on the two cards.
- **Logs**: the Logs button on the cards, or attach to the tmux sessions (`tmux attach -t flask`, `tmux attach -t celery`; `Ctrl+B` then `D` detaches).
- **No sessions or empty charts**: check that InfluxDB and Redis are reachable from the dashboard host and that `flask-backend/config.yml` carries the right token, org and bucket. The Status tab of the TUI shows what it can reach.
