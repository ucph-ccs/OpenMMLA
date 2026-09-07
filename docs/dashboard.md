# Dashboard Setup Guide

This document outlines the setup and configuration of the OpenMMLA dashboard. It consists of a Flask backend, Celery for background task processing, and a dependency-free static frontend (plain HTML/CSS/JS) served directly by Flask — no Node.js, npm, or build step required.

## Prerequisites

- Conda (for managing Python environments)
- Redis (required for Celery)
- InfluxDB (required for accessing measurements data)

The backend listens on port 5050 by default. Change it under **System Settings → Connections → Dashboard (Flask)** in the TUI (the Launcher passes it to `make flask` as `DASHBOARD_PORT`, and the Status tab probes that host:port), or run `make flask DASHBOARD_PORT=<port>` by hand.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/ucph-ccs/openmmla.git
cd OpenMMLA
```

### Step 2: Install Dependencies

```bash
conda create -n uber-server -c conda-forge python=3.10.12 -y
conda activate uber-server
pip install -e .[uber-server]
```

### Step 3: Configure the Backend

Edit `pipelines/uber-server/dashboard/flask-backend/config.yml` to set up your database connection:

```yml
InfluxDB:
  # Replace <influxdb-example-server> with the address of the machine running InfluxDB,
  # and <port> with its port (default 8086). Fill in your token and organization name.
  # e.g. url: http://uber-server.local:8086
  url: http://<influxdb-example-server>:<port>
  token: <influxdb-token>
  org: <org-name>

Redis:
  # Replace <redis-example-server> with the address of the machine running Redis,
  # and <port-number> with its port (default 6379). Pick a database number for the
  # Celery message queue (default 0).
  host: <redis-example-server>
  port: <port-number>
  db: <database-number>
```

That's it — the frontend needs no configuration. It lives in `pipelines/uber-server/dashboard/frontend/` and talks to the backend over same-origin URLs.

## Running the Dashboard

From the `pipelines/uber-server` directory:

```bash
# Start the Flask backend on port 5050 (Gunicorn, gevent worker) — also serves the frontend
make flask
# Start the Celery worker that generates post-time visualizations
make celery
```

## Accessing the Dashboard

```bash
# From the dashboard server
http://localhost:5050

# From other devices on the same network
http://<dashboard-example-server>:5050
```

## Troubleshooting

1. **Port conflicts**:
   ```bash
   make clean-ports 5050
   ```

2. **Service management**:
   ```bash
   make stop-flask
   make stop-celery
   ```

3. **Log files** (tmux sessions; `Ctrl+B` then `D` to detach):
   ```bash
   tmux attach -t flask   # Flask logs
   tmux attach -t celery  # Celery logs
   ```

4. **Database / Redis issues**:
   Ensure InfluxDB and Redis are running and reachable, and that `flask-backend/config.yml` is correctly set up.

## Dashboard Features

1. **Session explorer** (`/`): lists all recorded sessions from InfluxDB; open a session live or as a report.

2. **Live view** (`/realtime?session=<id>`):
   - Live transcript feed (who said what, when)
   - Speaking-participation bars (share of total speaking time per speaker)
   - Physical proximity map (badge positions and interaction links)
   - Raw diarization log

3. **Analysis report** (`/posttime?session=<id>`):
   - Triggers visualization generation automatically and polls until ready
   - Gallery of generated plots (interaction networks, heatmap, trajectories, interactive diarization) with zoom and per-image download
   - Measurement log selection and download
