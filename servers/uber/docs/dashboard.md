# Dashboard Setup Guide

This document outlines the setup and configuration of the OpenMMLA dashboard, which consists of a Flask backend, a Next.js frontend, and Celery for background task processing.

## Prerequisites

Before setting up the dashboard, ensure you have the following prerequisites installed:

- Conda (for managing Python environments)
- Redis (required for Celery)
- InfluxDB (required for accessing measurements data)
- [Node.js 18+ and npm](https://nodejs.org/en/download)

## Installation

### Step 1: Clone the Repository

If you haven't already done so, clone the OpenMMLA repository:

```bash
git clone https://github.com/ucph-ccs/openmmla.git
cd openmmla
```

### Step 2: Navigate to the Dashboard Directory

The dashboard components are located in the `servers/uber/dashboard` directory:

```bash
cd servers/uber/dashboard
```

### Step 3: Set Up the Flask Backend

1. **Create and activate a Conda environment**:

```bash
conda create -n uber-server python=3.10.12 -y
conda activate uber-server
```

2. **Install dependencies**:

```bash
pip install -r flask-backend/requirements.txt
```

3. **Configure the backend**:

Edit the configuration file at `flask-backend/config.yml` to set up your database connection and other settings:

```yml
InfluxDB:
  url: <your-influxdb-url> e.g., http://uber-server.local:8086
  token: <your-influxdb-operator-token> e.g., eNMnz5EuJlWNukW15iI8ys7
  org: <your-organization-name> e.g., admin

Redis:
  host: <your-redis-server-hostname> e.g., uber-server.local
  port: <your-redis-server-port> e.g., 6379
  db: <your-database-number> e.g., 1
```

### Step 4: Set Up the Next.js Frontend

1. **Install dependencies**:

```bash
cd next-frontend
npm install
```

2. **Configure the frontend**:

Update the `.env.local` file in the `next-frontend` directory: replace the `uber-server.local` with your Flask backend's hostname or IP address.

```
# .env.local
NEXT_PUBLIC_FLASK_BACKEND=uber-server.local
NEXT_PUBLIC_FLASK_PORT=5000
```

Update the `next.config.js` file in the `next-frontend` directory: replace the `uber-server.local` with your Flask backend's hostname or IP address.

```javascript
module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'uber-server.local',
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
    ];
  },
};
```

## Running the Dashboard

The easiest way to run the dashboard is using the provided Makefile commands from the main `servers/uber` directory:

```bash
cd ../  # Go back to servers/uber directory

# Start the Flask Backend on port 5000 using Gunicorn with the gevent worker
make flask 

# Start the Celery Worker for Flask that process background tasks
make celery

# Start the Next.js Frontend on port 3000
make next
```

## Accessing the Dashboard

Once all services are running, you can access the dashboard by opening a web browser and navigating to:

```bash
# Access from dashbaord server
http://localhost:3000   

# Access from other devices under the same network, replace `uber-server.local` with your dashbaord server's hostname or IP address
http://uber-server.local:3000
```

## Troubleshooting

1. **Port conflicts**:
   If you encounter port conflicts, you can clean specific ports:
   ```bash
   # Clean port if it conflicts with 5000
   make clean-ports PORT=5000
   ```

2. **Service management**:
   ```bash
   # Stop services if it already exists
   make stop-flask    
   make stop-celery   
   make stop-next     
   ```

3. **Log files**:
   Check the tmux sessions for logs:
   ```bash
   tmux attach -t flask  # View Flask logs
   tmux attach -t next   # View Next.js logs
   tmux attach -t celery # View Celery logs
   ```
   Press `Ctrl+B` then `D` to detach from a tmux session.

4. **Database / Redis issues**:
   Ensure InfluxDB and Redis is running and accessible and your config.yml file is correct 

## Dashboard Features

The dashboard provides the following features:

1. **Real-time visualization**:
   - View active sessions
   - Track participant conversations
   - Track participant positions

2. **Post-time visualization**:
   - Review past sessions
   - Generate diagram and interactive visualizations
   - Export data for further analysis

