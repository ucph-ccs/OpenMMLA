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
```bash
git clone https://github.com/ucph-ccs/openmmla.git
cd OpenMMLA
```

### Step 2: Setup Dashboard Flask Backend

1. **Install backend dependencies**:

```bash
conda create -n uber-server -c conda-forge python=3.10.12 -y
conda activate uber-server
pip install -e .[uber-server]
cd pipelines/uber-server/dashboard
```

2. **Configure the backend**:
Edit the configuration file at `flask-backend/config.yml` to set up your database connection and other settings:

```yml
InfluxDB:
  # Replace <inlfuxdb-example-server> with your actual sever address that runs the InfluxDB server, and the <port>
  # with the actual port number that InfluxDB server is listening to. The default port for InfluxDB is 8086.
  # Replace <influxdb-token> with your actual InfluxDB token, and <org-name> with your actual InfluxDB organization name.
  # =================================================================================
  # e.g., if your InfluxDB server is on uber-server.local machine and its IP address is 192.168.1.12 and listens on 8086,
  # and it has a token named lrn6PxfmD_Sbde1Ke3LL5QOIc-QQJLT9Tw58-Y1pHXWfcsdljWMfwSOJ8F3CP3c20qYPiJ9UL8bSslELPW27lw==
  # and organization named admin, then the following line should be:
  # url: http://uber-server.local:8086
  # token: lrn6PxfmD_Sbde1Ke3LL5QOIc-QQJLT9Tw58-Y1pHXWfcsdljWMfwSOJ8F3CP3c20qYPiJ9UL8bSslELPW27lw==
  # org: admin
  url: http://<influxdb-example-server>:<port>
  token: <influxdb-token>
  org: <org-name>

Redis:
  # Replace <redis-example-server> with your actual sever address that runs the Redis server, and the <port-number>
  # with the actual port number that Redis server is listening to. Specify a certain database number you want to use for
  # message queue. The default database number is 0. The default port for Redis is 6379.
  # =================================================================================
  # e.g., if your Redis server is on uber-server.local machine and its IP address is 192.168.1.12, and it has listening
  # on port 6379, then the following line should be:
  # host: uber-server.local
  # port: 6379
  # db: 1
  host: <redis-example-server>
  port: <port-number>
  db: <database-number>
```

### Step 3: Set Up the Next.js Frontend

1. **Install dependencies**:

```bash
cd next-frontend
npm install
```

2. **Configure the frontend**:
Edit `next-frontend/.env.local`: replace the `<flask-example-server>` with your
actual flask backend server hostname or IP address, e.g., `uber-server.local` or 
`192.168.1.12`.

```
# .env.local
NEXT_PUBLIC_FLASK_BACKEND=<flask-example-server>
NEXT_PUBLIC_FLASK_PORT=5000
```

Edit `next-frontend/next.config.js`: replace the `<flask-example-server>` with your
actual flask backend server hostname or IP address, e.g., `uber-server.local` or
`192.168.1.12`.

```javascript
module.exports = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: '<flask-example-server>',
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

Run the dashboard using the provided Makefile commands from the main `pipelines/uber-server` directory:

```bash
# go back to pipelines/uber-server directory if you are at pipelines/uber-server/dashboard/next-frontend
cd ../..
# Start the Flask backend on port 5000 using Gunicorn with the gevent worker
make flask 
# Start the Celery Workers for Flask backend to process background tasks (generate visualisations)
make celery
# Start the Next.js Frontend on port 3000
make next
```

## Accessing the Dashboard

Once all services are running, you can access the dashboard by opening a web browser and navigating to:

```bash
# Access from dashbaord server
http://localhost:3000   

# Access from other devices under the same network, replace `<dashboard-example-server>` with your actual dashboard 
# server address or hostname.
http://<dashboard-example-server>:3000
```

## Troubleshooting

1. **Port conflicts**:
   If you encounter port conflicts, you can clean specific ports:
   ```bash
   # Clean port if it conflicts with 5000
   make clean-ports 5000
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
   Ensure InfluxDB and Redis is running and accessible and your `flask_backend/config.yml` file is correctly setup. 

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

