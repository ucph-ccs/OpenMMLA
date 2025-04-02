#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
CONDA_ENV="uber-server"

ports=(8086 6379 1883 8080 3000 5000)
services=('influxdb' 'redis' 'mosquitto' 'nginx' 'next' 'flask' 'celery')

# Loop through each port and kill processes using those ports
for i in "${!ports[@]}"
do
    port="${ports[$i]}"
    service="${services[$i]}"
    echo "Checking for processes using port $port ($service)"

    PIDs=$(sudo lsof -ti:"$port" | xargs)
    if [ -n "$PIDs" ]; then
        echo "$PIDs" | tr " " "\n" | while read -r PID; do
            if [ -n "$PID" ]; then
                echo "Killing process on port $port with PID $PID"
                sudo kill -9 "$PID"
            fi
        done
    else
        echo "No process found on port $port ($service)"
    fi
done

# Tmux session command for each service
declare -a commands=(
    "source activate $CONDA_ENV && influxd"
    "source activate $CONDA_ENV && redis-server --protected-mode no"
    "source activate $CONDA_ENV && brew services restart mosquitto"
    "bash -c \"$BASH_DIR/../nginx/nginx.sh; exec bash\""
    "cd $BASH_DIR/../dashboard/next-frontend && source activate $CONDA_ENV && npm run start"
    "cd $BASH_DIR/../dashboard/flask-backend && source activate $CONDA_ENV && gunicorn -k gevent -w 1 -b 0.0.0.0:5000 dashboard:app"
    "cd $BASH_DIR/../dashboard/flask-backend && source activate $CONDA_ENV && celery -A dashboard.celery worker --loglevel=info"
)

# Loop for creating tmux session for each service
for i in "${!services[@]}"
do
    session_name="${services[$i]}"
    command="${commands[$i]}"

    # Check if the session already exists, close it if so
    tmux has-session -t "$session_name" 2>/dev/null

    if [ $? == 0 ]; then
        echo "Sending Ctrl+C to session: $session_name"
        tmux send-keys -t "$session_name" C-c
        tmux kill-session -t "$session_name"
    fi

    tmux new-session -d -s "$session_name" -n "$session_name" bash
    tmux send-keys -t "$session_name" "$command" C-m
done

# Switch to nginx session for password input
tmux switch -t "nginx"
