#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
EXAMPLES_DIR="$BASH_DIR/../examples"
PYHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="video-server"
ports=(5000)
services=("frame-analyze")

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
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 1 -b 0.0.0.0:5000 --pythonpath $PYHON_PATH serve_video_frame_analyzer:app"
)

# Loop for creating tmux session for each service
for i in "${!services[@]}"
do
    session_name="${services[$i]}"
    command="${commands[$i]}"

    # Check if the session already exists, create if not
    tmux has-session -t "$session_name" 2>/dev/null

    if [ $? == 0 ]; then
        echo "Sending Ctrl+C to session: $session_name"
        tmux send-keys -t "$session_name" C-c
        tmux kill-session -t "$session_name"
    fi

    tmux new-session -d -s "$session_name" -n "$session_name" bash
    tmux send-keys -t "$session_name" "$command" C-m
done

# Close the current tmux session
tmux kill-session -t "video-services"