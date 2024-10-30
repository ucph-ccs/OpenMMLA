#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
EXAMPLES_DIR="$BASH_DIR/../examples"
PYHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="audio-server"
ports=(5000 5001 5002 5003 5004 5005)
services=("transcribe" "separate" "infer" "enhance" "vad" "resample")

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
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 1 -b 0.0.0.0:5000 --pythonpath $PYHON_PATH serve_speech_transcriber:app"
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 3 -b 0.0.0.0:5001 --pythonpath $PYHON_PATH serve_speech_separator:app"
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 3 -b 0.0.0.0:5002 --pythonpath $PYHON_PATH serve_audio_inferer:app"
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 3 -b 0.0.0.0:5003 --pythonpath $PYHON_PATH serve_audio_enhancer:app"
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 3 -b 0.0.0.0:5004 --pythonpath $PYHON_PATH serve_vad:app"
  "cd $EXAMPLES_DIR && source activate $CONDA_ENV && gunicorn -k gevent -w 1 -b 0.0.0.0:5005 --pythonpath $PYHON_PATH serve_audio_resampler:app"
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
tmux kill-session -t "audio-services"