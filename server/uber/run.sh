#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"

# uber services
services=("influxdb" "nginx" "redis" "flask-backend" "react-frontend" "mosquitto" "celery")

shutdown_server() {
    echo "Shutting down selected tmux sessions..."
    active_sessions=$(tmux list-sessions -F "#S")

    # Filter and shut down the on-going service sessions
    for service in "${services[@]}"; do
        if [[ $active_sessions =~ $service ]]; then  # Check if the service is an active session
            echo "Sending Ctrl+C to session: $service"
            tmux send-keys -t "$service" C-c
            tmux kill-session -t "$service"
        fi
    done
    echo "Uber server shutdown process complete."
}

start_server() {
    tmux new-session -s uber-services "bash -c '$BASH_DIR/bash/uber_services.sh; exec bash'"
}

# Prompt user to start or shutdown the server
while true; do
    echo "Do you want to start or shutdown the server? (Y/n)"
    stty -echo -icanon time 0 min 0
    read -r -n 1 action
    stty echo icanon

    case "$action" in
        [Yy])
            echo "Starting the server..."
            start_server
            break
            ;;
        [Nn])
            echo "Shutting down the server..."
            shutdown_server
            break
            ;;
        *)
            echo "Invalid option. Please enter 'y' for start or 'n' for shutdown."
            exit 1
            ;;
    esac
done
