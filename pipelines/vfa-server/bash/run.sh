#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_FILE="$BASH_DIR/../config.yml"

# Conda environment setup
CONDA_ENV="${OPENMMLA_CONDA_ENV:-vfa-server}"
CONDA_INIT='if command -v conda >/dev/null 2>&1; then _base="$(conda info --base 2>/dev/null)" && [ -f "$_base/etc/profile.d/conda.sh" ] && source "$_base/etc/profile.d/conda.sh"; fi; if ! command -v conda >/dev/null 2>&1; then for _p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /home/*/miniforge3 /home/*/miniconda3 /opt/conda; do [ -f "$_p/etc/profile.d/conda.sh" ] && source "$_p/etc/profile.d/conda.sh" && break; done; fi; conda activate "$CONDA_ENV"'

# Function to get service names from config file
get_session_names() {
    # Activate conda environment to ensure we have yaml
    eval "$CONDA_INIT" > /dev/null 2>&1
    
    # Try to parse config file
    python -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract service names and convert to lowercase for tmux sessions
    for service_name, service_config in config.items():
        if isinstance(service_config, dict) and 'port' in service_config:
            # Convert service name to lowercase for tmux session name
            print(service_name.lower())
except Exception as e:
    sys.stderr.write(f'Error reading config: {e}\\n')
    sys.exit(1)
" 2>/dev/null
}

shutdown_server() {
    echo "Shutting down VFA services..."
    active_sessions=$(tmux list-sessions -F "#S" 2>/dev/null || echo "")
    
    if [ -z "$active_sessions" ]; then
        echo "No services are currently running."
        return
    fi
    
    # Get list of session names for our services
    session_names=$(get_session_names)
    
    # Filter and shut down the on-going service sessions
    for session in $session_names; do
        if [[ $active_sessions =~ $session ]]; then  # Check if the service is an active session
            echo "Sending Ctrl+C to session: $session"
            tmux send-keys -t "$session" C-c
            tmux kill-session -t "$session"
        fi
    done
    
    # Also try to kill vfa-services session if it exists
    if [[ $active_sessions =~ "vfa-services" ]]; then
        tmux kill-session -t "vfa-services" 2>/dev/null
    fi
    
    echo "VFA server shutdown process complete."
}

start_server() {
    # Check if vfa-services session exists and kill it
    if tmux has-session -t vfa-services 2>/dev/null; then
        echo "Found existing vfa-services session. Killing it..."
        tmux kill-session -t vfa-services
    fi
    
    echo "Starting VFA services..."
    tmux new-session -s vfa-services "bash -c '$BASH_DIR/services.sh; exec bash'"
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
            echo "Server started. Use 'tmux ls' to list all sessions, and 'tmux attach -t <service_name>' to view a specific service."
            break
            ;;
        [Nn])
            echo "Shutting down the server..."
            shutdown_server
            echo "Server shutdown complete. Use 'tmux ls' to list all sessions."
            break
            ;;
        *)
            echo "Invalid option. Please enter 'y' for start or 'n' for shutdown."
            exit 1
            ;;
    esac
done
