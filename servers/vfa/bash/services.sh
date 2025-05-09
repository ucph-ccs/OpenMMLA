#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PYHON_PATH="$BASH_DIR/../../.."
CONFIG_FILE="$BASH_DIR/../config.yml"

CONDA_ENV="vfa-server"
CONDA_INIT="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $CONDA_ENV"

# First ensure conda environment is activated
echo "Activating conda environment '$CONDA_ENV'..."
eval "$CONDA_INIT"

# Check if Python yaml module is available in the activated environment
if python -c "import yaml" 2>/dev/null; then
    YAML_AVAILABLE=true
    echo "YAML module available, will read configuration from $CONFIG_FILE"
else
    YAML_AVAILABLE=false
    echo "WARNING: Python YAML module not available in conda environment '$CONDA_ENV'."
    echo "Cannot read service configurations from config file."
    exit 1
fi

# Read service configurations from config.yml
echo "Reading service configurations from $CONFIG_FILE..."
SERVICES_CONFIG=$(python -c "
import yaml
import json
import sys
import os

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract service configurations
    services_config = {}
    for service_name, service_config in config.items():
        if isinstance(service_config, dict) and 'port' in service_config:
            app_path = service_config.get('app', '')
            # Expand tilde in path if present
            if app_path.startswith('~'):
                app_path = os.path.expanduser(app_path)
                
            services_config[service_name] = {
                'port': service_config.get('port'),
                'workers': service_config.get('workers', 1),
                'app': app_path
            }
    
    print(json.dumps(services_config))
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    print(json.dumps({}))
    exit(1)
")

# Parse the JSON response
if [ -z "$SERVICES_CONFIG" ] || [ "$SERVICES_CONFIG" == "{}" ]; then
    echo "Failed to read service configurations from $CONFIG_FILE."
    exit 1
fi

# Extract service information with Python to avoid bash parsing issues
SERVICE_INFO=$(python -c "
import json
import sys
import os

try:
    data = json.loads('$SERVICES_CONFIG')
    
    services = []
    ports = []
    workers = []
    apps = []
    app_dirs = []
    
    for service, config in data.items():
        app_path = config['app']
        app_dir = ''
        module_name = app_path
        
        # Handle file paths vs module names
        if os.path.exists(app_path) or os.path.exists(app_path + '.py'):
            # It's a file path
            if os.path.exists(app_path):
                # Full path with .py
                app_dir = os.path.dirname(app_path)
                module_name = os.path.basename(app_path)
                if module_name.endswith('.py'):
                    module_name = module_name[:-3]
            else:
                # Path without .py extension
                app_dir = os.path.dirname(app_path)
                module_name = os.path.basename(app_path)
        
        services.append(service)
        ports.append(str(config['port']))
        workers.append(str(config['workers']))
        apps.append(module_name)
        app_dirs.append(app_dir)
    
    print(json.dumps({
        'services': services,
        'ports': ports,
        'workers': workers, 
        'apps': apps,
        'app_dirs': app_dirs
    }))
except Exception as e:
    print(f'Error parsing service config: {e}', file=sys.stderr)
    print(json.dumps({}))
    sys.exit(1)
")

# Parse the service information
services=()
ports=()
workers=()
apps=()
app_dirs=()

eval "$(python -c "
import json
import sys

try:
    data = json.loads('$SERVICE_INFO')
    
    for i, service in enumerate(data['services']):
        service_safe = service.replace(' ', '_').replace('(', '_').replace(')', '_')
        print(f\"services[{i}]='{service}'\")
        print(f\"ports[{i}]='{data['ports'][i]}'\")
        print(f\"workers[{i}]='{data['workers'][i]}'\")
        print(f\"apps[{i}]='{data['apps'][i]}'\")
        print(f\"app_dirs[{i}]='{data['app_dirs'][i]}'\")
except Exception as e:
    sys.stderr.write(f'Error parsing service info: {e}\\n')
    sys.exit(1)
")"

if [ ${#services[@]} -eq 0 ]; then
    echo "No services configured in $CONFIG_FILE. Exiting."
    exit 1
fi

echo "Found ${#services[@]} services in configuration:"
for i in "${!services[@]}"; do
    echo "${services[$i]} using port ${ports[$i]} with ${workers[$i]} workers, app: ${apps[$i]}"
    if [ -n "${app_dirs[$i]}" ]; then
        echo "  Working directory: ${app_dirs[$i]}"
    fi
done

# Loop through each port and kill processes using those ports
for i in "${!ports[@]}"; do
    port="${ports[$i]}"
    service="${services[$i]}"
    echo "Checking for processes using port $port ($service)"

    PIDs=$(sudo lsof -ti:"$port" 2>/dev/null | xargs)
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

# Create commands array based on configured ports and workers
declare -a commands=()
for i in "${!services[@]}"; do
    service="${services[$i]}"
    port="${ports[$i]}"
    worker="${workers[$i]}"
    app="${apps[$i]}"
    app_dir="${app_dirs[$i]}"
    
    if [ -z "$app" ]; then
        echo "Warning: No app specified for $service. Skipping."
        continue
    fi

    # If we have a directory, cd to it first
    if [ -n "$app_dir" ]; then
        cmd="$CONDA_INIT && cd \"$app_dir\" && gunicorn -k gevent -w $worker -b 0.0.0.0:$port $app:app"
    else
        cmd="$CONDA_INIT && gunicorn -k gevent -w $worker -b 0.0.0.0:$port $app:app"
    fi
    
    commands+=("$cmd")
done

# Loop for creating tmux session for each service
for i in "${!services[@]}"; do
    service="${services[$i]}"
    # Use the service name as session name, but make it lowercase for tmux and ensure it's safe for tmux
    session_name=$(echo "$service" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_-')
    
    if [ -z "${commands[$i]}" ]; then
        echo "No command for $service. Skipping."
        continue
    fi
    
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
tmux kill-session -t "vfa-services" 2>/dev/null