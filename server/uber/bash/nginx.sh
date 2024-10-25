#!/bin/bash

# Get the root directory of the script
ROOT_DIR="$(dirname "$(readlink -f "$0")")"

# Define server names
SERVER_NAMES=("server-01" "server-02" "server-03")

# Determine the platform and set the NGINX configuration path
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NGINX_CONF="/opt/homebrew/etc/nginx/nginx.conf"
else
    # Assume Linux
    NGINX_CONF="/etc/nginx/nginx.conf"
fi

echo "Using NGINX configuration file: $NGINX_CONF"

# Process the NGINX configuration
TEMP_CONF="$ROOT_DIR/../conf/nginx_temp.conf"
python3 "$ROOT_DIR/process_nginx_conf.py" "$ROOT_DIR/../conf/nginx.conf" "$TEMP_CONF" "${SERVER_NAMES[@]}"

# Copy the processed config to the actual config file
sudo cp "$TEMP_CONF" "$NGINX_CONF"

# Remove the temporary file
rm "$TEMP_CONF"

# Start NGINX with the updated configuration
sudo nginx -c $NGINX_CONF

# Test the NGINX configuration
sudo nginx -t

# Reload NGINX to apply the changes
sudo nginx -s reload

# Close uber-services
tmux kill-session -t "uber-services"

# Detach the current session
tmux detach