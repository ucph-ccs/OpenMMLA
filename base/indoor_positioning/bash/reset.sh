#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"

# Folders to clean and recreate
paths=(
    "$BASH_DIR/visualizations"
    "$BASH_DIR/recordings"
    "$BASH_DIR/logger"
    "$BASH_DIR/logs"
)

# Clean the folders
for path in "${paths[@]}"; do
    if [ -d "$path" ]; then
        # Remove all the files and subdirectories
        rm -r "$path"/*
        echo "Cleaned folder: $path"
    else
        echo "Folder $path does not exist"
    fi
done

# Create or recreate folders
for path in "${paths[@]}"; do
    if [ ! -d "$path" ]; then
        echo "Create path: $path"
        mkdir -p "$path"
    fi
done
