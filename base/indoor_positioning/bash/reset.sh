#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."

# Folders to clean and recreate
paths=(
    "$PROJECT_DIR/visualizations"
    "$PROJECT_DIR/frames"
    "$PROJECT_DIR/logger"
    "$PROJECT_DIR/logs"
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
