#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$BASH_DIR")"

# Folders to clean and recreate
paths=(
    "$PROJECT_DIR/visualizations/"
    "$PROJECT_DIR/logs/"
    "$PROJECT_DIR/logger/"
    "$PROJECT_DIR/audio_db/"
    "$PROJECT_DIR/audio_db/post-time"
    "$PROJECT_DIR/audio/"
    "$PROJECT_DIR/audio/temp/"
    "$PROJECT_DIR/audio/real-time/"
    "$PROJECT_DIR/audio/post-time/"
    "$PROJECT_DIR/audio/post-time/origin/"
    "$PROJECT_DIR/audio/post-time/segments/"
    "$PROJECT_DIR/audio/post-time/formatted/"
    "$PROJECT_DIR/audio/post-time/chunks/"
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
