#!/bin/bash

ROOT_DIR="$(dirname "$(readlink -f "$0")")"
RECORD_DIR="$ROOT_DIR/../recordings"

# Check if the RECORD_DIR directory exists
if [[ ! -d $RECORD_DIR ]]; then
    echo "The specified project root does not exist: $RECORD_DIR"
    exit 1
fi

echo "Available directories:"
cd "$RECORD_DIR" || return

# List directories and capture selection
select dir in */; do
    if [ -n "$dir" ]; then
        # Remove trailing slash
        dir=${dir%/}
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

echo "You have selected: $dir"

# Construct the input path for ffmpeg
input_path="$RECORD_DIR/$dir/frame_%07d.jpg"

# Output file location
output_path="$RECORD_DIR/$dir.mp4"

# Use ffmpeg to create the video file
ffmpeg -framerate 30 -i "$input_path" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "$output_path"

echo "Video created at: $output_path"
