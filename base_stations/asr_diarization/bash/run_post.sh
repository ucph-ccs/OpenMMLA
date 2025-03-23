#!/bin/bash
# This script runs the post-time audio analyzer with specified parameters

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="audio-base"
FILENAMES=""
VAD=true
NR=true
SP=false
TR=true

is_boolean() {
    [[ $1 =~ ^(true|false)$ ]]
}

is_raspberry_pi() {
    grep -q "ID=debian" /etc/os-release && grep -q "Raspberry Pi" /proc/cpuinfo
}

is_ubuntu() {
    grep -q "ID=ubuntu" /etc/os-release
}

print_usage() {
    echo "Usage: $0 [-f FILENAMES] [-vad VAD] [-nr NR] [-sp SP] [-tr TR] [-h]"
    echo "  -f FILENAMES  : specified filenames in /post-time/origin/ to process, default to all files when not specified."
    echo "  -vad VAD     : Whether to use Voice Activity Detection (true/false, default: true)"
    echo "  -nr NR       : Whether to use Noise Reduction (true/false, default: true)"
    echo "  -sp SP       : Whether to use Speech Separation (true/false, default: true)"
    echo "  -tr TR       : Whether to transcribe audio (true/false, default: true)"
    echo "  -h           : Display this help message"
    exit 1
}

run_py_in_new_tab_mac() {
    CMD="$1"
    osascript -e "tell application \"Terminal\" to activate" \
              -e "tell application \"System Events\" to keystroke \"t\" using {command down}" \
              -e "tell application \"Terminal\" to do script \"export PYTHONPATH='$PYTHON_PATH':\$PYTHONPATH && source activate $CONDA_ENV && $CMD\" in front window"
}

run_py_in_new_win_lxterminal() {
    CMD="$1"
    lxterminal --command="bash -c \"export PYTHONPATH=$PYTHON_PATH/:\$PYTHONPATH; source ~/miniforge3/etc/profile.d/conda.sh; conda activate $CONDA_ENV; $CMD; exec bash\"" &
}

run_py_in_new_tab_gnome() {
    CMD="$1"
    gnome-terminal --tab -- bash -c "export PYTHONPATH=$PYTHON_PATH/:\$PYTHONPATH; source activate $CONDA_ENV; $CMD; exec bash"
}

# Parse arguments for multi-character options
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        -f)
            i=$((i+1))
            if [ $i -le $# ]; then
                FILENAMES="${!i}"
            else
                echo "Error: -f requires a value"
                print_usage
            fi
            ;;
        -vad)
            i=$((i+1))
            if [ $i -le $# ]; then
                VAD="${!i}"
            else
                echo "Error: -vad requires a value"
                print_usage
            fi
            ;;
        -nr)
            i=$((i+1))
            if [ $i -le $# ]; then
                NR="${!i}"
            else
                echo "Error: -nr requires a value"
                print_usage
            fi
            ;;
        -sp)
            i=$((i+1))
            if [ $i -le $# ]; then
                SP="${!i}"
            else
                echo "Error: -sp requires a value"
                print_usage
            fi
            ;;
        -tr)
            i=$((i+1))
            if [ $i -le $# ]; then
                TR="${!i}"
            else
                echo "Error: -tr requires a value"
                print_usage
            fi
            ;;
        -h)
            print_usage
            ;;
        *)
            echo "Invalid option: $arg"
            print_usage
            ;;
    esac
    i=$((i+1))
done

# Validate boolean arguments
for arg_name in "VAD" "NR" "SP" "TR"; do
    arg_value="${!arg_name}"
    if ! is_boolean "$arg_value"; then
        echo "Error: $arg_name must be either 'true' or 'false'."
        print_usage
    fi
done

# Display configuration
echo "Audio Post-Analyzer Configuration:"
echo "--------------------------------"
echo "Filename: $FILENAMES"
echo "Voice Activity Detection: $VAD"
echo "Noise Reduction: $NR"
echo "Speech Separation: $SP"
echo "Transcription: $TR"
echo "--------------------------------"
echo "Starting audio post analyzer..."

# Construct the command.
# Using single quotes around the filename avoids introducing extra double quotes.
CMD="python3 $PROJECT_DIR/examples/run_post_audio_analyzer.py -f '$FILENAMES' -v $VAD -n $NR -s $SP -t $TR"

# Run the command based on the OS
if [[ $OSTYPE == 'darwin'* ]]; then
    # macOS
    run_py_in_new_tab_mac "$CMD"
    echo "Started audio post analyzer in a new terminal tab."
elif is_raspberry_pi; then
    # Raspberry Pi
    run_py_in_new_win_lxterminal "$CMD"
    echo "Started audio post analyzer in a new terminal window."
elif is_ubuntu; then
    # Ubuntu
    run_py_in_new_tab_gnome "$CMD"
    echo "Started audio post analyzer in a new terminal tab."
else
    echo "Unknown OS or not supported. Running in current terminal:"
    export PYTHONPATH="$PYTHON_PATH":$PYTHONPATH
    source activate $CONDA_ENV
    eval $CMD
fi

echo "Audio post analyzer started successfully."
