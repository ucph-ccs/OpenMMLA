#!/bin/bash
# This script runs the real-time audio analyzer

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="asr-base"
NUM_BASE=3
NUM_SYNCHRONIZER=1
STORE=true
VOICE_ACTIVITY_DETECT=true
NOISE_REDUCE=true
TRANSCRIBE=true
SPEECH_SEPARATE=false
DOMINANT=false

print_usage() {
    echo "usage: $0 [-nb NUM_BASE] [-ns NUM_SYNCHRONIZER] [-s STORE] [-vad VOICE_ACTIVITY_DETECT] [-nr NOISE_REDUCE] [-tr TRANSCRIBE] [-sp SPEECH_SEPARATE] [-d DOMINANT] [-h]"
    echo ""
    echo "options:"
    echo "  -nb  NUM_BASE               : Number of audio bases to run (default: 3)"
    echo "  -ns  NUM_SYNCHRONIZER       : Number of synchronizers to run (default: 1)"
    echo "  -s   STORE                  : Whether to store audio data (true/false, default: true)"
    echo "  -vad VOICE_ACTIVITY_DETECT  : Whether to use Voice Activity Detection (true/false, default: true)"
    echo "  -nr  NOISE_REDUCE           : Whether to use Noise Reduction (true/false, default: true)"
    echo "  -tr  TRANSCRIBE             : Whether to transcribe audio (true/false, default: true)"
    echo "  -sp  SPEECH_SEPARATE        : Whether to use Speech Separation (true/false, default: false)"
    echo "  -d   DOMINANT               : Whether to apply dominant speaker (true/false, default: false)"
    echo "  -h                          : Display this help message"
    exit 1
}

# Helper functions
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

is_boolean() {
    [[ $1 =~ ^(true|false)$ ]]
}

is_raspberry_pi() {
    grep -q "ID=debian" /etc/os-release && grep -q "Raspberry Pi" /proc/cpuinfo
}

is_ubuntu() {
    grep -q "ID=ubuntu" /etc/os-release
}

run_py_in_new_tab_mac() {
    CMD=$1
    osascript -e "tell app \"Terminal\" to activate" \
              -e "tell app \"System Events\" to keystroke \"t\" using command down" \
              -e "tell app \"Terminal\" to do script \"export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH && source activate $CONDA_ENV && $CMD\" in the front window"
}

run_py_in_new_win_lxterminal() {
    CMD=$1
    lxterminal --command="bash -c \"export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH; source ~/miniforge3/etc/profile.d/conda.sh; conda activate $CONDA_ENV; $CMD; exec bash\"" &
}

run_py_in_new_tab_gnome() {
    CMD=$1
    gnome-terminal --tab -- bash -c "export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH; source activate $CONDA_ENV; $CMD; exec bash"
}

# Parse arguments for multi-character options
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        -nb)
            i=$((i+1))
            if [ $i -le $# ]; then
                NUM_BASE="${!i}"
            else
                echo "Error: -nb requires a value"
                print_usage
            fi
            ;;
        -ns)
            i=$((i+1))
            if [ $i -le $# ]; then
                NUM_SYNCHRONIZER="${!i}"
            else
                echo "Error: -ns requires a value"
                print_usage
            fi
            ;;
        -s)
            i=$((i+1))
            if [ $i -le $# ]; then
                STORE="${!i}"
            else
                echo "Error: -s requires a value"
                print_usage
            fi
            ;;
        -vad)
            i=$((i+1))
            if [ $i -le $# ]; then
                VOICE_ACTIVITY_DETECT="${!i}"
            else
                echo "Error: -vad requires a value"
                print_usage
            fi
            ;;
        -nr)
            i=$((i+1))
            if [ $i -le $# ]; then
                NOISE_REDUCE="${!i}"
            else
                echo "Error: -nr requires a value"
                print_usage
            fi
            ;;
        -tr)
            i=$((i+1))
            if [ $i -le $# ]; then
                TRANSCRIBE="${!i}"
            else
                echo "Error: -tr requires a value"
                print_usage
            fi
            ;;
        -sp)
            i=$((i+1))
            if [ $i -le $# ]; then
                SPEECH_SEPARATE="${!i}"
            else
                echo "Error: -sp requires a value"
                print_usage
            fi
            ;;
        -d)
            i=$((i+1))
            if [ $i -le $# ]; then
                DOMINANT="${!i}"
            else
                echo "Error: -d requires a value"
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

# Validate arguments
for arg_name in "NUM_BASE" "NUM_SYNCHRONIZER"; do
    arg_value="${!arg_name}"
    if ! is_number "$arg_value"; then
        echo "Error: $arg_name must be a number"
        print_usage
    fi
done

for arg_name in "STORE" "VOICE_ACTIVITY_DETECT" "NOISE_REDUCE" "TRANSCRIBE" "SPEECH_SEPARATE" "DOMINANT"; do
    arg_value="${!arg_name}"
    if ! is_boolean "$arg_value"; then
        echo "Error: $arg_name must be either 'true' or 'false'."
        print_usage
    fi
done

# Display configuration
echo "Audio Real-Analyzer Configuration:"
echo "--------------------------------"
echo "NUM_BASE: $NUM_BASE"
echo "NUM_SYNCHRONIZER: $NUM_SYNCHRONIZER"
echo "STORE: $STORE"
echo "VOICE_ACTIVITY_DETECT: $VOICE_ACTIVITY_DETECT"
echo "NOISE_REDUCE: $NOISE_REDUCE"
echo "TRANSCRIBE: $TRANSCRIBE"
echo "SPEECH_SEPARATE: $SPEECH_SEPARATE"
echo "DOMINANT: $DOMINANT"
echo "--------------------------------"

# Prompt user to choose base type
while true; do
    echo "1: Badge"
    echo "2: Jabra"
    echo "Please select the base type:"
    stty -echo -icanon time 0 min 0
    read -r -n 1 BASE_TYPE_SELECTION
    stty echo icanon

    case $BASE_TYPE_SELECTION in
        1)
            BASE_TYPE='Badge'
            echo "Selected: Badge"
            break
            ;;
        2)
            BASE_TYPE='Jabra'
            echo "Selected: Jabra"
            break
            ;;
        *)
            echo "Invalid selection. Please choose either 1 or 2."
            ;;
    esac
done

echo "Starting $NUM_BASE $BASE_TYPE audio base(s) and $NUM_SYNCHRONIZER synchronizer(s)..."

# Run audio bases
CMD="python3 $PROJECT_DIR/examples/run_audio_base.py -b $BASE_TYPE -s $STORE -vad $VOICE_ACTIVITY_DETECT -nr $NOISE_REDUCE -tr $TRANSCRIBE -sp $SPEECH_SEPARATE"
if [ "$NUM_BASE" -gt 0 ]; then
    echo "Starting audio bases..."
    for i in $(seq 1 "$NUM_BASE"); do
        if [[ $OSTYPE == 'darwin'* ]]; then
            run_py_in_new_tab_mac "$CMD"
        elif is_raspberry_pi; then
            run_py_in_new_win_lxterminal "$CMD"
        elif is_ubuntu; then
            run_py_in_new_tab_gnome "$CMD"
        else
            echo "Unknown OS or not supported. Running in current terminal:"
            export PYTHONPATH="$PYTHON_PATH":$PYTHONPATH
            source activate $CONDA_ENV
            eval "$CMD"
        fi
    done
fi

# Run synchronizer
CMD="python3 $PROJECT_DIR/examples/run_audio_synchronizer.py -b $BASE_TYPE -d $DOMINANT -sp $SPEECH_SEPARATE"
if [ "$NUM_SYNCHRONIZER" -gt 0 ]; then
    echo "Starting synchronizer..."
    if [[ $OSTYPE == 'darwin'* ]]; then
        run_py_in_new_tab_mac "$CMD"
    elif is_raspberry_pi; then
        run_py_in_new_win_lxterminal "$CMD"
    elif is_ubuntu; then
        run_py_in_new_tab_gnome "$CMD"
    else
        echo "Unknown OS or not supported. Running in current terminal:"
        export PYTHONPATH="$PYTHON_PATH":$PYTHONPATH
        source activate $CONDA_ENV
        eval "$CMD"
    fi
fi

echo "All components started successfully."