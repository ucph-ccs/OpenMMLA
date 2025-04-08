#!/bin/bash
# This script runs the post-time audio analyzer

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="asr-base"
FILENAMES=""
VOICE_ACTIVITY_DETECT=true
NOISE_REDUCE=true
SPEECH_SEPARATE=false
TRANSCRIBE=true

print_usage() {
    echo "usage: $0 [-f FILENAMES] [-vad VOICE_ACTIVITY_DETECT] [-nr NOISE_REDUCE] [-sp SPEECH_SEPARATE] [-tr TRANSCRIBE] [-h]"
    echo ""
    echo "options:"
    echo "  -f FILENAMES                 : specified filenames in /post-time/origin/ to process, default to all files when not specified."
    echo "  -vad VOICE_ACTIVITY_DETECT   : Whether to use Voice Activity Detection (true/false, default: true)"
    echo "  -nr NOISE_REDUCE             : Whether to use Noise Reduction (true/false, default: true)"
    echo "  -sp SPEECH_SEPARATE          : Whether to use Speech Separation (true/false, default: true)"
    echo "  -tr TRANSCRIBE               : Whether to transcribe audio (true/false, default: true)"
    echo "  -h                           : Display this help message"
    exit 1
}

# Helper functions
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
        -sp)
            i=$((i+1))
            if [ $i -le $# ]; then
                SPEECH_SEPARATE="${!i}"
            else
                echo "Error: -sp requires a value"
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
for arg_name in "VOICE_ACTIVITY_DETECT" "NOISE_REDUCE" "SPEECH_SEPARATE" "TRANSCRIBE"; do
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
echo "Voice Activity Detection: $VOICE_ACTIVITY_DETECT"
echo "Noise Reduction: $NOISE_REDUCE"
echo "Speech Separation: $SPEECH_SEPARATE"
echo "Transcription: $TRANSCRIBE"
echo "--------------------------------"
echo "Starting audio post analyzer..."

# Run post audio analyzer
CMD="python3 $PROJECT_DIR/examples/run_post_audio_analyzer.py -f '$FILENAMES' -vad $VOICE_ACTIVITY_DETECT -nr $NOISE_REDUCE -sp $SPEECH_SEPARATE -tr $TRANSCRIBE"
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

echo "Audio post analyzer started successfully."
