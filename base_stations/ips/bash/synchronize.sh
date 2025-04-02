#!/bin/bash
# This script runs the multi-camera synchronization system

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="video-base"
NUM_CAMERA=2
NUM_SYNCMANAGER=1

print_usage() {
    echo "usage: $0 [-nc] $NUM_CAMERA [-ns] $NUM_SYNCHRONIZER [-h]"
    echo ""
    echo "options:"
    echo "  -nc  NUM_CAMERA             : Number of camera detectors to run (default: 3)"
    echo "  -ns  NUM_SYNCHRONIZER       : Number of synchronizers to run (default: 1)"
    echo "  -h                          : Display this help message"
    exit 1
}

# Helper functions
is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
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

# Parse arguments
i=1
while [ $i -le $# ];do
    arg="${!i}"
    case "$arg" in
        -nc)
            i=$((i+1))
            if [ $i -le $# ]; then
                NUM_CAMERA="${!i}"
            else
                echo "Error: -nc requires a value"
                print_usage
            fi
            ;;
        -ns)
            i=$((i+1))
            if [ $i -le $# ]; then
                NUM_SYNCMANAGER="${!i}"
            else
                echo "Error: -ns requires a value"
                print_usage
            fi
            ;;
        -h)
            print_usage
            ;;
    esac
    i=$((i+1))
done

# Validate arguments
for arg_name in "NUM_CAMERA" "NUM_SYNCHRONIZER"; do
    arg_value="${!arg_name}"
    if ! is_number "$arg_value"; then
        echo "Error: $arg_name must be a number"
        print_usage
    fi
done

# Display configuration
echo "Multi-camera Synchronization Configuration:"
echo "--------------------------------"
echo "NUM_CAMERA: $NUM_CAMERA"
echo "NUM_SYNCHRONIZER: $NUM_SYNCHRONIZER"
echo "--------------------------------"

# Run camera tag detectors
CMD="python3 $PROJECT_DIR/examples/run_camera_tag_detector.py"
if [[ $NUM_CAMERA -gt 0 ]]; then
    for i in $(seq 1 $((NUM_CAMERA))); do
        if [[ "$OSTYPE" == "darwin"* ]]; then
            run_py_in_new_tab_mac "$CMD"
        elif is_raspberry_pi; then
            run_py_in_new_win_lxterminal "$CMD"
        elif is_ubuntu; then
            run_py_in_new_tab_gnome "$CMD"
        else
            echo "Unsupported OS."
            exit 1
        fi
    done
fi

# Run camera sync manager
CMD="python3 $PROJECT_DIR/examples/run_camera_sync_manager.py"
if [[ $NUM_SYNCMANAGER -gt 0 ]]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        run_py_in_new_tab_mac "$CMD"
    elif is_raspberry_pi; then
        run_py_in_new_win_lxterminal "$CMD"
    elif is_ubuntu; then
        run_py_in_new_tab_gnome "$CMD"
    else
        echo "Unsupported OS."
        exit 1
    fi
fi

echo "All components started successfully."