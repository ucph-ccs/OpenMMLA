#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$BASH_DIR")"
PYTHON_PATH="$(dirname "$(dirname "$(dirname "$BASH_DIR")")")"

CONDA_ENV="video-base"
NUM_DETECTORS=2
NUM_SYNCMANAGER=1

is_number() {
    [[ $1 =~ ^[0-9]+$ ]]
}

is_raspberry_pi() {
    grep -q "ID=debian" /etc/os-release && grep -q "Raspberry Pi" /proc/cpuinfo
}

is_ubuntu() {
    grep -q "ID=ubuntu" /etc/os-release
}

if ! is_number "$NUM_DETECTORS"; then
    echo "Error: Please provide a valid number for camera detectors."
    exit 1
fi

run_py_in_new_win_mac() {
    CMD=$1
    osascript -e "tell app \"Terminal\" to do script \"export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH && source activate $CONDA_ENV && $CMD\""
}

run_py_in_new_win_lxterminal() {
    CMD=$1
    lxterminal --command="bash -c \"export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH; source ~/miniforge3/etc/profile.d/conda.sh; conda activate $CONDA_ENV; $CMD; exec bash\"" &
}

run_py_in_new_win_gnome() {
    CMD=$1
    gnome-terminal -- bash -c "export PYTHONPATH=$PYTHON_PATH/:$PYTHONPATH; source activate $CONDA_ENV; $CMD; exec bash"
}

if ! is_number "$NUM_SYNCMANAGER"; then
    echo "Error: Please provide a valid number for camera synchronizers."
    exit 1
fi

# Parse arguments
while getopts "d:s:" opt; do
  case $opt in
    d) NUM_DETECTORS=$OPTARG ;;
    s) NUM_SYNCMANAGER=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1 ;;
  esac
done

# Validate arguments
if ! is_number "$NUM_DETECTORS"; then
    echo "Error: NUM_DETECTORS must be a number."
    exit 1
fi

if ! is_number "$NUM_SYNCMANAGER"; then
    echo "Error: NUM_SYNCMANAGER must be a number."
    exit 1
fi

# Run camera tag detectors
if [[ $NUM_DETECTORS -gt 0 ]]; then
  for i in $(seq 1 $((NUM_DETECTORS))); do
    if [[ "$OSTYPE" == "darwin"* ]]; then
      run_py_in_new_win_mac "python3 $PROJECT_DIR/examples/run_camera_tag_detector.py"
    elif is_raspberry_pi; then
      run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_camera_tag_detector.py"
    elif is_ubuntu; then
      run_py_in_new_win_gnome "python3 $PROJECT_DIR/examples/run_camera_tag_detector.py"
    else
      echo "Unsupported OS."
      exit 1
    fi
  done
fi

# Run camera sync manager
if [[ $NUM_SYNCMANAGER -gt 0 ]]; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    run_py_in_new_win_mac "python3 $PROJECT_DIR/examples/run_camera_sync_manager.py"
  elif is_raspberry_pi; then
    run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_camera_sync_manager.py"
  elif is_ubuntu; then
    run_py_in_new_win_gnome "python3 $PROJECT_DIR/examples/run_camera_sync_manager.py"
  else
    echo "Unsupported OS."
    exit 1
  fi
fi