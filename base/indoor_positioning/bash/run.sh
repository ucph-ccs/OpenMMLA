#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$BASH_DIR")"
PYTHON_PATH="$(dirname "$(dirname "$(dirname "$BASH_DIR")")")"

CONDA_ENV="video-base"
NUM_BASES=1
NUM_SYNCHRONIZER=1
NUM_VISUALIZER=1
GRAPHICS=true
RECORD=false
STORE=false

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

# Parse arguments
while getopts "b:s:v:g:r:t:" opt; do
  case $opt in
    b) NUM_BASES=$OPTARG ;;
    s) NUM_SYNCHRONIZER=$OPTARG ;;
    v) NUM_VISUALIZER=$OPTARG ;;
    g) GRAPHICS=$OPTARG ;;
    r) RECORD=$OPTARG ;;
    t) STORE=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1 ;;
  esac
done

# Validate arguments
if ! is_number "$NUM_BASES"; then
    echo "Error: -b NUM_BASES must be a number."
    exit 1
fi

if ! is_number "$NUM_SYNCHRONIZER"; then
    echo "Error: -s NUM_SYNCHRONIZER must be a number."
    exit 1
fi

if ! is_number "$NUM_VISUALIZER"; then
    echo "Error: -v NUM_VISUALIZER must be a number."
    exit 1
fi

if ! is_boolean "$GRAPHICS"; then
    echo "Error: -g GRAPHICS must be a boolean (true or false)."
    exit 1
fi

if ! is_boolean "$RECORD"; then
    echo "Error: -r RECORD must be a boolean (true or false)."
    exit 1
fi

if ! is_boolean "$STORE"; then
    echo "Error: -s STORE must be a boolean (true or false)."
    exit 1
fi

# Open new Terminal windows and run video bases
if [ "$NUM_BASES" -gt 0 ]; then
    for i in $(seq 1 "$NUM_BASES"); do
        if [[ $OSTYPE == 'darwin'* ]]; then
            run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/run_video_base.py -g $GRAPHICS -r $RECORD"
        elif is_raspberry_pi; then
            run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_video_base.py -g $GRAPHICS -r $RECORD"
        elif is_ubuntu; then
            run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/run_video_base.py -g $GRAPHICS -r $RECORD"
        else
            echo "Unknown OS or not supported."
        fi
    done
fi

# Open new Terminal windows and run video base synchronizer
if [ "$NUM_SYNCHRONIZER" -gt 0 ]; then
  if [[ $OSTYPE == 'darwin'* ]]; then
      run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/run_synchronizer.py"
  elif is_raspberry_pi; then
      run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_synchronizer.py"
  elif is_ubuntu; then
      run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/run_synchronizer.py"
  else
      echo "Unknown OS or not supported."
  fi
fi

# Open new Terminal windows and run visualizer
if [ "$NUM_VISUALIZER" -gt 0 ]; then
  if [[ $OSTYPE == 'darwin'* ]]; then
      run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/run_visualizer.py -s $STORE"
  elif is_raspberry_pi; then
      run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_visualizer.py -s $STORE"
  elif is_ubuntu; then
      run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/run_visualizer.py -s $STORE"
  else
      echo "Unknown OS or not supported."
  fi
fi
