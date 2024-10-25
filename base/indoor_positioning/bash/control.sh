#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$(dirname "$BASH_DIR")"
PYTHON_PATH="$(dirname "$(dirname "$(dirname "$BASH_DIR")")")"

CONDA_ENV="video-base"

is_raspberry_pi() {
    grep -q "ID=debian" /etc/os-release && grep -q "Raspberry Pi" /proc/cpuinfo
}

is_ubuntu() {
    grep -q "ID=ubuntu" /etc/os-release
}

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

# Run control py script
if [[ "$OSTYPE" == "darwin"* ]]; then
  run_py_in_new_win_mac "python3  $PROJECT_DIR/examples/control.py"
elif is_raspberry_pi; then
  run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/control.py"
elif is_ubuntu; then
  run_py_in_new_win_gnome "python3 $PROJECT_DIR/examples/control.py"
else
  echo "Unsupported OS."
  exit 1
fi
