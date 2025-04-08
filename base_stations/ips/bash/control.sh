#!/bin/bash
# This script runs the control base

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="ips-base"

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

# Run controller
CMD="python3 $PROJECT_DIR/examples/control.py"
if [[ "$OSTYPE" == "darwin"* ]]; then
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
