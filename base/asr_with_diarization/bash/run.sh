#!/bin/bash

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="audio-base"
NUM_BASES=3
NUM_SYNCHRONIZER=1
LOCAL=false
SP=false

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
while getopts "b:s:l:p:" opt; do
  case $opt in
    b) NUM_BASES=$OPTARG ;;
    s) NUM_SYNCHRONIZER=$OPTARG ;;
    l) LOCAL=$OPTARG ;;
    p) SP=$OPTARG ;;
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
    echo "Error: -b NUM_SYNCHRONIZER must be a number."
    exit 1
fi

if ! is_boolean "$LOCAL"; then
    echo "Error: -l LOCAL must be a boolean (true or false)."
    exit 1
fi

if [ "$LOCAL" == "true" ]; then
    CONDA_ENV="audio-server"
fi

# Prompt user to choose base type
while true; do
    echo "0: Badge"
    echo "1: Jabra"
    echo "Please select the base type:"
    stty -echo -icanon time 0 min 0
    read -r -n 1 BASE_TYPE_SELECTION
    stty echo icanon

    case $BASE_TYPE_SELECTION in
        0)
            BT='Badge'
            BASE_SCRIPT="run_badge_audio_base.py"
            break
            ;;
        1)
            BT='Jabra'
            BASE_SCRIPT="run_jabra_audio_base.py"
            break
            ;;
        *)
            echo "Invalid selection. Please choose either 0 or 1."
            ;;
    esac
done

# Run audio bases scripts
if [ "$NUM_BASES" -gt 0 ]; then
  for i in $(seq 1 "$NUM_BASES"); do
    if [[ $OSTYPE == 'darwin'* ]]; then
      run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -l $LOCAL -s $SP -v true -n true -t true -st true"
    elif is_raspberry_pi; then
      run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -l $LOCAL -s $SP -v true -n true -t true -st false"
    elif is_ubuntu; then
      run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -l $LOCAL -s $SP -v true -n true -t true -st true"
    else
      echo "Unknown OS or not supported."
    fi
  done
fi

# Run synchronizer script
if [ "$NUM_SYNCHRONIZER" -gt 0 ]; then
  if [[ $OSTYPE == 'darwin'* ]]; then
    run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
  elif is_raspberry_pi; then
    run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
  elif is_ubuntu; then
    run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
  else
    echo "Unknown OS or not supported."
  fi
fi