#!/bin/bash
# This script runs audio bases and synchronizer with specified parameters

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="audio-base"
NUM_BASES=3
NUM_SYNCHRONIZER=1
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

print_usage() {
    echo "Usage: $0 [-nb NUM_BASES] [-ns NUM_SYNCHRONIZER] [-sp SP] [-h]"
    echo "  -nb NUM_BASES        : Number of audio bases to run (default: 3)"
    echo "  -ns NUM_SYNCHRONIZER : Number of synchronizers to run (default: 1)"
    echo "  -sp  SP              : Whether to use Speech Separation (true/false, default: false)"
    echo "  -h                   : Display this help message"
    exit 1
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
    gnome-terminal --tab -- bash -c "export PYTHONPATH=$PYTHON_PATH/:$PYTHON_PATH; source activate $CONDA_ENV; $CMD; exec bash"
}

# Parse arguments for multi-character options
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        -nb)
            i=$((i+1))
            if [ $i -le $# ]; then
                NUM_BASES="${!i}"
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
        -sp)
            i=$((i+1))
            if [ $i -le $# ]; then
                SP="${!i}"
            else
                echo "Error: -sp requires a value"
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
if ! is_number "$NUM_BASES"; then
    echo "Error: -nb NUM_BASES must be a number."
    print_usage
fi

if ! is_number "$NUM_SYNCHRONIZER"; then
    echo "Error: -ns NUM_SYNCHRONIZER must be a number."
    print_usage
fi

if ! is_boolean "$SP"; then
    echo "Error: -sp SP must be either 'true' or 'false'."
    print_usage
fi

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
            BT='Badge'
            BASE_SCRIPT="run_badge_audio_base.py"
            echo "Selected: Badge"
            break
            ;;
        2)
            BT='Jabra'
            BASE_SCRIPT="run_jabra_audio_base.py"
            echo "Selected: Jabra"
            break
            ;;
        *)
            echo "Invalid selection. Please choose either 1 or 2."
            ;;
    esac
done

echo "Starting $NUM_BASES $BT audio base(s) and $NUM_SYNCHRONIZER synchronizer(s)..."
echo "Speech Separation: $SP"

# Run audio bases scripts
if [ "$NUM_BASES" -gt 0 ]; then
  echo "Starting audio bases..."
  for i in $(seq 1 "$NUM_BASES"); do
    if [[ $OSTYPE == 'darwin'* ]]; then
      run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -s $SP -v true -n true -t true -st true"
      echo "Started $BT audio base $i in a new terminal tab."
    elif is_raspberry_pi; then
      run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -s $SP -v true -n true -t true -st false"
      echo "Started $BT audio base $i in a new terminal window."
    elif is_ubuntu; then
      run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/$BASE_SCRIPT -s $SP -v true -n true -t true -st true"
      echo "Started $BT audio base $i in a new terminal tab."
    else
      echo "Unknown OS or not supported."
    fi
    # Small delay to prevent race conditions
    sleep 1
  done
fi

# Run synchronizer script
if [ "$NUM_SYNCHRONIZER" -gt 0 ]; then
  echo "Starting synchronizer..."
  if [[ $OSTYPE == 'darwin'* ]]; then
    run_py_in_new_tab_mac "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
    echo "Started synchronizer in a new terminal tab."
  elif is_raspberry_pi; then
    run_py_in_new_win_lxterminal "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
    echo "Started synchronizer in a new terminal window."
  elif is_ubuntu; then
    run_py_in_new_tab_gnome "python3 $PROJECT_DIR/examples/run_synchronizer.py -b $BT -s $SP -d false"
    echo "Started synchronizer in a new terminal tab."
  else
    echo "Unknown OS or not supported."
  fi
fi

echo "All components started successfully."