#!/bin/bash
# This script runs the indoor positioning system

BASH_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_DIR="$BASH_DIR/.."
PYTHON_PATH="$BASH_DIR/../../.."

CONDA_ENV="ips-base"
NUM_BASE=1
NUM_SYNCHRONIZER=1
NUM_VISUALIZER=1
GRAPHICS=true
STORE=false
VERBOSE=false

print_usage() {
    echo "Usage: $0 [-nb NUM_BASE] [-ns NUM_SYNCHRONIZER] [-g GRAPHICS] [-s STORE] [-h]"
    echo ""
    echo "options:"
    echo "  -nb NUM_BASE         : Number of IPS bases to run (default: 3)"
    echo "  -ns NUM_SYNCHRONIZER : Number of synchronizers to run (default: 1)"
    echo "  -g GRAPHICS          : Enable graphics (default: true)"
    echo "  -s STORE             : Enable store (default: false)"
    echo "  -v VERBOSE           : Enable verbose mode (default: false)"
    echo "  -h                   : Display this help message"
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
        -g)
            i=$((i+1))
            if [ $i -le $# ]; then
                GRAPHICS="${!i}"
            else
                echo "Error: -g requires a value"
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
        -v)
            i=$((i+1))
            if [ $i -le $# ]; then
                VERBOSE="${!i}"
            else
                echo "Error: -v requires a value"
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


for arg_name in "GRAPHICS" "STORE" "VERBOSE"; do
    arg_value="${!arg_name}"
    if ! is_boolean "$arg_value"; then
        echo "Error: $arg_name must be either 'true' or 'false'."
        print_usage
    fi
done


# Display configuration
echo "Indoor Positioning System Configuration:"
echo "--------------------------------"
echo "NUM_BASE: $NUM_BASE"
echo "NUM_SYNCHRONIZER: $NUM_SYNCHRONIZER"
echo "GRAPHICS: $GRAPHICS"
echo "STORE: $STORE"
echo "--------------------------------"
echo "Starting $NUM_BASE video base(s) and $NUM_SYNCHRONIZER synchronizer(s)..."

# Run video bases
CMD="python3 $PROJECT_DIR/examples/run_video_base.py -g $GRAPHICS -s $STORE -v $VERBOSE"
if [ "$NUM_BASE" -gt 0 ]; then
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
CMD="python3 $PROJECT_DIR/examples/run_video_synchronizer.py -v $VERBOSE"
if [ "$NUM_SYNCHRONIZER" -gt 0 ]; then
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

# Run visualizer
CMD="python3 $PROJECT_DIR/examples/run_video_visualizer.py -s $STORE"
if [ "$NUM_VISUALIZER" -gt 0 ]; then
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