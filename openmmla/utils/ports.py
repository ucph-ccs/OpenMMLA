import os
import signal
import subprocess


def find_process_using_port(port):
    """Find the PID of the process using the specified port."""
    command = f"lsof -i tcp:{port} | awk 'NR!=1 {{print $2}}'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pids = result.stdout.strip().split('\n')
    unique_pids = list(set(pid for pid in pids if pid.isdigit()))
    return unique_pids


def kill_process(pid):
    """Kill the process with the given PID."""
    try:
        os.kill(int(pid), signal.SIGKILL)
        print(f"Process {pid} killed successfully.")
    except OSError as error:
        print(f"Error: {error}")


def free_port(port):
    pids = find_process_using_port(port)
    if pids:
        print(f"Processes using port {port}: {pids}")
        for pid in pids:
            kill_process(pid)
    else:
        print(f"No process is using port {port}.")
