import os
import signal
import subprocess


def find_process_using_port(port):
    """Find the PID of any process using the specified port (TCP, UDP, or others)."""
    # This command lists all processes using the given port, regardless of protocol.
    command = f"lsof -i :{port} | awk 'NR>1 {{print $2}}'"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pids = result.stdout.strip().split()
    unique_pids = list(set(pid for pid in pids if pid.isdigit()))
    return unique_pids


def kill_process(pid):
    """Kill the process with the given PID if it's not the current process."""
    current_pid = os.getpid()
    if int(pid) == current_pid:
        print(f"Skipping killing current process (PID: {pid}).")
        return

    try:
        os.kill(int(pid), signal.SIGKILL)
        print(f"Process {pid} killed successfully.")
    except OSError as error:
        print(f"Error killing process {pid}: {error}")


def free_port(port):
    """Free the port by killing any process using it (excluding the current program)."""
    pids = find_process_using_port(port)
    if pids:
        print(f"Processes using port {port}: {pids}")
        for pid in pids:
            kill_process(pid)
    else:
        print(f"No process is using port {port}.")