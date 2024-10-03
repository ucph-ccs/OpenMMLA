import os
import select
import shutil
import sys


def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def safe_delete_file(file_path):
    """Safely deletes a file, ensuring it exists first."""
    if os.path.exists(file_path):
        os.remove(file_path)


def flush_input():
    """Flush all input from stdin buffer."""
    while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)
