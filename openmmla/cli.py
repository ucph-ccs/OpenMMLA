# openmmla/cli.py
import sys
from openmmla.cli_utils import run_cli


def main():
    # Support both 'openmmla' and 'mmla' commands
    if sys.argv[0].endswith('mmla'):
        # If called as 'mmla', remove the first argument to maintain consistent argv
        sys.argv = [sys.argv[0].replace('mmla', 'openmmla')] + sys.argv[1:]
    run_cli()
