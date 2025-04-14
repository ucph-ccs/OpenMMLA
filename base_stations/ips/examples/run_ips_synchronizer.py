"""This script runs the IPSSynchronizer of the real-time indoor positioning system."""
import argparse
import functools
import os

from openmmla.bases.ips import IPSSynchronizer
from openmmla.utils.args import add_arguments, print_arguments

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')


def run_ips_synchronizer(args):
    ips_synchronizer = IPSSynchronizer(project_dir=args.project_dir, config_path=args.config_path,
                                       verbose=args.verbose)
    ips_synchronizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, project_dir, 'path to the project directory', shortname='-p')
    add_arg('config_path', str, config_path, 'path to the configuration file', shortname='-c')
    add_arg('verbose', bool, False, 'whether prints debug information', shortname='-v')

    input_args = parser.parse_args()
    print_arguments(input_args)
    run_ips_synchronizer(input_args)
