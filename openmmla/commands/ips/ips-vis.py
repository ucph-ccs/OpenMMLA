import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run IPS video visualizer for visualizing the bases results.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('store', bool, False,
            'whether to store the visualization images in the local directory', shortname='-s')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import VideoVisualizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    visualizer = VideoVisualizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        store=args.store
    )
    visualizer.run()


if __name__ == "__main__":
    main()
