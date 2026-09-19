import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ips-vis",
        description="Run IPS visualizer of real-time indoor positioning system.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('store', bool, False, 'whether to store the real-time visualizations', shortname='-s')
    add_arg('session_id', str, None, 'session id to use; if set, start at once and exit when the run ends; '
            'if not set, choose/create one interactively', shortname='-sid')
    add_arg('dimension', str, '2d', 'plot the positions in 2d or 3d', shortname='-d')
    return parser


def parse_dimension(value) -> tuple[bool, str]:
    """(use_3d, problem) for the -d/--dimension value: 2d or 3d, in any case (empty is 2d); anything
    else plots in 2d and says why, so the visualizer shows its menu instead of starting."""
    text = str(value or '2d').strip().lower() or '2d'
    if text in ('2d', '3d'):
        return text == '3d', ''
    return False, (f"-d/--dimension is '{value}', which is neither 2d nor 3d: pick 2d or 3d on the IPS Base "
                   f"card, or choose it in the IPS Visualizer menu below.")


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.bases.ips import IPSVisualizer
    from openmmla.utils.args import print_arguments

    print_arguments(args)

    use_3d, problem = parse_dimension(args.dimension)
    ips_visualizer = IPSVisualizer(
        project_dir=args.project_dir,
        config_path=args.config_path,
        store=args.store,
        use_3d=use_3d,
        session_id=args.session_id
    )
    ips_visualizer.run(launch_problem=problem)


if __name__ == "__main__":
    main()
