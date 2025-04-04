import argparse
import functools
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Start audio resampling server.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('host', str, '0.0.0.0', 'host address to bind the server', shortname='-H')
    add_arg('port', int, 5002, 'port number to bind the server', shortname='-P')
    return parser


def get_app():
    """
    Creates the WSGI app using the configuration from environment variables.
    Raises an error if the required environment variable is not set.
    """
    from openmmla.services.asr import AudioResampler
    from openmmla.utils.apps import create_app

    project_dir = os.environ.get("AUDIO_RESAMPLER_PROJECT_DIR")

    if not project_dir:
        print("WARNINGS: Environment variable AUDIO_RESAMPLER_PROJECT_DIR not set. Using current working directory.")

    return create_app(
        class_type=AudioResampler,
        endpoint='resample',
        method_name='process_request',
        class_args={'project_dir': project_dir},
    )


# Create a module-level WSGI app for use with WSGI servers like gunicorn
try:
    app = get_app()
except Exception:
    app = None


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    os.environ["AUDIO_RESAMPLER_PROJECT_DIR"] = args.project_dir if args.project_dir else os.getcwd()

    application = get_app()
    application.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
