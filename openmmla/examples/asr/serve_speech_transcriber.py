import argparse
import functools
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Start speech transcription server.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('host', str, '0.0.0.0', 'host address to bind the server', shortname='-H')
    add_arg('port', int, 5005, 'port number to bind the server', shortname='-P')
    return parser


def get_app():
    """
    Creates the WSGI app using configuration from environment variables.
    Raises an error if the required environment variables are not set.
    """
    from openmmla.services.asr import SpeechTranscriber
    from openmmla.utils.apps import create_app

    project_dir = os.environ.get("SPEECH_TRANSCRIBER_PROJECT_DIR")
    config_path = os.environ.get("SPEECH_TRANSCRIBER_CONFIG_PATH")

    if not project_dir:
        print("WARNINGS: Environment variable SPEECH_TRANSCRIBER_PROJECT_DIR not set. Using current working directory.")

    if not config_path:
        raise RuntimeError("Environment variable SPEECH_TRANSCRIBER_CONFIG_PATH must be set.")

    return create_app(
        class_type=SpeechTranscriber,
        endpoint='transcribe',
        method_name='process_request',
        class_args={'project_dir': project_dir, 'config_path': config_path},
    )


# Create module-level app for WSGI servers (e.g., gunicorn)
try:
    app = get_app()
except Exception:
    app = None


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.utils.args import print_arguments
    print_arguments(args)

    os.environ["SPEECH_TRANSCRIBER_PROJECT_DIR"] = args.project_dir
    os.environ["SPEECH_TRANSCRIBER_CONFIG_PATH"] = args.config_path

    application = get_app()
    application.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
