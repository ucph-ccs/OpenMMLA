import argparse
import sys

from openmmla.collection.recording import (
    DEFAULT_AUDIO_CHANNEL,
    DEFAULT_AUDIO_DEVICE_LINUX,
    DEFAULT_AUDIO_DEVICE_MACOS,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_INPUT_FORMAT_LINUX,
    DEFAULT_AUDIO_INPUT_FORMAT_MACOS,
    DEFAULT_AUDIO_SAMPLE_RATE,
)


def _strtobool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _default_audio_input_format() -> str:
    return DEFAULT_AUDIO_INPUT_FORMAT_MACOS if sys.platform == "darwin" else DEFAULT_AUDIO_INPUT_FORMAT_LINUX


def _default_audio_device() -> str:
    return DEFAULT_AUDIO_DEVICE_MACOS if sys.platform == "darwin" else DEFAULT_AUDIO_DEVICE_LINUX


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla collect-audio",
        description="Record raw audio for post-time OpenMMLA file-source processing.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150),
    )
    parser.add_argument("-p", "--project-dir", default=None, help="OpenMMLA project directory; defaults to the current working directory")
    parser.add_argument("-o", "--output-root", default="artifacts", help="recording output root, relative to project-dir unless absolute")
    parser.add_argument("--session-id", default=None, help="recording session id; auto-generated when omitted")
    parser.add_argument("--initial-sync-time", type=float, default=None, help="shared unix timestamp used later as file-source initial_sync_time")
    parser.add_argument("--host-label", default=None, help="host label used in output filenames")
    parser.add_argument("--input-format", "--audio-input-format", dest="input_format", default=_default_audio_input_format(), help="ffmpeg input format, e.g. avfoundation or alsa")
    parser.add_argument("--device", "--audio-device", dest="device", default=_default_audio_device(), help="audio device id/name; avfoundation index 0 becomes :0")
    parser.add_argument("--channels", "--audio-channels", dest="channels", type=int, default=None, help="input channel count; auto-detected when omitted")
    parser.add_argument(
        "--channel", "--audio-channel",
        dest="channel",
        default=DEFAULT_AUDIO_CHANNEL,
        help="0-based channel to record as mono, or 'mix' to average all input channels",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_AUDIO_SAMPLE_RATE, help="output sample rate")
    parser.add_argument("--format", "--audio-format", dest="audio_format", choices=["wav", "flac", "aac"], default=DEFAULT_AUDIO_FORMAT, help="output audio format")
    parser.add_argument(
        "--interactive", "--audio-interactive",
        dest="interactive",
        nargs="?",
        const=True,
        default=False,
        type=_strtobool,
        help="list audio devices and prompt for device and channel before recording",
    )
    parser.add_argument("--list-devices", action="store_true", help="list devices for the selected ffmpeg input format and exit")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.collection.recording import list_audio_devices, prompt_audio_options, record_audio

    if args.list_devices:
        raise SystemExit(list_audio_devices(args.input_format))

    if args.interactive:
        selected = prompt_audio_options(
            input_format=args.input_format,
            device=args.device,
            channels=args.channels,
            channel=args.channel,
            sample_rate=args.sample_rate,
            audio_format=args.audio_format,
        )
        args.input_format = selected["input_format"]
        args.device = selected["device"]
        args.channels = selected["channels"]
        args.channel = selected["channel"]
        args.sample_rate = selected["sample_rate"]
        args.audio_format = selected["audio_format"]

    raise SystemExit(record_audio(
        project_dir=args.project_dir,
        output_root=args.output_root,
        session_id=args.session_id,
        initial_sync_time=args.initial_sync_time,
        host_label=args.host_label,
        input_format=args.input_format,
        device=args.device,
        channels=args.channels,
        channel=args.channel,
        sample_rate=args.sample_rate,
        audio_format=args.audio_format,
    ))


if __name__ == "__main__":
    main()
