import argparse
import sys

from openmmla.collection.recording import (
    DEFAULT_VIDEO_BITRATE_LINUX,
    DEFAULT_VIDEO_BITRATE_MACOS,
    DEFAULT_VIDEO_BUFSIZE_LINUX,
    DEFAULT_VIDEO_BUFSIZE_MACOS,
    DEFAULT_VIDEO_DEVICE_LINUX,
    DEFAULT_VIDEO_DEVICE_MACOS,
    DEFAULT_VIDEO_FRAMERATE,
    DEFAULT_VIDEO_INPUT_FORMAT_LINUX,
    DEFAULT_VIDEO_INPUT_FORMAT_MACOS,
    DEFAULT_VIDEO_MAXRATE_LINUX,
    DEFAULT_VIDEO_MAXRATE_MACOS,
    DEFAULT_VIDEO_PRESET,
    DEFAULT_VIDEO_SIZE,
    DEFAULT_VIDEO_SOURCE_FORMAT_LINUX,
    DEFAULT_VIDEO_SOURCE_FORMAT_MACOS,
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


def _default_video_input_format() -> str:
    return DEFAULT_VIDEO_INPUT_FORMAT_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_INPUT_FORMAT_LINUX


def _default_video_device() -> str:
    return DEFAULT_VIDEO_DEVICE_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_DEVICE_LINUX


def _default_video_source_format() -> str:
    return DEFAULT_VIDEO_SOURCE_FORMAT_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_SOURCE_FORMAT_LINUX


def _default_video_size() -> str:
    return DEFAULT_VIDEO_SIZE


def _default_video_bitrate() -> str:
    return DEFAULT_VIDEO_BITRATE_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_BITRATE_LINUX


def _default_video_maxrate() -> str:
    return DEFAULT_VIDEO_MAXRATE_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_MAXRATE_LINUX


def _default_video_bufsize() -> str:
    return DEFAULT_VIDEO_BUFSIZE_MACOS if sys.platform == "darwin" else DEFAULT_VIDEO_BUFSIZE_LINUX


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla collect-video",
        description="Record raw video for post-time OpenMMLA file-source processing.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150),
    )
    parser.add_argument("-p", "--project-dir", default=None, help="OpenMMLA project directory; defaults to the current working directory")
    parser.add_argument("-o", "--output-root", default="artifacts", help="recording output root, relative to project-dir unless absolute")
    parser.add_argument("--session-id", default=None, help="recording session id; auto-generated when omitted")
    parser.add_argument("--initial-sync-time", type=float, default=None, help="shared unix timestamp used later as file-source initial_sync_time")
    parser.add_argument("--host-label", default=None, help="host label used in output filenames")
    parser.add_argument("--input-format", "--video-input-format", dest="input_format", default=_default_video_input_format(), help="ffmpeg input format, e.g. v4l2 or avfoundation")
    parser.add_argument("--device", "--video-device", dest="device", default=_default_video_device(), help="video device path or ffmpeg input id")
    parser.add_argument("--source-format", "--video-source-format", dest="source_format", default=_default_video_source_format(), help="v4l2 camera source format, e.g. mjpeg")
    parser.add_argument("--framerate", default=DEFAULT_VIDEO_FRAMERATE, help="capture framerate")
    parser.add_argument("--size", default=_default_video_size(), help="capture size, e.g. 1920x1080")
    parser.add_argument("--bitrate", default=_default_video_bitrate(), help="target H.264 bitrate")
    parser.add_argument("--maxrate", default=_default_video_maxrate(), help="maximum H.264 bitrate")
    parser.add_argument("--bufsize", default=_default_video_bufsize(), help="H.264 rate-control buffer size")
    parser.add_argument("--preset", default=DEFAULT_VIDEO_PRESET, help="libx264 preset")
    parser.add_argument("--camera-label", default=None, help="camera label used in output filenames")
    parser.add_argument(
        "--interactive", "--video-interactive",
        dest="interactive",
        nargs="?",
        const=True,
        default=False,
        type=_strtobool,
        help="list video devices and prompt for device before recording",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    from openmmla.collection.recording import prompt_video_options, record_video

    if args.interactive:
        selected = prompt_video_options(
            input_format=args.input_format,
            device=args.device,
            source_format=args.source_format,
            framerate=args.framerate,
            size=args.size,
            bitrate=args.bitrate,
            maxrate=args.maxrate,
            bufsize=args.bufsize,
            preset=args.preset,
            camera_label=args.camera_label,
        )
        args.input_format = selected["input_format"]
        args.device = selected["device"]
        args.source_format = selected["source_format"]
        args.framerate = selected["framerate"]
        args.size = selected["size"]
        args.bitrate = selected["bitrate"]
        args.maxrate = selected["maxrate"]
        args.bufsize = selected["bufsize"]
        args.preset = selected["preset"]
        args.camera_label = selected["camera_label"]

    raise SystemExit(record_video(
        project_dir=args.project_dir,
        output_root=args.output_root,
        session_id=args.session_id,
        initial_sync_time=args.initial_sync_time,
        host_label=args.host_label,
        input_format=args.input_format,
        device=args.device,
        source_format=args.source_format,
        framerate=args.framerate,
        size=args.size,
        bitrate=args.bitrate,
        maxrate=args.maxrate,
        bufsize=args.bufsize,
        preset=args.preset,
        camera_label=args.camera_label,
    ))


if __name__ == "__main__":
    main()
