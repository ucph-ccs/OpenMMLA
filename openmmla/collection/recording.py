from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from openmmla.utils.artifact_paths import collection_artifact_dir


DEFAULT_OUTPUT_ROOT = "artifacts"
DEFAULT_AUDIO_INPUT_FORMAT_MACOS = "avfoundation"
DEFAULT_AUDIO_INPUT_FORMAT_LINUX = "alsa"
DEFAULT_AUDIO_DEVICE_MACOS = "0"
DEFAULT_AUDIO_DEVICE_LINUX = "default"
DEFAULT_AUDIO_CHANNEL = "0"
DEFAULT_AUDIO_CHANNEL_COUNT_FALLBACK = 2
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_VIDEO_INPUT_FORMAT_MACOS = "avfoundation"
DEFAULT_VIDEO_INPUT_FORMAT_LINUX = "v4l2"
DEFAULT_VIDEO_DEVICE_MACOS = "0"
DEFAULT_VIDEO_DEVICE_LINUX = "/dev/video0"
DEFAULT_VIDEO_SOURCE_FORMAT_MACOS = ""
DEFAULT_VIDEO_SOURCE_FORMAT_LINUX = "mjpeg"
DEFAULT_VIDEO_FRAMERATE = "30"
DEFAULT_VIDEO_SIZE = "1920x1080"
DEFAULT_VIDEO_BITRATE_MACOS = "2M"
DEFAULT_VIDEO_BITRATE_LINUX = "3M"
DEFAULT_VIDEO_MAXRATE_MACOS = "2M"
DEFAULT_VIDEO_MAXRATE_LINUX = "6M"
DEFAULT_VIDEO_BUFSIZE_MACOS = "4M"
DEFAULT_VIDEO_BUFSIZE_LINUX = "10M"
DEFAULT_VIDEO_PRESET = "veryfast"


def short_hostname() -> str:
    return socket.gethostname().split(".", 1)[0] or "host"


def sanitize_label(value: str | None, default: str) -> str:
    raw = str(value or "").strip() or default
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    sanitized = sanitized.strip("._-")
    return sanitized or default


def format_epoch_ms(value: float | None = None) -> str:
    return f"{value if value is not None else time.time():.3f}"


def make_session_id(host_label: str | None = None) -> str:
    host = sanitize_label(host_label, short_hostname())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"collection_{stamp}_{host}"


def resolve_project_dir(project_dir: str | None) -> Path:
    return Path(project_dir or os.getcwd()).expanduser().resolve()


def resolve_output_root(project_dir: Path, output_root: str | None) -> Path:
    root = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser()
    if not root.is_absolute():
        root = project_dir / root
    return root.resolve()


def is_artifact_output_root(output_root: str | None) -> bool:
    value = str(output_root or DEFAULT_OUTPUT_ROOT).strip().rstrip("/")
    return value in {"", "artifacts", "~/artifacts"}


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _dump_yaml(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
    return lines


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def existing_initial_sync_time(session_dir: Path) -> float | None:
    value = _read_json(session_dir / "manifest.json").get("initial_sync_time")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_sync_time(*values: float | None) -> float:
    candidates = [float(value) for value in values if value is not None]
    return max(candidates) if candidates else time.time()


def update_manifest(
    session_dir: Path,
    session_id: str,
    initial_sync_time: float,
    recording: dict[str, Any],
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = session_dir / "manifest.json"
    manifest_yml = session_dir / "manifest.yml"
    lock_path = session_dir / ".manifest.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        data = _read_json(manifest_json)
        existing_sync_time = None
        try:
            existing_sync_time = float(data.get("initial_sync_time"))
        except (TypeError, ValueError):
            pass
        sync_time = _max_sync_time(initial_sync_time, existing_sync_time)
        data["session_id"] = session_id
        data["initial_sync_time"] = sync_time
        data.setdefault("created_at", format_epoch_ms(sync_time))
        data["updated_at"] = format_epoch_ms()
        data["file_sources"] = {
            "asr": {
                "source": "file",
                "file_dir": str(session_dir / "audio"),
                "initial_sync_time": sync_time,
            },
            "ips": {
                "source": "file",
                "file_dir": str(session_dir / "video"),
                "initial_sync_time": sync_time,
            },
            "vfa": {
                "source": "file",
                "file_dir": str(session_dir / "video"),
                "initial_sync_time": sync_time,
            },
        }
        recordings = data.setdefault("recordings", [])
        if not isinstance(recordings, list):
            recordings = []
            data["recordings"] = recordings

        for index, existing in enumerate(recordings):
            if isinstance(existing, dict) and existing.get("id") == recording.get("id"):
                merged = dict(existing)
                merged.update(recording)
                recordings[index] = merged
                break
        else:
            recordings.append(recording)

        tmp_json = manifest_json.with_suffix(".json.tmp")
        with tmp_json.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=False)
            file.write("\n")
        tmp_json.replace(manifest_json)

        tmp_yml = manifest_yml.with_suffix(".yml.tmp")
        with tmp_yml.open("w", encoding="utf-8") as file:
            file.write("\n".join(_dump_yaml(data)))
            file.write("\n")
        tmp_yml.replace(manifest_yml)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    _update_session_manifest(session_dir, session_id, sync_time, recording)


def _upsert_by_keys(entries: list[dict[str, Any]], entry: dict[str, Any], keys: tuple[str, ...]) -> None:
    for index, existing in enumerate(entries):
        if isinstance(existing, dict) and all(existing.get(key) == entry.get(key) for key in keys):
            merged = dict(existing)
            merged.update(entry)
            entries[index] = merged
            return
    entries.append(entry)


def _normalize_file_source_sync_times(data_sources: dict[str, Any], initial_sync_time: float) -> None:
    for sources in data_sources.values():
        if not isinstance(sources, list):
            continue
        for source in sources:
            if isinstance(source, dict) and source.get("source") == "file":
                source["initial_sync_time"] = initial_sync_time


def _update_session_manifest(
    collection_host_dir: Path,
    session_id: str,
    initial_sync_time: float,
    recording: dict[str, Any],
) -> None:
    if collection_host_dir.parent.name != "collection":
        return
    session_root = collection_host_dir.parent.parent
    if not session_root.name:
        return

    session_root.mkdir(parents=True, exist_ok=True)
    manifest_json = session_root / "manifest.json"
    manifest_yml = session_root / "manifest.yml"
    lock_path = session_root / ".manifest.lock"
    host = str(recording.get("host") or collection_host_dir.name)

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        data = _read_json(manifest_json)
        data["session_id"] = session_id
        data["initial_sync_time"] = _max_sync_time(initial_sync_time, data.get("initial_sync_time"))
        data.setdefault("created_at", format_epoch_ms(data["initial_sync_time"]))
        data["updated_at"] = format_epoch_ms()

        artifacts = data.setdefault("artifacts", {})
        collection = artifacts.setdefault("collection", [])
        if not isinstance(collection, list):
            collection = []
            artifacts["collection"] = collection
        _upsert_by_keys(
            collection,
            {
                "host": host,
                "local_path": str(collection_host_dir),
                "updated_at": data["updated_at"],
            },
            ("host", "local_path"),
        )

        data_sources = data.setdefault("file_sources", {})
        audio_dir = collection_host_dir / "audio"
        video_dir = collection_host_dir / "video"
        def upsert_source(modality: str, entry: dict[str, Any]) -> None:
            sources = data_sources.get(modality)
            if not isinstance(sources, list):
                sources = []
                data_sources[modality] = sources
            _upsert_by_keys(sources, entry, ("host", "source", "file_dir"))

        if audio_dir.is_dir():
            upsert_source("asr", {
                "host": host,
                "pipeline": "collection",
                "source": "file",
                "file_dir": str(audio_dir),
                "initial_sync_time": data["initial_sync_time"],
            })
        if video_dir.is_dir():
            video_source = {
                "host": host,
                "pipeline": "collection",
                "source": "file",
                "file_dir": str(video_dir),
                "initial_sync_time": data["initial_sync_time"],
            }
            upsert_source("ips", video_source)
            upsert_source("vfa", video_source)

        _normalize_file_source_sync_times(data_sources, data["initial_sync_time"])

        recordings = data.setdefault("recordings", [])
        if not isinstance(recordings, list):
            recordings = []
            data["recordings"] = recordings
        _upsert_by_keys(recordings, dict(recording), ("id",))

        tmp_json = manifest_json.with_suffix(".json.tmp")
        with tmp_json.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=False)
            file.write("\n")
        tmp_json.replace(manifest_json)

        tmp_yml = manifest_yml.with_suffix(".yml.tmp")
        with tmp_yml.open("w", encoding="utf-8") as file:
            file.write("\n".join(_dump_yaml(data)))
            file.write("\n")
        tmp_yml.replace(manifest_yml)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def prepare_recording_paths(
    *,
    modality: str,
    project_dir: str | None,
    output_root: str | None,
    session_id: str | None,
    initial_sync_time: float | None,
    host_label: str | None,
    leaf_dir: str,
    filename: str,
) -> tuple[Path, Path, str, float, str]:
    project = resolve_project_dir(project_dir)
    host = sanitize_label(host_label, short_hostname())
    session = sanitize_label(session_id, make_session_id(host))
    if is_artifact_output_root(output_root):
        session_dir = collection_artifact_dir(project, session, host)
    else:
        root = resolve_output_root(project, output_root)
        session_dir = root / session
    modality_dir = session_dir / leaf_dir
    modality_dir.mkdir(parents=True, exist_ok=True)
    manifest_sync_time = existing_initial_sync_time(session_dir)
    recording_start_time = None
    try:
        recording_start_time = float(Path(filename).stem.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        pass
    sync_time = _max_sync_time(initial_sync_time, manifest_sync_time, recording_start_time)
    output_file = modality_dir / filename
    print(f"Session: {session}")
    print(f"Initial sync time: {sync_time:.3f}")
    print(f"{modality.capitalize()} output: {output_file}")
    return session_dir, output_file, session, sync_time, host


def ensure_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found in PATH")


def run_recording_process(command: list[str]) -> int:
    print("Command:")
    print(" ".join(command))
    print("Press Ctrl+C to stop.")
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGINT)
        return proc.wait()


def list_audio_devices(input_format: str = "avfoundation") -> int:
    ensure_command("ffmpeg")
    command = ["ffmpeg", "-hide_banner", "-f", input_format, "-list_devices", "true", "-i", ""]
    try:
        subprocess.run(command, check=False)
    except OSError as e:
        print(f"Failed to list audio devices: {e}")
        return 1
    return 0


def probe_audio_channels(input_format: str, device: str) -> int | None:
    if shutil.which("ffprobe") is None:
        return None
    if input_format == "avfoundation":
        input_spec = device if ":" in device else f":{device}"
    else:
        input_spec = device
    command = [
        "ffprobe",
        "-v",
        "error",
        "-f",
        input_format,
        "-i",
        input_spec,
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def prompt_audio_options(
    *,
    input_format: str,
    device: str,
    channels: int | None,
    channel: str,
    sample_rate: int,
    audio_format: str,
) -> dict[str, Any]:
    print("\nListing audio devices:")
    list_audio_devices(input_format)
    print()

    selected_device = input(f"Audio device [{device}]: ").strip() or device
    detected_channels = channels or probe_audio_channels(input_format, selected_device)
    if detected_channels is None:
        forced = input(f"Input channel count [{DEFAULT_AUDIO_CHANNEL_COUNT_FALLBACK}]: ").strip()
        detected_channels = int(forced) if forced else DEFAULT_AUDIO_CHANNEL_COUNT_FALLBACK

    if detected_channels is not None:
        print(f"Detected/forced input channels: {detected_channels}")
        print(f"Channel choices: mix or 0..{max(0, detected_channels - 1)}")
    else:
        print("Could not detect channel count. Use mix or a 0-based channel number.")

    selected_channel = input(f"Audio channel [{channel}]: ").strip() or str(channel)
    _audio_channel_filter(selected_channel, detected_channels)

    return {
        "input_format": input_format,
        "device": selected_device,
        "channels": detected_channels,
        "channel": selected_channel,
        "sample_rate": sample_rate,
        "audio_format": audio_format,
    }


def _audio_channel_filter(channel: str, channel_count: int | None) -> tuple[str | None, str | int]:
    channel_text = str(channel).strip().lower()
    if channel_text in ("", "mix", "all", "auto"):
        return None, "mix"

    try:
        channel_index = int(channel_text)
    except ValueError as exc:
        raise ValueError("audio channel must be 'mix' or a 0-based channel number") from exc

    if channel_count is not None and channel_index >= channel_count:
        raise ValueError(f"audio channel {channel_index} is out of range for {channel_count} channels")
    if channel_index < 0:
        raise ValueError("audio channel must be non-negative")
    return f"pan=mono|c0=c{channel_index}", channel_index


def record_audio(
    *,
    project_dir: str | None,
    output_root: str | None,
    session_id: str | None,
    initial_sync_time: float | None,
    host_label: str | None,
    input_format: str,
    device: str,
    channels: int | None,
    channel: str,
    sample_rate: int,
    audio_format: str,
) -> int:
    ensure_command("ffmpeg")
    fmt = audio_format.lower()
    codecs = {
        "wav": ("pcm_s16le", "wav"),
        "flac": ("flac", "flac"),
        "aac": ("aac", "m4a"),
    }
    if fmt not in codecs:
        raise ValueError("audio format must be one of: wav, flac, aac")
    codec, extension = codecs[fmt]

    detected_channels = channels or probe_audio_channels(input_format, device)
    audio_filter, channel_label = _audio_channel_filter(channel, detected_channels)

    start_time = time.time()
    start_text = format_epoch_ms(start_time)
    host = sanitize_label(host_label, short_hostname())
    filename_channel = "mix" if channel_label == "mix" else f"ch{channel_label}"
    filename = f"audio_{host}_{filename_channel}_{start_text}.{extension}"
    session_dir, output_file, session, sync_time, host = prepare_recording_paths(
        modality="audio",
        project_dir=project_dir,
        output_root=output_root,
        session_id=session_id,
        initial_sync_time=initial_sync_time,
        host_label=host,
        leaf_dir="audio",
        filename=filename,
    )

    if input_format == "avfoundation":
        input_spec = device if ":" in device else f":{device}"
    else:
        input_spec = device
    input_options = ["-f", input_format]
    if channels:
        input_options.extend(["-ac", str(channels)])
    input_options.extend(["-i", input_spec])

    recording_id = f"audio_{host}_{filename_channel}_{start_text}"
    recording = {
        "id": recording_id,
        "modality": "audio",
        "status": "recording",
        "path": str(output_file),
        "start_time": float(start_text),
        "host": host,
        "input_format": input_format,
        "device": device,
        "channels": detected_channels,
        "channel": channel_label,
        "sample_rate": sample_rate,
        "format": fmt,
    }
    update_manifest(session_dir, session, sync_time, recording)

    command = [
        "ffmpeg",
        "-hide_banner",
        *input_options,
    ]
    if audio_filter:
        command.extend(["-af", audio_filter])
    else:
        command.extend(["-ac", "1"])
    command.extend([
        "-c:a",
        codec,
        "-ar",
        str(sample_rate),
        str(output_file),
    ])
    return_code = run_recording_process(command)
    recording.update({
        "status": "finished" if return_code == 0 else "stopped",
        "stopped_at": float(format_epoch_ms()),
        "returncode": return_code,
    })
    update_manifest(session_dir, session, sync_time, recording)
    return return_code


def _camera_label_from_device(device: str) -> str:
    label = os.path.basename(device.rstrip("/")) if device else "camera"
    return sanitize_label(label, "camera")


def list_video_devices(input_format: str = "v4l2") -> int:
    ensure_command("ffmpeg")
    if input_format == "v4l2":
        if shutil.which("v4l2-ctl") is not None:
            try:
                subprocess.run(["v4l2-ctl", "--list-devices"], check=False)
                return 0
            except OSError as e:
                print(f"Failed to list video devices: {e}")
                return 1
        print("Video devices:")
        for path in sorted(Path("/dev").glob("video*")):
            print(f"  /dev/{path.name}")
        return 0

    command = ["ffmpeg", "-hide_banner", "-f", input_format, "-list_devices", "true", "-i", ""]
    try:
        subprocess.run(command, check=False)
    except OSError as e:
        print(f"Failed to list video devices: {e}")
        return 1
    return 0


def prompt_video_options(
    *,
    input_format: str,
    device: str,
    source_format: str | None,
    framerate: str,
    size: str,
    bitrate: str,
    maxrate: str,
    bufsize: str,
    preset: str,
    camera_label: str | None,
) -> dict[str, Any]:
    print("\nListing video devices:")
    list_video_devices(input_format)
    print()

    selected_device = input(f"Video device [{device}]: ").strip() or device

    return {
        "input_format": input_format,
        "device": selected_device,
        "source_format": source_format or "",
        "framerate": str(framerate),
        "size": str(size),
        "bitrate": str(bitrate),
        "maxrate": str(maxrate),
        "bufsize": str(bufsize),
        "preset": str(preset),
        "camera_label": camera_label,
    }


def record_video(
    *,
    project_dir: str | None,
    output_root: str | None,
    session_id: str | None,
    initial_sync_time: float | None,
    host_label: str | None,
    input_format: str,
    device: str,
    source_format: str | None,
    framerate: str,
    size: str,
    bitrate: str,
    maxrate: str,
    bufsize: str,
    preset: str,
    camera_label: str | None,
) -> int:
    ensure_command("ffmpeg")
    start_time = time.time()
    start_text = format_epoch_ms(start_time)
    host = sanitize_label(host_label, short_hostname())
    camera = sanitize_label(camera_label, _camera_label_from_device(device))
    filename = f"video_{host}_{camera}_{start_text}.mp4"
    session_dir, output_file, session, sync_time, host = prepare_recording_paths(
        modality="video",
        project_dir=project_dir,
        output_root=output_root,
        session_id=session_id,
        initial_sync_time=initial_sync_time,
        host_label=host,
        leaf_dir="video",
        filename=filename,
    )

    input_options = ["-f", input_format]
    if input_format == "v4l2" and source_format:
        input_options.extend(["-input_format", source_format])
    if input_format == "avfoundation" and ":" not in str(device):
        input_spec = f"{device}:none"
    else:
        input_spec = device
    input_options.extend(["-framerate", str(framerate), "-video_size", size, "-i", input_spec])

    recording_id = f"video_{host}_{camera}_{start_text}"
    recording = {
        "id": recording_id,
        "modality": "video",
        "status": "recording",
        "path": str(output_file),
        "start_time": float(start_text),
        "host": host,
        "input_format": input_format,
        "device": device,
        "source_format": source_format,
        "camera_label": camera,
        "framerate": framerate,
        "size": size,
        "bitrate": bitrate,
        "maxrate": maxrate,
        "bufsize": bufsize,
        "preset": preset,
    }
    update_manifest(session_dir, session, sync_time, recording)

    if input_format == "avfoundation":
        command = [
            "ffmpeg",
            "-hide_banner",
            *input_options,
            "-c:v",
            "h264_videotoolbox",
            "-realtime",
            "true",
            "-g",
            str(framerate),
            "-b:v",
            bitrate,
            "-maxrate",
            maxrate,
            "-bufsize",
            bufsize,
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(output_file),
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            *input_options,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-tune",
            "zerolatency",
            "-g",
            str(framerate),
            "-keyint_min",
            str(framerate),
            "-sc_threshold",
            "0",
            "-x264-params",
            f"keyint={framerate}:min-keyint={framerate}:no-scenecut=1:repeat-headers=1",
            "-b:v",
            bitrate,
            "-maxrate",
            maxrate,
            "-bufsize",
            bufsize,
            "-f",
            "mp4",
            str(output_file),
        ]
    return_code = run_recording_process(command)
    recording.update({
        "status": "finished" if return_code == 0 else "stopped",
        "stopped_at": float(format_epoch_ms()),
        "returncode": return_code,
    })
    update_manifest(session_dir, session, sync_time, recording)
    return return_code
