"""capture-side recordings, by session.

A managed stream with `record: true` writes one file per run on its capture
host, filed by day because the stream is shared by the sessions that pull it.
The footage of one session is the part of those files between the session's
start and end. It is cut on the capture host, with the ffmpeg that made the
recording and without re-encoding, so only the part that is wanted travels.

The shell commands here run through `bash -c` on the capture host (or on this
machine for a local stream); everything else is plain parsing, so it can be
tested without a host."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

# folder under the record root that holds cuts until they are fetched
STAGING_DIR = ".session-cuts"

EXTENSIONS = {"video": "mkv", "audio": "wav"}

# shorter than this is a recording that ended as the session began
MIN_CUT_SECONDS = 1.0


@dataclass(frozen=True)
class RecordedFile:
    path: str     # absolute, on the capture host
    start: float  # unix time of the first frame: the file name says it
    end: float    # unix time of the last write: the file's mtime


@dataclass(frozen=True)
class Cut:
    source: RecordedFile
    offset: float    # seconds into the source
    duration: float  # seconds

    @property
    def start(self) -> float:
        return self.source.start + self.offset


def _quote_root(root: str) -> str:
    """a record root for the shell: $HOME and ~ stay expandable, the rest is quoted."""
    text = str(root or "").strip().rstrip("/") or "$HOME/artifacts"
    for prefix in ("$HOME/", "~/"):
        if text.startswith(prefix):
            return "$HOME/" + "/".join(shlex.quote(part) for part in text[len(prefix):].split("/") if part)
    if text in ("$HOME", "~"):
        return "$HOME"
    return shlex.quote(text)


def bash(script: str) -> str:
    """bash, whatever the login shell is: zsh stops at a glob that matches nothing."""
    return f"bash -c {shlex.quote(script)}"


def list_script(record_root: str, host_label: str, kind: str, stream_name: str) -> str:
    """print `<mtime> <path>` for every recording of one stream, any day."""
    pattern = (
        f"{_quote_root(record_root)}/streams-*/collection/{shlex.quote(host_label)}/{kind}/"
        f"{shlex.quote(stream_name)}_*.{EXTENSIONS[kind]}"
    )
    return (
        f"for f in {pattern}; do "
        'if [ -f "$f" ]; then '
        # GNU stat and BSD stat spell the format differently
        'm=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null); '
        'printf "%s %s\\n" "$m" "$f"; fi; done; echo LISTED'
    )


def parse_listing(text: str) -> list[RecordedFile] | None:
    """None when the listing never finished (the host could not be asked)."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    if "LISTED" not in lines:
        return None
    files = []
    for line in lines:
        mtime, _, path = line.partition(" ")
        if not path.startswith("/") or not mtime.isdigit():
            continue
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        try:
            start = float(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        files.append(RecordedFile(path, start, max(float(mtime), start)))
    return sorted(files, key=lambda item: item.start)


def cuts_for_window(files: list[RecordedFile], start: float, end: float) -> list[Cut]:
    """the part of each recording that falls between start and end (unix times)."""
    cuts = []
    for item in files:
        cut_start, cut_end = max(start, item.start), min(end, item.end)
        if cut_end - cut_start >= MIN_CUT_SECONDS:
            cuts.append(Cut(item, round(cut_start - item.start, 3), round(cut_end - cut_start, 3)))
    return cuts


def keyframe_script(cut: Cut) -> str:
    """print the times of the keyframes just before the cut. Without
    re-encoding a video can only begin on a keyframe, and the file name has to
    say when its first frame was taken, so the cut is moved back onto one."""
    window_start = max(0.0, cut.offset - 5.0)
    return (
        "ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=pts_time -of csv=p=0 "
        f"-read_intervals {window_start:.3f}%{cut.offset + 0.001:.3f} {shlex.quote(cut.source.path)} 2>/dev/null; echo PROBED"
    )


def on_keyframe(cut: Cut, probe_output: str) -> Cut | None:
    """the cut moved back to the last keyframe at or before it; None when
    ffprobe gave nothing to go by (it is missing, or the file has no index yet)."""
    times = []
    for line in str(probe_output or "").splitlines():
        try:
            times.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    times = [t for t in times if t <= cut.offset + 0.001]
    if not times:
        return None
    keyframe = max(times)
    return Cut(cut.source, round(keyframe, 6), round(cut.duration + (cut.offset - keyframe), 3))


def cut_name(stream_name: str, cut: Cut, kind: str) -> str:
    """<name>_<start>.<ext>: what the `file` source reads the start time from."""
    return f"{stream_name}_{cut.start:.6f}.{EXTENSIONS[kind]}"


def cut_script(cut: Cut, destination_dir: str, stream_name: str, kind: str, quoted_dir: bool = False) -> str:
    """write the cut next to the others of its session; no re-encoding.
    `destination_dir` is quoted here unless the caller already did."""
    folder = destination_dir if quoted_dir else shlex.quote(destination_dir)
    target = f"{folder}/{shlex.quote(cut_name(stream_name, cut, kind))}"
    fmt = "matroska" if kind == "video" else "wav"
    # -ss before -i seeks by keyframe, which is where on_keyframe() put the cut;
    # the half millisecond keeps a rounded keyframe time from landing on the one before
    seek = cut.offset + 0.0005 if kind == "video" and cut.offset > 0 else cut.offset
    return (
        f"mkdir -p {folder} && "
        f"ffmpeg -nostdin -v error -y -ss {seek:.6f} -i {shlex.quote(cut.source.path)} -t {cut.duration:.3f} "
        f"-map 0 -c copy -f {fmt} {target} && echo CUT"
    )


def staging_root(record_root: str, session_id: str) -> str:
    """where the cuts of one session wait on the capture host, shell-quoted."""
    return f"{_quote_root(record_root)}/{STAGING_DIR}/{shlex.quote(session_id)}"


def cleanup_script(record_root: str, session_id: str) -> str:
    """remove the staged cuts of a session (never the recordings themselves)."""
    return f"rm -rf {staging_root(record_root, session_id)}; echo CLEANED"
