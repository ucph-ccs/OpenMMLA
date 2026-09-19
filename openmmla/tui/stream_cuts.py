"""capture-side recordings, by session.

A managed stream with `record: true` writes one file per run on its capture
host, <record_root>/streams/capture/<YYYY-MM-DD>/<host label>/<video|audio>/
<name>_<start>.<mkv|wav>, filed by day because the stream is shared by the
sessions that pull it. The footage of one session is the part of those files
between the session's start and end. It is cut on the capture host, with the
ffmpeg that made the recording and without re-encoding, so only the part that
is wanted travels; the cuts wait under <record_root>/streams/.session-cuts/
<session>/<host label>/<video|audio>/ until they are fetched.

The shell commands here run through `bash -c` on the capture host (or on this
machine for a local stream); everything else is plain parsing, so it can be
tested without a host."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from openmmla.utils.artifact_paths import CAPTURE_DAY_GLOB, CAPTURE_RECORD_REL, SESSION_CUTS_DIR, STREAMS_DIR

# folder under <record_root>/streams/ that holds cuts until they are fetched
STAGING_DIR = SESSION_CUTS_DIR

EXTENSIONS = {"video": "mkv", "audio": "wav"}

# shorter than this is a recording that ended as the session began
MIN_CUT_SECONDS = 1.0

# a cut this much shorter than asked for is still whole (artifacts.MEDIA_SLACK_SECONDS):
# a video cut starts on a keyframe and container lengths are rounded
CUT_SLACK_SECONDS = 1.5


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


# where a capture host's ffmpeg and ffprobe are found by a command sshd starts
# (no login profile read); the PATH a stream's own ffmpeg is started with
# (stream_panel.STREAM_REMOTE_PATH, which a test holds equal to this)
TOOL_PATH = "/opt/homebrew/bin:/usr/local/bin:/opt/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def with_tool_path(command: str) -> str:
    return f"export PATH={TOOL_PATH}:$PATH; {command}"


def list_script(record_root: str, host_label: str, kind: str, stream_name: str) -> str:
    """print `<mtime> <path>` for every recording of one stream, any day."""
    pattern = (
        f"{_quote_root(record_root)}/{CAPTURE_RECORD_REL}/{CAPTURE_DAY_GLOB}/{shlex.quote(host_label)}/{kind}/"
        f"{shlex.quote(stream_name)}_*.{EXTENSIONS[kind]}"
    )
    return (
        f"for f in {pattern}; do "
        'if [ -f "$f" ]; then '
        # GNU stat and BSD stat spell the format differently
        'm=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null); '
        'printf "%s %s\\n" "$m" "$f"; fi; done; echo LISTED'
    )


def parse_listing(text: str, stream_name: str | None = None) -> list[RecordedFile] | None:
    """None when the listing never finished (the host could not be asked).
    With `stream_name`, only its own files: the glob of cam also finds those
    of a stream called cam_x."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    if "LISTED" not in lines:
        return None
    files = []
    for line in lines:
        mtime, _, path = line.partition(" ")
        if not path.startswith("/") or not mtime.isdigit():
            continue
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        name, _, stamp = stem.rpartition("_")
        if stream_name is not None and name != stream_name:
            continue
        try:
            start = float(stamp)
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
        # best_effort_timestamp_time: ffprobe 4 (Ubuntu 22.04, Raspberry Pi OS) has no pts_time for a frame
        "ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=best_effort_timestamp_time "
        "-of csv=p=0 "
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
    `destination_dir` is quoted here unless the caller already did.

    ffmpeg writes <name>.part, which takes the cut's name only once ffmpeg has
    finished: a cut that failed or was stopped never passes for a whole one
    (its header already says the full length), and the transfers leave *.part
    alone. A whole cut already there, staged by an earlier export whose fetch
    did not finish, is kept as it is (KEPT), so that fetch resumes where it
    stopped instead of starting on a new file. Prints CUT when the cut is there."""
    folder = destination_dir if quoted_dir else shlex.quote(destination_dir)
    name = cut_name(stream_name, cut, kind)
    target = f"{folder}/{shlex.quote(name)}"
    part = f"{folder}/{shlex.quote(name + '.part')}"
    fmt = "matroska" if kind == "video" else "wav"
    whole = max(0.0, cut.duration - CUT_SLACK_SECONDS)
    # -ss before -i seeks by keyframe, which is where on_keyframe() put the cut;
    # the half millisecond keeps a rounded keyframe time from landing on the one before
    seek = cut.offset + 0.0005 if kind == "video" and cut.offset > 0 else cut.offset
    return (
        f"mkdir -p {folder} || exit 1; "
        f"if [ -s {target} ] && d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 {target} "
        f"2>/dev/null) && awk -v d=\"$d\" 'BEGIN {{ exit !(d + 0 >= {whole:.3f}) }}'; then echo KEPT; echo CUT; "
        f"elif ffmpeg -nostdin -v error -y -ss {seek:.6f} -i {shlex.quote(cut.source.path)} -t {cut.duration:.3f} "
        f"-map 0 -c copy -f {fmt} {part}; then mv -f {part} {target} && echo CUT; "
        f"else rm -f {part}; echo FAILED; fi"
    )


def staging_root(record_root: str, session_id: str) -> str:
    """where the cuts of one session wait on the capture host,
    <record_root>/streams/.session-cuts/<session>, shell-quoted."""
    return f"{_quote_root(record_root)}/{STREAMS_DIR}/{STAGING_DIR}/{shlex.quote(session_id)}"


def staging_dir(record_root: str, session_id: str, host_label: str, kind: str | None = None) -> str:
    """the staged cuts of one host label (and kind) of a session, shell-quoted:
    the tree a transfer fetches into artifacts/<session>/streams/capture/<host label>/."""
    folder = f"{staging_root(record_root, session_id)}/{shlex.quote(host_label)}"
    return f"{folder}/{kind}" if kind else folder


def resolve_script(quoted_dir: str) -> str:
    """print the absolute path of a folder given shell-quoted ($HOME expanded,
    links resolved) when it is there; RESOLVED when done."""
    return f"d={quoted_dir}; if [ -d \"$d\" ]; then (cd \"$d\" && pwd -P); fi; echo RESOLVED"


def parse_resolved(text: str | None) -> str | None:
    """the path resolve_script() printed; None when the folder is not there or
    the host could not be asked."""
    lines = [line.strip() for line in str(text or "").splitlines()]
    if "RESOLVED" not in lines:
        return None
    paths = [line for line in lines if line.startswith("/")]
    return paths[-1] if paths else None


def cleanup_script(record_root: str, session_id: str, host_label: str | None = None) -> str:
    """remove the staged cuts of a session (never the recordings themselves):
    those of one host label when given, then the session's folder once it is empty."""
    if host_label is None:
        return f"rm -rf {staging_root(record_root, session_id)}; echo CLEANED"
    return (
        f"rm -rf {staging_dir(record_root, session_id, host_label)}; "
        f"rmdir {staging_root(record_root, session_id)} 2>/dev/null; echo CLEANED"
    )
