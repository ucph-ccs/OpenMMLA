"""start a command where macOS lets it use the camera and the microphone.

macOS gives the camera and the microphone only to an app someone allowed, in
the desktop session. Anything started over ssh counts as sshd, which is never
asked: its ffmpeg waits for frames forever, or records silence, and says
nothing. A command started from a Terminal window on the Mac's own screen
counts as Terminal, which can be allowed. The window only starts it: a keeper
holds the command in a session of its own and notes its pid and exit status,
and the window closes itself once the command runs, so nothing stays on the
Mac's screen and no window closed by hand stops a recording or a stream.

Used by the Streams tab and by the Collection recorders, which run it on the
capture host: standard library only.
"""

from __future__ import annotations

import shlex


# holds the command for a window's script, in a session of its own: the window
# may close (it closes itself) without taking the command along. It notes the
# command's pid, clears the opening mark once the command runs, and notes its
# exit status once it has ended. argv: the file prefix, the command
KEEPER = """\
import os, signal, subprocess, sys
state, command = sys.argv[1], sys.argv[2]
try:
    os.setsid()
except OSError:
    pass
signal.signal(signal.SIGHUP, signal.SIG_IGN)
ffmpeg = subprocess.Popen(["/bin/bash", "-c", "exec " + command], stdin=subprocess.DEVNULL)
with open(state + ".pid", "w") as noted:
    noted.write("%d\\n" % ffmpeg.pid)
if os.path.exists(state + ".opening"):
    os.remove(state + ".opening")
code = ffmpeg.wait()
with open(state + ".rc", "w") as noted:
    noted.write("%d\\n" % (code if code >= 0 else 128 - code))
if os.path.exists(state + ".pid"):
    os.remove(state + ".pid")
"""

# closes the Terminal window whose tab has the given tty, when it holds that tab
# alone and nothing runs in it any more (Terminal would ask first otherwise).
# Terminal takes this from a process of its own windows, not from an ssh session
CLOSE_WINDOW = """\
on run argv
    tell application "Terminal"
        repeat with w in windows
            if (count of tabs of w) is 1 then
                if tty of tab 1 of w is item 1 of argv and not busy of tab 1 of w then
                    close w
                    return
                end if
            end if
        end repeat
    end tell
end run
"""

# runs CLOSE_WINDOW a second after the window's script has ended, from a session
# of its own, so that no process of the window is left when it is closed.
# argv: the AppleScript, the window's tty
WINDOW_CLOSER = """\
import os, subprocess, sys, time
try:
    os.setsid()
except OSError:
    pass
time.sleep(1)
command = ["osascript"]
for line in sys.argv[1].splitlines():
    command += ["-e", line]
subprocess.run(command + [sys.argv[2]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
"""

# seconds a window waits for its command to run before it closes anyway
WINDOW_WAIT_SECONDS = 10


def window_script(state: str, command: str, *, about: str, setup: str = "", python: str = "python3") -> str:
    """the bash script a Terminal window runs (a .command file for `open -a Terminal`).

    state is a shell word for the prefix of the files the script, the keeper and
    whoever follows them share: <state>.opening, which the caller makes before
    the window opens (the script starts nothing without it, so removing it
    cancels a start; the keeper removes it once the command runs), <state>.pid,
    <state>.rc (the exit status) and <state>.log (what the command prints).
    command is one shell command, run as `bash -c "exec <command>"`, so that the
    noted pid is the command's own; it sees what the setup lines export. python
    runs the keeper and the window closer.
    """
    words = shlex.quote
    return (
        "#!/bin/bash\n"
        f"# starts {about}: macOS lets a command started from this window use the camera and\n"
        "# the microphone, one started over ssh not. It runs on after this window has closed itself\n"
        f"F={state}\n"
        "TTY=$(tty)\n"
        f"close_window() {{ {python} -c {words(WINDOW_CLOSER)} {words(CLOSE_WINDOW)} "
        '"$TTY" </dev/null >/dev/null 2>&1 & }\n'
        "trap close_window EXIT\n"
        # stopped before this window came up
        '[ -e "$F.opening" ] || exit 0\n'
        f"echo {words(f'Starting {about}; this window closes by itself.')}\n"
        f"{setup}"
        f'{python} -c {words(KEEPER)} "$F" {words(command)} </dev/null >"$F.log" 2>&1 &\n'
        # the window goes once the command runs, or has already stopped
        f'i=0; while [ -e "$F.opening" ] && [ ! -e "$F.rc" ] && [ $i -lt {WINDOW_WAIT_SECONDS * 2} ]; '
        "do sleep 0.5; i=$((i+1)); done\n"
    )
