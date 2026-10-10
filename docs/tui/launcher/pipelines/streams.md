# Streams tab

Every base card (ASR, IPS and VFA) has a **Streams** tab that runs its pipeline's streams: it starts FFmpeg on the machine that captures each camera or microphone, stops it, records it, and shows whether the Stream Server receives it. How streams work in general is in [Streaming](../../../streaming/index.md).

The streams are the `Streams` entries of the pipeline's config, defined on the card's [Config tab](index.md#config-tab), where `target` takes the path alone (`ips/cam-1`) and **Save** completes it with the Stream Server of System Settings.

![A pipeline card's Streams tab: one row per camera stream with its machine, device, Record, Rotate, status and Stream Server column](../../../img/tui/streams.png)

## Columns

Rows are listed by name, a number in a name counted as one (`cam-2` before `cam-10`). Click a dropdown cell, or press Enter on the row, to open it.

| Column | What it does |
|---|---|
| **Name** | the stream's name, which is one capture ([Stream names](#stream-names)) |
| **SSH Profile** | dropdown: the machine that runs the stream's FFmpeg: `local`, an SSH profile, or `-` for a stream someone else publishes (`External`) |
| **Device** | dropdown: the microphones or cameras that machine has, by the stream's `kind`; `-` lets Start take the machine's first; **type another…** takes one not listed |
| **Target** | where FFmpeg sends the stream |
| **Record** | dropdown, `yes` or `no`: the stream's `record`, a raw recording on the capture machine, shown with its keep time (`yes, 7 d`); `n/a` for an external stream |
| **Rotate** | dropdown: the stream's `rotate`, how the capture machine turns the picture clockwise: `0°`, `180°` (a camera mounted upside down), `90°`, `270°`; `-` for a microphone, `n/a` for an external stream |
| **Status** | the stream's FFmpeg ([Stream status](#stream-status)) |
| **Stream Server** | `● live` or `○ not live`: whether the Stream Server of System Settings receives it |

All four picks are written into the card's config on its host. **SSH Profile** and **Device** do not show on the Config tab, and its Save keeps them; **Record** and **Rotate** do. A running stream keeps the machine, Record and Rotate it was started with, so stop it before changing them; it takes a new device at its next Start.

??? info "Details: the Device list"
    - Linux microphones come from `arecord -l` (`hw:1,0`), Linux cameras from `v4l2-ctl --list-devices` (`/dev/video0`), a Mac's devices from `ffmpeg -f avfoundation -list_devices` (`:0` for a microphone, `0` for a camera). Microphones show their channel count when the machine says it ([Recorder table](../collection/index.md#recorder-table)).
    - Another machine is asked over SSH the first time the list opens, and again after **Refresh**.
    - A device picked for one machine goes when the stream moves to another.

## Buttons

| Button | What it does |
|---|---|
| **Start** | installs what the host lacks ([Tools Start installs](#tools-start-installs)), then starts the selected stream in a tmux session named `mmla-stream-<name>` on its host, and waits until FFmpeg has run three seconds |
| **Stop** | Ctrl-C to FFmpeg, which finishes its recording, then closes the tmux session; names the recording |
| **Logs** | shows the last lines of the stream's tmux pane |
| **Probe** | FFmpeg on this machine decodes two seconds of the URL the bases pull (`read_target`, else `target`); `rtmp://`, `rtsp://` and `srt://` URLs only |
| **Start All**, **Stop All** | the same for every row, in the order listed |
| **Manage** | in the **Recordings** row above the table: the recordings on each capture host ([Recordings and Manage](#recordings-and-manage)) |
| **Refresh** | asks every host again |

A stream with an `ssh_profile` sends H.264 to MediaMTX for an `rtmp://`, `rtsp://` or `srt://` target, and raw PCM for the `udp://` and `tcp://` targets of ASR bases. A stream without one is `External` and only pulled from.

??? info "Details: Start, Stop and Refresh"
    - Start quotes FFmpeg's last lines when it does not run for three seconds.
    - A Start or Stop sent to a host goes on to its end when the card is refreshed or left, notes what came of it in the stream registry, and still writes its lines into the log.
    - A Refresh notes the start time a host wrote down for a running stream the registry does not have (a log line says so), and notes as stopped a stream whose host has nothing left of it.
    - Stop looks for a stream on the machine its SSH Profile names, which is why a running stream keeps its machine.
    - A Mac captures through AVFoundation (`device` `0` is its first camera, `:0` its microphone). On a Mac reached over SSH, FFmpeg is started from a Terminal window on its screen, which closes once FFmpeg runs ([Recording on a Mac](../collection/index.md#recording-on-a-mac)).
    - The buttons wrap onto another line when the pane is too narrow.

### Stream status

| Status | What it means |
|---|---|
| `Asking` | its machine has not answered yet |
| `Starting`, `Running` | FFmpeg runs |
| `Stopped` | nothing runs |
| `Exited` | FFmpeg stopped by itself while its tmux session outlived it; **Logs** shows why, **Start** starts it again |
| `Installing` | Start is installing tools on the host |
| `No answer` | its machine did not answer |
| `External` | no `ssh_profile`; the console only pulls from it |
| `Name clash` | another card gives this name to another stream; Start does not start it |
| `Name in use` | its machine runs another card's stream under this name |

A push the Stream Server dropped does not end FFmpeg: it opens the push again every five seconds, and the recording goes on meanwhile. Such a row stays `Running` while **Stream Server** reads `○ not live` ([Declare a stream](../../../streaming/index.md#devices-pushing-a-stream)).

### Tools Start installs

Before it starts a stream, **Start** checks the host for `tmux`, `ffmpeg` and the Device list's tool (`v4l2-ctl` for a camera, `arecord` for a microphone), and installs what is missing: with `apt-get` through `sudo` on Debian, Ubuntu and Raspberry Pi OS, or `brew` on a Mac.

??? info "Details: the install"
    - `sudo` is answered with the SSH profile's password, or on `local` with the Sudo password of System Settings.
    - The row reads `Installing`. A second stream on the same host waits for that install instead of starting its own; one install runs on a host at a time, the Environment tab's included.
    - A host still without tmux or ffmpeg starts nothing. The log says why (sudo refused the password, apt busy, ...) and what to type there by hand.

## Stream names

A stream's name is what it is on its machine (the tmux session) and in the stream registry, so a name is one capture. Cards that name a stream the same share it when they reach the same path on the Stream Server and run it on the same machine, or one of them is external, as an IPS and a VFA Base pulling one camera do. Any other use of one name is a clash: rename one of the entries on its card's Config tab, and its target can stay.

??? info "Details: clashes, names in use and left-over captures"
    - A clash is another path (or, where the stream goes elsewhere, another target) under the name, or managed entries on two machines, checked against the `Streams` of every card.
    - A row whose machine runs another card's stream under its name reads `Name in use`. Its **Stop** leaves that stream alone, and **Logs** says whose pane it shows.
    - An entry renamed while it runs leaves its capture running under the old name, listed below its card's entries to be stopped. Until it is, the other card's row reads `Name in use`.
    - A capture whose entry was deleted or renamed is listed as `<name>  (not in Streams)`, from the stream registry, on the card whose app its target names. The row takes no edits; **Stop** and **Logs** work, and it goes once its host has nothing left of it.
    - While a card's config cannot be read, the log says so, and **Stop** leaves alone a capture that publishes somewhere other than the row's target.

## Turning the picture

**Rotate** turns the picture once, on the capture host, so the bases, both recordings and the dashboard get it upright, and the bases turn the camera's intrinsics with it.

!!! warning "A stream running with another turn"
    The bases read the config's turn. A running stream whose config was given another turn since shows both in yellow, `180° (runs 0°)`, and each **Refresh** logs it. Stop and start the stream: until then the bases turn the camera's intrinsics and poses for a picture that is not turned so, and its tags come out mirrored through the camera's axis.

??? info "Details: one camera on two cards"
    A Refresh also logs a camera that another card's entry turns otherwise, when the two share it (the same path on the Stream Server, on the same machine or pulled as external, by any name). The capture is turned once, as the card that starts it says, and each card's bases read their own entry's turn, so give the entries the same Rotate.

## Steady frame rate

With an entry's `steady_fps` on, the default, Start sets a Linux camera's `exposure_dynamic_framerate` to 0 (`exposure_auto_priority` on older kernels) with `v4l2-ctl`, so auto exposure cannot lower the frame rate in a dim room; the picture is darker instead. Turn it off on the Config tab for a brighter picture at a lower rate. A host without `v4l2-ctl`, or a camera without either control, starts without it.

## Recordings and Manage

A stream with **Record** `yes` also writes a raw recording on its capture host, one file per **Start**, filed by day under `<record_root>/streams/capture/`; **Stop** waits for FFmpeg to finish the file. The Stream Server records its own copy while a session runs, and an external stream has only that one. [Record on the capture host](../../../streaming/recording.md#on-the-capture-device) has the file layout.

**Manage** lists the recordings on each capture host (`Host`, `Day`, `Stream`, `Started`, `Last write`, `Length`, `Size`, `State`), with the room left there and the file being written now, and deletes them there; it copies nothing to this machine. [Manage the recordings](../../../streaming/recording.md#manage-the-recordings) lists its controls. A session's part of both copies comes with [Sessions → Export](../../sessions.md#export).
