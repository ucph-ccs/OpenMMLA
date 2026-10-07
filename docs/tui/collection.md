# Collection

`Launcher → Collection → Collection Session` records raw audio and video with FFmpeg on one or more machines, into one session. Use it to keep a session's recordings for a later replay through the pipelines with `source: file`, or for coding by hand.

![Collection Session card, Audio tab: Num Audio, the Start Audio, Stop, Stop All Hosts, Logs, Refresh, Download and Delete Remote buttons, the Session ID, Experiment Group and Output Root fields, and the recorder table with Host, Device, Channel, Device Label and Participant](../img/tui/collection.png)

## Record a session

1. **Pick the session.** Leave **Session ID** on `Create MongoDB Session` and pick the **Experiment Group** for a new take, or pick a running session to record into it.
2. **Set up the recorders.** On the **Audio** and **Video** tabs, set the recorder count and fill each recorder's row in the [recorder table](#recorder-table).
3. **Press Start Audio and Start Video.** Each opens one terminal window per recorder of its tab, on every host of the tab at once.
4. **Press Stop** when the session is over. It stops the recorders on every host of the card and marks the session ended.
5. **Press Download** to copy the remote recordings here ([Download a session](#downloading-a-session)), or use **Sessions → Export**, which takes them along with the rest of the session.

!!! note "A Mac records only with someone logged in"
    A Mac that records needs someone logged in on its screen, with Terminal allowed to use the camera and the microphone ([Recording on a Mac](#recording-on-a-mac)).

??? info "Details: what Start does"
    - Start resolves the session first: `Create MongoDB Session` makes the id from the Experiment Group. A session id MongoDB does not have (deleted, or only known from files on disk) is registered again under the selected Experiment Group.
    - It gets every remote host of the tab ready at once: it checks SSH and copies the recorder code there, since a recording host needs no checkout. A host that cannot be reached is named in the log and left out, and the others record.
    - Each recorder runs in its own terminal window, this machine's here and a remote host's over SSH, each writing into the Output Root of its own machine. The input formats follow the platform: `avfoundation` on macOS, `alsa` and `v4l2` on Linux.
    - A recorder with a Device asks nothing (`--audio-interactive false --audio-device hw:2,0 --audio-channel mix --audio-channels 1`). One without asks in its terminal for its device and channel, and, given several channels, the wearer of each (the card's Participant is the first channel's default).

## Card fields

| Field | Default | What it does |
|---|---|---|
| **Num Audio**, **Num Video** | `1` | recorders on each tab, one table row each; `0` skips that role |
| **Session ID** | `Create MongoDB Session` | an existing session, or a new one from the **Experiment Group**; a session deleted in the Sessions tab is not offered |
| **Experiment Group** | | the `<experiment>/<group>` a new session is made in ([Experiments](system-settings.md#experiments)) |
| **Output Root** | `artifacts` | where every host writes, into `<session>/collection/<host>/`; a remote host reads `artifacts` as `~/artifacts` |

## Recorder table

One row per recorder. Click a cell, or press Enter on it, to open its dropdown. The Host selector at the top of the Launcher plays no part in this card and reads `Per recorder`.

| Column | Tab | What it does |
|---|---|---|
| **Host** | both | the machine the recorder runs on: `Local` (where new recorders start) or an SSH profile, Windows ones excluded; moving a recorder clears its Device |
| **Device** | both | what it opens, from the devices its Host has (with a microphone's channel count when known); **type another…** takes one not listed; `-` makes the recorder ask in its terminal |
| **Channel** | audio | `mix` (all channels averaged into one mono file), `each` (a file per channel), `ch0`, `ch1`, ..., or a list typed after **type another…** (`0,2`); picking a Device sets `mix`, or `each` for a worn receiver (`vimo`, `badge`) of several channels |
| **Device Label** | both | the device's name in the file names, offered from the `Streams` entries of the pipeline configs; recorder *n* opens on the *n*-th name; **type another…** takes any name (`badge-0`) |
| **Rotate** | video | clockwise turn as it records, as a stream's Rotate turns it: `0°` (default, as the camera gives it), `180°` (a camera mounted upside down), `90°` or `270°` (one on its side); filled from the label's video `Streams` entry |
| **Participant** | audio | whose voice it records: `<name> (tag <id>)` of the session's group, **Group (room microphone)**, or **bind later**; a recorder of several channels gets a row per channel |

Start refuses two recorders of one tab on the same device of one machine, `each` on a device whose channel count is unknown, and one tag on two rows or channels.

??? info "Details: Device and Channel"
    - Devices come from `arecord -l` (`hw:1,0`) and `v4l2-ctl --list-devices` (`/dev/video0`) on Linux, and `ffmpeg -f avfoundation -list_devices` on a Mac (`:0` for a microphone, `0` for a camera). Another machine is asked over SSH the first time the list opens, and the answer is kept until **Refresh**.
    - The channel count (`hw:2,0  Jabra SPEAK 510 USB — USB Audio (1 ch)`) comes from `system_profiler SPAudioDataType` on a Mac, and on Linux from `/proc/asound/card<N>/stream<M>` for a USB device (read without opening it), else `arecord --dump-hw-params` (a device in use says nothing). The device lists of the Streams tab and the Config tab show the count too.
    - A device whose count is unknown offers `mix`, `ch0` and `ch1`, and opening its Channel asks its machine. `mix` is the only choice for a one-channel device.
    - Start asks a Linux machine for the count of a device it does not know yet (one a Device Label filled in): ALSA opens a device with the channels it is given, and FFmpeg asks for two when given none, which a one-channel `hw:` device refuses.
    - One device is one FFmpeg. Record several channels of one device (two worn microphones on one receiver) as one recorder: one FFmpeg reads the device once and writes a file per channel (`vimo-0-ch0`, `vimo-0-ch1`) with the same start, so they hold the same samples and lose the same ones. Two recorders on one device would each open it, start at different moments and lose audio on their own.

??? info "Details: Device Label"
    - Microphones come from `pipelines/asr-base/config.yml` (`jabra-1`), cameras from `pipelines/ips-base/config.yml` and `pipelines/vfa-base/config.yml` (`c920-01`). The list is read from this checkout, whichever host a recorder runs on.
    - The number in a label is the device's own, the one its stream carries: not a base id (the `id` of a `Bases` entry) and not a tag id.
    - A single channel of a multi-channel device is appended (`vimo-0-ch1`); a mix appends nothing.
    - A label whose `Streams` entry names its machine and device (`ssh_profile: <host>`, `device: /dev/video0`) fills the row's Host and Device, unless they were picked by hand on the row (a Device picked by hand keeps its Host too). A machine that is no SSH profile here is not filled in. The log says what it filled or kept; Rotate is filled the same way.
    - A typed label stays the recorder's when the card is drawn again. A recorder without a label falls back to the FFmpeg device (`/dev/video0` gives `video0`, a Mac's first camera `0`), which names no device.
    - The recorder's terminal prints the label next to the device it takes, so a wrong pick shows before it records.

??? info "Details: Rotate"
    - FFmpeg turns the picture as it records, so the file is upright, and the manifest entry notes the turn as `rotate`. A recorder gets `--video-rotate <degrees>`; `0°` passes nothing.
    - A replay (`scripts/replay_sessions.py`) and `mmla ses-calibrate` read `rotate`, so the IPS and VFA bases turn the camera's intrinsics with the picture and report poses in the sensor's frame ([Calibrate from a recorded session](../pipelines/ips/calibration.md#calibrate-from-a-recorded-session)).
    - Every Linux recording is written as 4:2:0 (`format=yuv420p`, after the turn): the camera's MJPEG is 4:2:2, which H.264 would otherwise keep and few players decode.

??? info "Details: Participant"
    - A `jabra`, and a label a group base pulls, open on Group. Other microphones open on the group's tag ids, lowest first, in natural order of their file names (`vimo-0-ch0 < vimo-0-ch1 < vimo-1`). A recorder's own row reads `per channel` when it has channel rows.
    - The group comes from the Experiment Group for `Create MongoDB Session`, else from the picked session (an inactive experiment's too, else its MongoDB document). Without a participant list the column offers Group and bind later, and Start says so and goes ahead.
    - A pick is kept for that session and device (or channel). A tag another host already bound in the session is only warned about.
    - The pick goes into the manifest (`participant`, `scope: personal` or `scope: group`), and the session's MongoDB document notes `wearers` by device, which a live ASR base pulling that stream reads ([Personal microphones](../pipelines/asr/speakers-and-diarization.md#personal-microphones-and-energy-attribution)). What is left on bind later is bound with [`mmla ses-tidy`](session-tools.md#microphone-scope-and-wearers).

## Buttons

| Button | What it does |
|---|---|
| **Start Audio**, **Start Video** | start the recorders of their own tab; start both tabs of one take one after the other, into the same session |
| **Stop** | stops the session on every host of the card and marks it ended in MongoDB, unless one of those hosts still records it |
| **Stop All Hosts** | stops every recorder of the session on this machine, the hosts it was started on, and every other reachable profile (not Windows); then marks it ended |
| **Logs** | lists the live recorders of each host (role, session, pid); each prints into its own terminal window |
| **Refresh** | reads the card and the device lists again |
| **Download** | copies the recordings of every remote host of the card here ([Download a session](#downloading-a-session)) |
| **Delete Remote** | removes the recordings of every remote host of the card from those machines; the first press names each host and session, the second deletes |

The hosts of the card are this machine, the hosts its rows name on both tabs, and the hosts this console started its session on. The card reads `Running` while a recorder is alive on one of them, whichever session it records.

??? info "Details: which session the card shows"
    - The session id is remembered while this console runs. After a restart, or in a second console, the card opens on `Create MongoDB Session`; pick the running session to join it. Creating a session for an Experiment Group that still has an active one says so in the log.
    - While a host of the card records, the card opens on the session it records, and the log says which hosts record it. A session picked by hand meanwhile (to download an older one) is left alone until the recording changes.
    - After Stop or Stop All Hosts the card goes back to `Create MongoDB Session`, so the next Start is a new take. Stop, Download and Delete Remote then fall back, per host, to the session it recorded last.
    - A Start into an ended session picked by hand is held back once: pick `Create MongoDB Session` for a new take, or press Start again to record into it, which makes it active again.

??? info "Details: Delete Remote"
    - With the default Output Root it removes each host's `~/artifacts/<session>/collection/<host label>/`, then the session folder around it once only the session's manifests are left in it (also on a later press). Anything else there keeps it, and the log names what.
    - With an Output Root of its own, the whole `<root>/<session>/` goes.
    - It is refused while a Download of that session from one of those hosts runs, at either press, so a delete armed before the Download began is dropped. A Download is refused while that folder is being deleted.

## Files and manifests

Each host writes into `<Output Root>/<session>/collection/<host>/`. The folder names the machine that records (this machine's short name, or the SSH profile), and each file names the device after it: `<kind>_<host>_<device>_<start>.<ext>`. Each recorder writes `manifest.yml` and `manifest.json` beside the files, with `host` and `device` per recording, the shared `initial_sync_time` and ready-made `file_dir` values for the pipelines.

??? info "Details: dropped audio and the length checks"
    - Every recording starts its filters with `aresample=async=1`, which writes silence where the input's timestamps jump ahead by more than 10 ms and drops audio where they fall more than 10 ms behind, and never stretches it. A dropped buffer therefore does not shorten a file or move the audio after it earlier than the video. On a Mac this matters: FFmpeg's AVFoundation input holds one buffer of about 10 ms, and a WAV file keeps no timestamps.
    - When an audio recorder stops, it adds to each file's manifest entry `audio_seconds`, `wall_seconds` (from FFmpeg creating the file to closing it; on Linux, which keeps no creation time, from when the recorder saw it appear), their difference `shortfall_seconds`, and `startup_seconds` (from the start stamp in the name to the file's creation, the time FFmpeg took to open the device, so the first sample is about that much later than the stamp).
    - A file shorter by more than 0.5 s or 0.1 %, whichever is larger, gets a note: it lost audio its timestamps do not show, for example from a device that stopped delivering before the stop.

## Recording on a Mac

macOS lets an app use the camera and the microphone only once someone has allowed it, and a program started over SSH is never even asked. So on a remote Mac the recorder starts FFmpeg from a Terminal window on the Mac's own screen:

1. Log in on the Mac's screen.
2. Allow Terminal under **System Settings → Privacy & Security → Camera** and **Microphone**. The first recording asks there, on the Mac.

The window closes once FFmpeg runs, so closing it cannot end a recording. The recorder prints and asks in your terminal as usual, and Ctrl+C there, **Stop** and **Stop All Hosts** stop it. Linux hosts and Raspberry Pis need no such permission: a user in the `video` and `audio` groups records over SSH.

## Recording from a terminal

The card runs `mmla collect-audio` and `mmla collect-video`. The flags behind its columns are below; `--help` lists the rest (sample rate, format, size, bitrate, ...).

```bash
mmla collect-audio --device-label vimo-0 --channel 0,1 --participant 5,7
mmla collect-video --device-label c920-01 --rotate 180
```

| Flag | Command | Default | What it does |
|---|---|---|---|
| `--device` | both | audio: `0` on macOS (taken as `:0`), `default` on Linux; video: `0` on macOS, `/dev/video0` on Linux | the device to open |
| `--device-label` | both | the FFmpeg device | the device's name in the file names; any name |
| `--channel` | `collect-audio` | `0` | `mix`, one channel (`1`), several (`0,1`, one file each, one FFmpeg), or `each` |
| `--participant` | `collect-audio` | none | the wearer's tag id, one per channel in channel order (`5,group,none`) |
| `--scope` | `collect-audio` | from the device name | `personal` or `group`; `jabra` is group, `vimo` and `badge` personal |
| `--rotate` | `collect-video` | `0` | clockwise turn in degrees |
| `--interactive` | both | `False` | list the devices and ask for one (and a channel) before recording |

## Download a session { #downloading-a-session }

**Download** copies the recordings of every remote host of the card into `artifacts/<session>/collection/<host>/` here, one transfer per host, and merges the manifests; this machine's own recordings are in place already. It takes each host's whole session folder, audio and video, whichever tab it is pressed on. A progress bar under the log shows the bytes, rate and time left, with **Cancel** beside it. **Sessions → Export** runs the same transfer for every host the session recorded on, without the card.

??? info "Details: staging and resume"
    - It lists the files before it starts. A recording already here at the same size is not fetched again, since a recording is named after the moment it started and is not written again once it is over. A second press brings only what is new, plus the manifests, which a stop rewrites.
    - Files are staged under `artifacts/<session>/.staging/` and merged only once every file has arrived at its full remote size, so an interrupted download never leaves a truncated recording.
    - After an interruption (the network drops, **Cancel**, or quitting the console) the staged data is kept, and **Download** goes on where it stopped. With `rsync` on both machines the resume is byte-exact; with scp it is per file, and a partly written file is fetched again. The log names the transport.
    - A second Download for a session and host that is already downloading is ignored.

## Troubleshooting

**A download leaves some files in staging.** The session is still recording: growing files are named in the log and kept in staging, since FFmpeg finalizes a container only when the recorder stops. Stop the recorders and download again.

**On a Mac, a camera recorder waits for frames forever, or a microphone records silence.** Terminal has no permission for the camera or the microphone, or nobody is logged in on the Mac's screen ([Recording on a Mac](#recording-on-a-mac)).

**Start refuses two recorders.** Two recorders of one tab are on the same device of one machine. Record its channels with one recorder and **Channel** `each` or a list.
