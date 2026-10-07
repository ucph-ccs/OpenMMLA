# Live video and sound

The Live page plays camera video and microphone sound beside the measurements: live from the Stream Server while a session runs, and from the session's recordings in a replay. The Analysis page offers the same recordings for download.

!!! warning "Participants on camera"
    The camera tiles show the participants, and the sound carries their voices. Neither the dashboard nor MediaMTX has a login: keep both on a trusted network or a tailnet, and never expose ports 5050, 5051, 5052, 8889 or 8189 to the internet. One switch keeps the recordings and the live microphones off the dashboard: see [Keep the dashboard private](deploy.md#keep-the-dashboard-private).

## Camera tiles

The **Cameras** card of the Live page, closed until you open it, has one tile per camera of the session. VFA cameras come first, then IPS ones, each named by its stream and what took it: `cam-1 · VFA`, `cam-3 · IPS`, or `cam-2 · VFA + IPS` for a camera that a file-mode session's IPS and VFA bases both read.

- A **VFA camera's tile** draws the overlay of the newest frame set: boxes, COCO-17 skeletons, AprilTag centres with their tag, and gaze rays with the category they land on.
- An **IPS camera's tile** shows its video alone; its badge positions are on the **Room** card.

| Session | Under the overlay |
|---|---|
| running, with cameras streamed through MediaMTX | the live stream ([Live video](#live-video)) |
| ended and replayed, with the camera's file on the dashboard's machine | the recorded file ([Recorded video in a replay](#recorded-video-in-a-replay)) |
| anything else: no live stream, the replay of a session that is still running, the raw recordings turned off, no file of the camera on the dashboard's machine, or a file the browser cannot play | a blank stage that says `Skeleton view` and why |

### How a tile labels people

The tile draws each person by how sure it is of their AprilTag badge, and the live measures count a pupil's frames by the same rule. Hover a dashed or grey person to see why.

| Drawing | Label | When | Counted live |
|---|---|---|---|
| solid, in the tag's colour | `Tag 0` | the camera read the badge in this frame set | yes |
| dashed, in the tag's colour | `Tag 0? 23 s` | the VFA server kept the badge on the person's track from a read up to 60 s ago | yes |
| grey | `track 1836` | no counted badge: none read on the track, the last read more than 60 s ago, or a read before the moments the page has loaded | no |
| grey | `unknown 2` | a session the VFA server did not track, and no badge read on the person in this frame set | no |

??? info "Details: kept badges and the Analysis page"
    - A badge goes unread when it is out of view, turned away, or too small in the picture.
    - A kept badge counts for 60 s only, because a track that lost its pupil can be picked up by another body and carry the tag on to the wrong child.
    - A replay loads at most the five minutes before its moment, so a read before that is not seen.
    - The live measures leave a grey person's frames out, but another camera that reads the same badge at that moment still counts the pupil.
    - The Analysis page counts more of the `track` frames: it names a track's frames after the track's nearest read within 60 s, before or after them, which a live view cannot see coming. Neither page counts the frames of an `unknown` person.

### Tile buttons

| Button | What it does |
|---|---|
| **Hide video**, **Show video** | stops or plays this camera's video on this machine, so a computer that cannot decode every video plays the others smoothly; the overlay stays |
| **Turn 180°** | turns the video and the overlay of an upside-down camera whose capture did not turn the picture; off by default, and absent on a camera whose capture turned it |
| **Sync overlay** | holds a VFA camera's live video back by the overlay's lag, up to 4 s; on by default, and shown only while a VFA camera's live video plays (an IPS camera's video plays as it comes) |
| **Play** | on a tile past the limit of cameras that play at once, takes another tile's place |

**Hide video** and **Turn 180°** are kept per session and camera in this browser only.

??? info "Details: which tile is which camera"
    - For each base, the session's MongoDB document lists the stream it took (`sources[].stream`, `server_path`) and how the capture host turned the picture (`capture.rotate`). The session's metadata makes one `video` item per camera from them, keyed by the stream's name, else by the VFA base id, else by the IPS one; the bases of a file-mode session that read the same file share an item. A VFA camera of the frame sets that the document does not name gets a tile too.
    - A tile draws the frame sets of its VFA base id. It plays the live stream of one of its stream paths, else of its stream's name, else (when the document names no paths) of its base id.
    - In a replay it plays an archived stream cut of one of its stream paths, or a Collection file whose device is the camera's name or one of its base ids: a cut of `vfa/cam-1` plays on the `cam-1 · VFA` tile, a Collection file of `cam-1` on the `cam-1` tile.

??? info "Details: Hide video and Turn 180°"
    - A hidden VFA camera keeps its skeletons on a blank stage; a hidden IPS camera folds to its heading. Its connection goes to a camera waiting for a place, which plays at once.
    - **Turn 180°** is offered when `capture.rotate` is 0 or the session's document has no such field; quarter turns are not offered. It is a CSS rotation of the video, and the overlay is drawn turned the same way: x becomes w − x and y becomes h − y of the frame set, for boxes, labels and hover regions. For new sessions, turn the picture at capture instead ([Turning the picture](../pipelines/vfa/configuration.md#turning-the-picture)).
    - Both choices live in the browser's local storage, so another machine opens with every camera playing and upright; a private window keeps them for the page alone.

## Live video

Live video plays under a tile's overlay when all of these hold:

1. The session's bases pulled their cameras from MediaMTX (a `stream` source). A session that read files, a local camera or a raw UDP stream has no live video.
2. The stream is publishing now: MediaMTX's API lists the path as ready.
3. MediaMTX has WebRTC on and the browser reaches it on TCP 8889 and on UDP 8189 (or TCP 8189 when UDP is blocked); see [Live video and sound in MediaMTX](deploy.md#live-video-and-sound-in-mediamtx).

The browser plays the stream straight from MediaMTX over WebRTC (WHEP, `POST http://<stream-server>:8889/<path>/whep`); the dashboard only tells it which paths are live. A tile connects only while the Cameras card is open, the tile is on screen and the tab is visible, and at most six tiles play at once (`Up to 6 cameras play at once` on the others). The video plays muted.

The overlay arrives one to two seconds after the video, after inference and the synchronizer. It updates once per frame set (once a second at the default `keyframe_interval`), so it steps while the video runs smoothly. With **Sync overlay** on, a browser that supports it holds a VFA camera's video back by that lag, and the tile says by how much.

??? info "Details: what live video costs"
    - MediaMTX forwards the H.264 packets it receives without transcoding, so each tile that plays is one WebRTC session carrying the camera's own bitrate to that browser: four cameras at 0.8 Mbit/s are about 3.2 Mbit/s per viewer.
    - The camera's upload to the server is the same whether nobody or ten browsers watch. The bases pull their own copy over RTSP and are not affected.
    - A closed card or a hidden tab drops its connections, so a dashboard nobody looks at pulls no video.

## Recorded video in a replay

In the replay of an ended session, each tile plays its camera's recorded file from the dashboard's own machine: a Collection file or an archived stream cut ([Raw recordings](#raw-recordings)). The video follows the replay clock: it plays at the replay speed, pauses with the replay, and shows the new moment after a jump. The overlay is drawn at the video's own time, so the skeletons stay on the picture, and before a file starts or after it ends the tile says `No recording at this moment`.

The page asks for the list of files when the replay opens and each time the Cameras card is opened, so files copied to the machine later play too. With the default two media ports, up to nine recorded cameras play at once (`Up to 9 cameras play at once` on the others).

??? info "Details: how the video follows the replay clock"
    - The video's position is the clock minus the file's start. It seeks when it is more than 0.5 s off the clock, or a quarter second at 4x and faster, where the clock itself steps that much between the stream's batches. A smaller gap closes by playing a tenth slower or faster.
    - The overlay is mapped onto the area the picture fills in the tile, letterboxed when the file's shape differs from the frame set's.
    - A file ends where the list says, never at the length the browser reads from it, since Firefox reads an archived cut (a fragmented MP4 whose header names no length) one fragment at a time. The list's length is the manifest's, else what `ffprobe` reads on the dashboard's machine, else the recorder's own start and stop times, so a file the manifest gives no length for, such as a Collection file before `mmla ses-tidy`, still ends where it ends.
    - A file that starts within a few seconds loads ahead, so it appears on time. A file loads from the moment the clock asks for (a media fragment, `#t=`), and until the video stands at that moment the tile shows the skeletons alone, never the file's first frame or an old picture.
    - A browser that refuses a speed shows a still picture that steps once a second, and the tile says so.
    - A tile loads its file only while the card is open, the tile is on screen and the tab is visible. A tile scrolled off screen keeps its file, paused, while its media port has room; the tile seen least recently lets go first. The browser keeps a fetched recording for a day.

??? info "Details: how many recorded cameras play at once"
    A browser opens at most six HTTP/1.1 connections per origin (scheme, host and port), and every recording it loads holds one, even paused. The page therefore loads the recordings from the dashboard's **media ports** ([Ports](deploy.md#ports)), which are origins of their own, and leaves the dashboard's port its six connections for the page's requests and its data stream.

    | Media ports that answer | Recorded cameras that play at once | Message on the other tiles |
    |---|---|---|
    | two (the default) | up to 4 on the first port beside the sound and up to 5 on the second: 9 | `Up to 9 cameras play at once` |
    | one | 4 beside the sound | `Up to 4 cameras play at once` |
    | none | 3 beside the sound; 4 with the sound `Off` or above 4x | `Up to 3 cameras play beside the sound` |

    - Each media port keeps one of its six connections free for a seek. The sound's place on the first port stays free while the sound is `Off`, so no video moves when it comes on.
    - The cameras, in the order of their names, are dealt over the ports that answer, so six cameras play three from each. A camera loads from its own port while it has room, else from the port with the most room left, and keeps that port while it holds its file, so another tile letting go of a file reloads nothing. As the browser keeps a fetched file by origin, a camera scrolled back to, or a card opened again, finds its file in the cache of its own port.
    - Before it uses the media ports, the page asks each one `/api/media-origin`, all at once, waiting up to 3 s. The answer has to name that port and carry the `media_instance` the recordings route gave the page, so another dashboard on that port number is not taken for this one. A port that does not answer is left out, and asked again when the Cameras card is opened (at most once a minute) or on a reload. The sound goes to the first port that answers.
    - With no media port answering (a dashboard started without them, a firewall or tailnet rule that lets only the dashboard's port through, a proxy or tunnel to that port alone, or another dashboard answering on that port number), the files load from the page's own origin, at most four at once with the sound counting as one while it plays, so one connection stays free for the page's requests.

## Sound

The **Sound** control beside the speed buttons plays one microphone at a time, and the speaker button beside it mutes it. The camera videos always play muted.

| Page | What the control plays |
|---|---|
| Replay of an ended session | a microphone recording on the dashboard's machine; starts on the group microphone ([Sound in a replay](#sound-in-a-replay)) |
| Follow mode | a microphone of the running session, live; starts on `Off` ([Live sound](#live-sound)) |
| Replay of a running session | nothing: the tooltip says whether **Back to live** hears the microphones |
| Raw recordings turned off | nothing, live or recorded: the control is disabled with that reason |

### Sound in a replay

The control lists `Off` and each microphone recording the dashboard's machine holds: `Group mic jabra-1`, `Worn mic lapel-1, Tag 0`, or `Stream asr/jabra-1` for an archived cut. It starts on the group microphone (the longest when there are several), else on the first one listed. The sound follows the replay clock as the video does, and is silent where its file does not cover the moment (`No recording at this moment`).

Above 4x the sound is muted (`Muted above 4x`): speech that fast cannot be understood, and browsers silence it there anyway. Nothing loads before your first click on the page's controls, so the browser's autoplay rules do not hold the sound back; a browser that holds it back all the same shows **Play the sound**, which plays it in one click.

??? info "Details: replay sound"
    - An archived cut is listed apart from a capture file of the same microphone. A microphone whose recorder was restarted plays whichever of its files holds the moment. The control shows `Off` when the machine holds no microphone file.
    - The first click that loads the sound is on **Play**, **Pause**, **Replay from start**, the speaker button (to unmute), or a pick in the control.
    - At 4x, the fastest it plays, the sound cannot close a lag by playing faster: when it is more than half a second behind the clock, it seeks a little ahead, at most once a second. Above 4x, and in a hidden tab, it lets go of its file.
    - The file loads from the dashboard's first media port, beside the cameras' videos; when no media port answers, it is one of the four recordings the page loads at once from its own port.

### Live sound

In follow mode, pick a microphone in the control to hear it live; it starts on `Off`, and the pick is what lets the browser play it. The list holds each microphone the session takes through the Stream Server, by its stream name. The sound arrives a moment after it is spoken, well ahead of its transcript, and goes on in a hidden tab.

| Note beside the control | Meaning |
|---|---|
| `Connecting` | the page is opening the microphone's sound |
| `Live` | the microphone plays |
| `Not publishing now` | the microphone's stream was not publishing when the page last asked the Stream Server, so the page does not ask for its sound |
| `Reconnecting` | the Stream Server refused the request (its answer is in the tooltip), the connection dropped or failed (`The sound connection dropped.`, `The sound connection failed.`), or the microphone stopped; the page tries again by itself |
| `Could not play` | the browser has no WebRTC; the tooltip gives the reason |

??? info "Details: when live sound ends"
    - The page ends live sound, in a hidden tab too, when you pick `Off` or start a replay, when the session ends, when the page leaves follow mode, or when the microphone is no longer offered.
    - The page asks the dashboard's `/media` once a minute, also in a hidden tab, so turning the raw recordings off, or a stream that stopped, reaches it within about a minute. A microphone that stops between two asks reads `Reconnecting` until the next one marks it `Not publishing now`.
    - When the session ends while the page follows it, the control is disabled: its microphones are no longer the session's, because a stream outlives the sessions that take it. The tooltip says whether **Replay this session** plays its recorded microphones, or whether the dashboard's machine holds none yet ([Get the files onto the dashboard's machine](#get-the-files-onto-the-dashboards-machine)).

??? info "Details: how live sound works"
    - The microphones push AAC (the capture command's `-c:a aac`; an RTMP push from the capture hosts cannot carry Opus), which the ASR bases pull. The WebRTC of MediaMTX carries Opus, G.722, G.711 and LPCM audio, not AAC, and MediaMTX transcodes nothing, so a WHEP request for a microphone's own path is refused (`the stream doesn't contain any supported codec`).
    - `mediamtx.yml` therefore holds a path entry `~^listen/(.+)$` with a `runOnDemand` command. Reading `listen/<app>/<name>` starts an FFmpeg inside MediaMTX that reads `<app>/<name>` over RTSP and publishes its first audio track on `listen/<app>/<name>`, as Opus at 32 kbit/s and 48 kHz in one channel (`-ac 1`: a stereo microphone is mixed down). The page reads `listen/asr/jabra-1` for the microphone `asr/jabra-1`.
    - The FFmpeg is up in about half a second, and stops 10 s after its last reader left (`runOnDemandCloseAfter`). For a microphone that is not publishing, the request fails after 10 s (`source of path 'listen/...' has timed out`).
    - Nothing of a `listen/` path is recorded (`record: no`), and it is never a session's source, so START never switches it on.
    - The Docker service runs `bluenviron/mediamtx:1.21.0-ffmpeg`, the MediaMTX image that carries FFmpeg with libopus. A native MediaMTX (`make mediamtx`) needs an `ffmpeg` with libopus on the PATH: `brew install ffmpeg` on macOS, `sudo apt install -y ffmpeg` on Debian or Ubuntu.
    - The dashboard offers the microphones only while the running MediaMTX has WebRTC on and holds that entry; it asks `/v3/config/paths/list` at most once a minute. Otherwise the control is disabled in follow mode, and its tooltip says why.
    - What it costs: one FFmpeg on the Stream Server per microphone someone listens to, and about 32 kbit/s of Opus to each listener (some 50 kbit/s on the wire). The capture hosts and the bases are not affected.

## Raw recordings

The **Downloads** card of **Data and exports** on the Analysis page lists the session's recordings that the dashboard's machine holds, and the Live page's replay plays the same files.

!!! warning
    The files go out over plain HTTP without a login: anyone who reaches port 5050, or the media ports 5051 and 5052, can download a session's footage and the participants' voices. See [Keep the dashboard private](deploy.md#keep-the-dashboard-private).

The card starts with a line on the session's archive (`mmla ses-archive`): `Archive: complete, 32 files, 99 MB, verified 15 Jan 2026 10:05, on this machine`, or `on <host>:<path>` when the archive is on another machine. An incomplete archive reads `partial` with a warning mark, and a session without one `Not archived`.

Then come the session's files in two lists. Each row has the file's format, length and size, a **Download** link, and a **Play** link that opens the file in a new tab.

| List | What it holds | Labels |
|---|---|---|
| **Collection recordings** | the camera and microphone files of the Collection card's recorders | `Camera cam-1 on pi-01`, `Group mic jabra-1`, `Worn mic lapel-1, Tag 0` |
| **Stream cuts (archived)** | the fMP4 cuts that `mmla ses-archive` or the console's **Sessions → Export** took of each stream from the Stream Server | `Stream vfa/cam-1`, `Stream asr/jabra-1` |

A live session whose capture hosts recorded nothing has only the stream cuts, and the card says so.

??? info "Details: which files are listed"
    - The archive line comes from the session's MongoDB document, else from the manifest of the session folder on the dashboard's machine. The page asks for the list once, when the section is first shown.
    - **Collection files**: the files under `artifacts/<session>/collection/<host>/<audio|video>/`, as the Collection card's recorders, its **Download** and `mmla ses-import` leave them, in the formats wav, flac, m4a, mp3, mp4, mov, mkv and webm.
    - `artifacts/<session>/manifest.json` says which files they are, with each one's device, start and length and, for a microphone, whether it is the group's or worn, and by which tag. A manifest's `path` was written on the machine that recorded or imported the session; when it does not lead to the file here, the file is looked up by its place in `collection/`. What `collection/` holds besides the manifest's rows is listed too, and a manifest whose `recordings` list is empty lists them all.
    - **Stream cuts**: the files `artifacts/<session>/streams/server/<stream path>_<start>.<ext>`, found by the manifest's `stream_cuts` rows (their `relpath`, or a `path` inside the session folder; the file has to carry the row's id) and by a scan of `streams/server/` for those the manifest does not list.

!!! note "Never served"
    Whatever a manifest or a link says, the dashboard never lists or serves:

    - the speaker profiles (`profiles/`, the participants' voice biometrics, wherever they sit in the session folder);
    - the coding clips under `labels/`, and `raw/`, `analysis/` and `pipelines/`;
    - the manifests, the archive's ledger under `.archive/`, and any folder whose name starts with a dot;
    - everything under `streams/` but the Stream Server's cuts in `streams/server/` (the capture hosts' cuts under `streams/capture/` included);
    - any file that a symlink or a manifest path leads to outside the session's folder.

    Every file is checked on its real path, and a recording id that is not on the list answers 404.

### Get the files onto the dashboard's machine

The card lists only what the dashboard's machine holds under `artifacts/<session>/` (or `DASHBOARD_ARTIFACTS_DIR`). A session recorded or imported on another machine has nothing to list until its files are copied over:

- On the console's **Sessions** tab, **Export** gathers the session onto the console, and **Archive** sends it to the System Settings host. Archive sends only the console's own copy, and refuses a session with no folder there ([Archive](../tui/sessions.md#archive)).
- The Collection card's **Download** fetches the recorders' files into the `artifacts/` of the machine the console runs on.

### Stream recordings on the Stream Server

When the session's sources went through the Stream Server, the card also lists what MediaMTX recorded of each stream during the session: one row per unbroken stretch, linked to MediaMTX's playback server (`/get`, an fMP4 file on port 9996, `playback_port`), from which the browser fetches it. MediaMTX deletes a recording `recordDeleteAfter` after it began (72 h in `mediamtx.yml`); **Archive** and **Sessions → Export** keep a session's part for good, as stream cuts.

??? info "Details: stream recordings and archived cuts"
    - A stream restarted during the session gives two stretches. Stretches shorter than 1 s are left out.
    - A stretch that an archived cut on this machine holds (a cut of the same path that starts within 2 s of it, or that it lies inside) is listed once, as the cut. With cuts present, this section is titled `Stream recordings not archived`.
    - The list needs MongoDB (which streams the session used) and MediaMTX's API. Without them it is empty, and the files are listed all the same.

## Troubleshooting

**A camera tile says `Skeleton view`.** The tile names the reason. When the session should have live video, check the three conditions under [Live video](#live-video): the browser's developer console shows a failed `whep` request when 8889 is not reachable or WebRTC is off, and a connection that never starts when 8189 is not reachable or `MEDIAMTX_WEBRTC_HOSTS` is unset under Docker.

**Fewer recorded cameras play than expected.** `Up to 4 cameras play at once` means one media port answered, and `Up to 3 cameras play beside the sound` none. Check that the dashboard was restarted with the current Makefile (`curl http://<dashboard-host>:5052/api/media-origin` answers `"media_ports":[5051,5052]` and the same `media_instance` as port 5050), and that the browser's machine reaches ports 5051 and 5052 past any tailnet ACL or firewall. Then open the Cameras card again a minute later, or reload the page.

**No skeletons or gaze.** The VFA synchronizer ran without **Pose** or **Gaze**, so the session has no `vfa_features`.

**No sound in follow mode.** The control's tooltip says why: no microphone of the session went through the Stream Server, the Stream Server has WebRTC off or no `listen/` entry, the raw recordings are turned off, or the session has ended. A missing `listen/` entry means the running MediaMTX started from a `mediamtx.yml` without it: recreate it between sessions ([Live video and sound in MediaMTX](deploy.md#live-video-and-sound-in-mediamtx)).

**Live sound stays at `Reconnecting`.** `source of path 'listen/...' has timed out` in the tooltip means the FFmpeg that makes the sound could not read the microphone's stream (it stopped) or could not run: a container on the plain `bluenviron/mediamtx` image has no `ffmpeg`, and `docker logs openmmla-infra-mediamtx-1` shows the command failing. A connection that never starts has the same causes as for video: 8189 is not reachable, or `MEDIAMTX_WEBRTC_HOSTS` is unset under Docker.

**No sound in a replay.** The session is still running, the raw recordings are turned off, or the dashboard's machine holds no microphone file of the session ([Get the files onto the dashboard's machine](#get-the-files-onto-the-dashboards-machine)). A file the browser cannot play says `Could not play`.

**The recorded video and the sound of a replay stop for a moment.** The replay's clock waits for InfluxDB, a slow link or a slow query ([Replay mode](reference.md#replay-mode)). They go on when it does.

**No recordings in the downloads.** The dashboard's machine holds neither `artifacts/<session>/collection/` nor the archive's `streams/server/` for the session: copy them over ([Get the files onto the dashboard's machine](#get-the-files-onto-the-dashboards-machine)). The archive line says where the archive is when it is on another machine.
