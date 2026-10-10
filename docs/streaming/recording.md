# Recording

A stream is recorded in up to two places: the Stream Server records the streams a session's bases pull while the session runs, and a stream with `record: true` also records on its capture host, session or not. **Sessions → Export** takes a session's part of both afterwards.

## Where the files are

```
# on each capture host (record_root: ~/artifacts on a remote host, artifacts/ of the project for local)
<record_root>/streams/capture/<YYYY-MM-DD>/<host label>/video/<name>_<start time>.mkv
<record_root>/streams/capture/<YYYY-MM-DD>/<host label>/audio/<name>_<start time>.wav

# on the Stream Server's host, MediaMTX's own segments
artifacts/streams/server/<app>/<name>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4

# on the console, a session's part (Sessions → Export)
artifacts/<session>/streams/server/<app>/<name>_<start time>.mp4
artifacts/<session>/streams/capture/<host label>/video/<name>_<start time>.mkv
artifacts/<session>/streams/capture/<host label>/audio/<name>_<start time>.wav
```

`artifacts/streams/` holds no session, so the **Sessions** tab does not list it. `artifacts/<session>/collection/<host>/` holds the Collection card's own recorders only.

A recording is filed by day, not under a session, because a stream belongs to no session: it is started once, with nothing to choose, and any number of sessions pull it, one after another or at the same time, such as several groups in one room sharing a camera. A stream is not restarted between sessions. What ties a recording to a session is the session's record: each base notes in the session's MongoDB document which stream it takes, when it joins and when it leaves (its `sources`, see [Databases](../database.md#mongodb)), the session's `start_time` and `end_time` are there too, and the file names carry the start times.

## Record on the capture host { #on-the-capture-device }

1. **Switch it on.** Set a stream's **Record** cell on the **Streams** tab to `yes` (`record: true`), and **Start** the stream. A running stream records from its next **Start**.
2. **Let it run.** The same FFmpeg writes the stream to a file while it pushes it, one file from **Start** to **Stop**, however long it runs.
3. **Stop** the stream. **Stop** sends Ctrl-C, waits for FFmpeg to finish the file, and names it.

The recording does not depend on the network or the server: it goes on in one file when the push drops, while the push recovers ([How a dropped push recovers](index.md#start-and-stop-streams)). It shares the stream's `bitrate`, about 450 MB an hour or 11 GB a day at the default `1M`, and costs the capture host next to nothing beyond the encode the push already makes.

| Part of the path | What it is |
|---|---|
| `<record_root>` | the stream's `record_root`: `~/artifacts` on a remote host, `artifacts/` of the project for `local` |
| `<YYYY-MM-DD>` | the day the stream was started, by the console's clock at **Start** |
| `<host label>` | the SSH profile's name; this machine's short host name for `local` |
| `video/`, `audio/` | the stream's `kind`, as its card reads it; `.mkv` for video, `.wav` for audio |
| `<start time>` | the unix time the stream started, which a `file` source reads back |

!!! note "Streamed or recorded by the Collection card"
    A camera or a microphone is either streamed or recorded by the Collection card, not both at once. `record: true` is the way to get both. The Collection card's own **Download** fetches one session of its recorders, which a session starts and ends.

??? info "Details: how the file is written"
    - Video is one H.264 encode with two outputs through FFmpeg's tee muxer, the file first. Audio is one capture split into an AAC stream for the push and a PCM stream for the file, through a tee the same way.
    - FFmpeg appends to the file a fraction of a second at a time. **Stop** adds its end: the seek index and the duration. A file that never got them (FFmpeg killed, a power cut) still plays, and `ffmpeg -i <file>.mkv -c copy <fixed>.mkv` writes them.
    - The push does not wait on the file either. When the file cannot be written, a full disk for one, FFmpeg 7 drops that output and goes on streaming, so the recording ends while **Status** still says `Running`. **Manage** shows the room left on each host.

### Manage the recordings

**Manage**, in the **Recordings** row of the Streams tab, lists what the card's streams recorded on each capture host: newest day first, with each file's size, the room left on the host's disk, and the file a stream is writing now. The Stream Server card's **Streams** tab has one over the streams of every pipeline ([Stream Server Streams tab](../tui/launcher/system-services.md#stream-server-streams-tab)). Manage copies nothing to this machine; a session's part comes with **Sessions → Export** ([Export a session's part](#a-sessions-part-sessions-export)).

| Control | What it deletes on the capture hosts |
|---|---|
| **Delete File** | the selected recording |
| **Delete Day** | every recording of the selected recording's day (the **Day** column), on its host |
| **Delete Older Than** | every recording last written longer ago than the age beside it, on every host |
| **Keep recordings for** | sets `record_keep_days` of every stream of the card: a recording last written longer ago is deleted at its stream's next **Start** and at **Refresh**, which say what they removed; `0` keeps them; the **Record** column shows it (`yes, 7 d`) |
| **Delete Expired** | what **Keep recordings for** would delete, at once |

Each delete takes a second press: the first says what would go and how much room it frees, and **Delete Day** names the day and the host. Nothing else deletes these files.

!!! warning
    Manage never deletes the file a stream is writing, by hand or by the keep time: a file deleted while it is written keeps its room until the stream stops, and that take is lost. Stop the stream first.

??? info "Details: folders and exported copies"
    - The folders a deletion leaves empty go with it, except today's. An empty folder of an earlier day goes whenever Manage lists the host, so one left today goes at the first listing after today.
    - Deleting on a capture host leaves what **Sessions → Export** copied to the console alone. A session's part, under `artifacts/<session>/streams/`, goes with that session's **Delete Files**, on the machine the Sessions tab's **Host** selector picks.

## Record on the server { #on-the-server }

The Stream Server records a stream's path while a session that pulls it runs, in fMP4 segments of at most ten minutes, to `artifacts/streams/server/<app>/<name>/<YYYY-MM-DD_HH-MM-SS-ffffff>.mp4` on its host.

1. **Start the bases.** A base notes its stream in the session's `sources` when it starts, before it waits for START.
2. **Send START** in Session Control (or `mmla ses-ctl`). It switches recording on, one path at a time through the control API, for the paths the session's bases noted and took through this server; a stream pulled from another server or straight from a camera is no path of it.
3. **Send STOP.** It switches them off again, as do **End Session** on the Sessions tab and the Collection card's **Stop** when it ends the session.

The session's MongoDB document keeps the times under `recording_windows` (`paths`, `start`, `end`). Two sessions in two rooms switch only their own paths, and a path that two running sessions share stays on until the last of them stops. A base started after START waits for the next START, which switches its stream on too.

The status line after a signal names the paths, and says in yellow what could not be done: no Stream Server under System Settings, or a server or MongoDB that does not answer. The signal goes out either way.

!!! warning "The switch lives in the running server"
    A server restarted during a session records nothing more of it until START is sent again. So does a server that reads its `mediamtx.yml` again, which drops every path the API switched on: a native MediaMTX reloads the file whenever it changes, and a container reads it when it starts. A few seconds after a **Save** or a **Sync** on the card's **Config** tab, the console switches the paths of the running sessions on again and says so; after anything else, send START again.

A segment is deleted `recordDeleteAfter` after it began, by MediaMTX itself: three days in the shipped file, set by **Keep recordings for** on the card's **Config** tab (`0s` keeps everything). Export a session's footage before then; the **Sessions** table says until when, in its **Recordings until** column.

With **Server-side recording** on `every path` (the card's **Config** tab, `pathDefaults.record: yes`), the server records every published path, sessions or not. The card's **Recordings** tab shows what the server holds, path by path, and deletes a path or every segment older than an age ([Stream Server Recordings tab](../tui/launcher/system-services.md#stream-server-recordings-tab)).

Segment names are the server's clock, or the publisher's for a path with `useAbsoluteTimestamp` on ([Timestamps](index.md#timestamps)). Treat them as an archive of what arrived rather than as a capture-side recording.

### Fetch a time range by hand

The playback server returns any time range of a path as one file:

```bash
curl -o front.mp4 "http://<stream-server>:9996/get?path=vfa/front&start=2026-09-14T10:00:00Z&duration=600"
```

The file's clock starts at the requested start, so the start plus a frame's time is the server's wall clock again. To replay the file through a pipeline, name it `<prefix>_<unix start time>.mp4` and use it as a `file` source. An audio path comes out as `.mp4` with an AAC track: convert it before replaying it through ASR:

```bash
ffmpeg -i mic1_<start>.mp4 mic1_<start>.wav
```

## Export a session's part { #a-sessions-part-sessions-export }

Press **Export** on the **Sessions** tab, or run `mmla ses-export <session> --only streams`. It takes the session's part of every stream the session used, both copies, with nothing to fill in, into `artifacts/<session>/streams/`.

| Copy | Taken | Lands in |
|---|---|---|
| the Stream Server's | over HTTP from the playback server, for each source whose URL names the Stream Server of System Settings, by its path exactly as named, so another group's camera recorded at the same time stays out | `streams/server/<app>/<name>_<start>.mp4`, one file per unbroken stretch |
| the capture host's | over SSH, for each stream the console captures: cut on its capture host with the FFmpeg that recorded it, without re-encoding, and fetched like a Collection download, resumable | `streams/capture/<host label>/video/<name>_<start>.mkv`, or `audio/` and `.wav` |

The part is each stretch from a START to its STOP, as the session's `recording_windows` note, and both copies are cut to the same stretches. What the capture hosts recorded before the first START, while the bases started, is no part of it. A video cut begins on the last keyframe before the stretch's start (the recorder writes one a second) and is named after that frame; audio is cut to the sample.

Export names the folders in the session's `manifest.yml` as file sources (audio for ASR, video for IPS and VFA), so a base replays either copy with `source: file` and the file's full path as `source_index`. The recordings themselves are never touched, so every session that shared a stream can take its own part.

!!! note
    The capture host's clock decides what is cut there: keep it on NTP ([Keep the clocks in sync](index.md#keep-the-clocks-in-sync)).

??? info "Details: which stretches and which streams"
    - Export reads the session's record from MongoDB again at the press, since a base can join after the table was listed, and goes by its `sources`: which stream each base took, the URL it pulled, and the machine that captures and records it ([Databases](../database.md#mongodb)).
    - A path is cut only to the stretches whose START switched it on; a stream that went through no server, or that no START switched on, to all of them.
    - A session without `recording_windows` (one never STARTed) takes the time between its `start_time` and its `end_time`; a session never ended, until its last base left; while a base is still in it, up to now. The log says which it went by.
    - A source names the Stream Server by its name, an alias or its address, or as `localhost` from a base that ran on the server's host. A stream published to another server is named in the log and skipped. Audio pushed straight to an ASR base (`udp://`, `tcp://`) goes through no server, so only its capture host can have it.
    - The capture host is asked whatever **Record** the session's bases noted, since a base notes it from the config of the machine it runs on, which can differ from the one the stream was started with. When the bases noted it differently, the one that noted it on says where it is recorded (capture host and `record_root`). A stream noted with Record off that its host recorded all the same is cut like the others, and the log says so.
    - A stream its capture host holds nothing of from that time, or whose host does not answer, is named in the log with where its only copy is (on the Stream Server), and is not counted as missing. So is a stream someone else publishes (no SSH profile).
    - The cuts are staged under `<record_root>/streams/.session-cuts/<session>/<host label>/` on the capture host, and the staging is removed once they have arrived. A stream captured on this machine is cut straight into place.
    - A session with no record of its streams (no base joined it) has nothing to export, and the log says so. Export takes nothing on a guess: the server records the streams of every running session, and the capture hosts whatever runs.

??? info "Details: exporting again and cancelling"
    - A file already here in full is not fetched or cut again. One exported while the session was still going has the same name but is shorter: its length is checked with `ffprobe`, and it is replaced by the full one. Without `ffprobe` on this machine, a Stream Server clip of an ended session counts as final.
    - While the export runs, a progress row under the button shows what it does. Its **Cancel** stops it between steps, and a clip being downloaded at once. What arrived stays, and cuts made on a capture host stay there until the next press fetches them.
    - Export takes the streams after the measurements and the Collection recordings, and before the bases' files.
