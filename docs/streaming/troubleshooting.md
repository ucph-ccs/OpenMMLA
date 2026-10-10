# Streaming troubleshooting

What to do when a stream does not start, does not reach the bases, drops, or is not recorded. Each entry starts with what you see.

## The Stream Server

**MediaMTX does not start, or its log says `port 1935 answers but 8554 does not`.** Another program holds port 1935, such as an Nginx whose config still has an `rtmp { ... }` block. Press **Start** on the **Gateway (Nginx)** card, which renders the config again from the current template, with no RTMP block, and reloads Nginx. Then start MediaMTX.

**Nothing in `/v3/paths/list` while the capture host runs FFmpeg.** The host does not reach the Stream Server on port 1935. Check the host in the stream's `target` URL and `StreamServer.host` in System Settings, which the MediaMTX card uses, and the server's firewall (on macOS, **System Settings → Privacy & Security → Firewall**).

**The server runs with old settings.** A container reads `mediamtx.yml` only when it starts. `curl -s http://<stream-server>:9997/v3/config/global/get` shows what it runs with; recreate it ([Server settings](index.md#configuration)).

## Streams tab

**Status says `Exited`.** FFmpeg stopped by itself; a push the server drops does not do that. **Start** quoted its last lines, and **Logs** shows the whole pane. The usual causes are a wrong device name (`v4l2-ctl --list-devices`, `arecord -l`; on a Mac `ffmpeg -f avfoundation -list_devices true -i ""`), or a resolution or frame rate the camera cannot deliver.

**A Mac's stream never starts** (`Could not open a Terminal window`, or `did not start ffmpeg`). Nobody is logged in on the Mac's screen, which is where macOS lets FFmpeg use the camera ([Capture on a Mac](index.md#capture-on-a-mac)).

**A row reads `Name clash` or `Name in use`.** Entries of two pipelines give one name to two different captures ([Stream names](../tui/launcher/pipelines/streams.md#stream-names)). Rename one of them on its card's **Config** tab; its target can stay.

**The Rotate cell reads `180° (runs 0°)` in yellow.** The config was given another turn after the stream started. Stop and start the stream before the bases, which read the config's turn ([Turning the picture](../tui/launcher/pipelines/streams.md#turning-the-picture)).

**`Running`, but the Stream Server column reads `○ not live`.** Nothing arrives. The push cannot reach the server and keeps trying: **Logs** shows what FFmpeg says, and a Mac may be asking on its screen whether Terminal may use the camera. **Probe** decodes two seconds of the read URL with FFmpeg on this machine; its error text is the server's answer, and `404 Not Found` means nobody publishes that path.

## Bases

**A base card's Start says a stream is not live.** The Stream Server receives nothing on a path the bases would pull. Look at the stream's row on the Streams tab: `Exited` needs **Start** (**Logs** says why it stopped), and `Running` with `○ not live` is a push that cannot reach the server. Press **Start** on the base card again once the stream reads `● live`, or press it again at once to start the bases all the same, each waiting `connect_wait` seconds for its stream.

**A base says `Stream <url> is not up: trying again every 2 s for up to 30 s.`, then `did not come up within 30 s`, and exits.** Its stream did not come up within `connect_wait`. Start the stream on the Streams tab, then the bases again.

**Grey or torn frames.** The reader chose UDP transport. The bases use TCP; keep `rtsp_transport;tcp` in `stream_kwargs.capture_options`.

**The delay grows over the session.** The RTMP push over Wi-Fi is buffering. Publish with SRT instead ([FFmpeg commands](index.md#ffmpeg-commands)), or move the capture host to Ethernet.

**A stream dropped during a session.** A video base says `dropped: its reads failed` (its connection was ended) or `dropped: no frame for 10 s` (the stream stalled). An ASR base says `Audio stream <url> ended` or `Audio stream <url> stalled: no audio came for 10 s`, and finishes the transcription chunk it had open at the gap, so no chunk holds speech from both sides of it. MediaMTX's log says `closed: read tcp ...: i/o timeout`:

```bash
docker compose -f docker/docker-compose.infra.yml logs --tail 50 mediamtx
```

Nothing arrived from the publisher for longer than `readTimeout`, 30 s as shipped, so the server closed it. Its FFmpeg publishes again by itself every five seconds, the **Stream Server** column reads `● live` once it is back, and the bases open their streams again and say `is back after N s`. The frames of the gap are missing.

??? info "Details: stalls through a Tailscale relay"
    A capture host that reaches the server through a Tailscale relay sees such stalls: `tailscale ping <stream-server>` on it answers `via DERP(...)` and `direct connection not established`, as when the server sits behind a NAT that gives no direct path. A direct path avoids them: UDP 41641 let through to the server, the server on a public address, or the capture hosts and the server on one LAN, publishing to its LAN address.

**A base says `did not come back`.** Its stream stayed away longer than `reconnect_wait`, and the base ended its run. The stream's **Status** on the Streams tab tells why. Start that base alone again, then send START for the session again: the components already running ignore a second START, and Session Control adds the stream's path to the session's open recording window.

- On its card: set **Num Bases** to 1, with that base's `Bases` entry, and the synchronizers to 0. The card's **Start** opens every instance it counts, and the others still run.
- Or on its host: `mmla ips-base -sid <session> -b <entry>` (`vfa-base`, `asr-base`).

## Recordings

**The server recorded nothing of a session after a restart.** The recording switch lives in the running server. Send START for the session again ([Record on the server](recording.md#on-the-server)).

**The Stream Server card's Recordings tab is empty.** The tab says which of three cases it is, from what the server records (its path defaults and path entries) and what is published to it:

- Nothing is recorded now: with the shipped `record: no`, no running session takes a published path.
- Nothing is published.
- A path it records is publishing right now with nothing written for it, shown in red. The recorder cannot write: usually a container whose record folder was renamed or removed under it, which keeps the old folder open and can create nothing in it. Press **Stop** and then **Start** on the card to make a new container; its log says so in as many words.

**A capture host's recording ends while the stream still runs.** The file could not be written, usually a full disk; FFmpeg drops the file and goes on streaming. **Manage** shows the room left ([Manage the recordings](recording.md#manage-the-recordings)).

**A recording has no length or cannot be seeked.** FFmpeg was killed before it finished the file. It still plays; `ffmpeg -i <file>.mkv -c copy <fixed>.mkv` writes the seek index and the duration.

**Export finds no streams for a session.** No base of it noted a stream in its `sources`, and Export takes nothing on a guess ([Export a session's part](recording.md#a-sessions-part-sessions-export)).

**No live video or sound on the dashboard.** See [Live video and sound in MediaMTX](../dashboard/deploy.md#live-video-and-sound-in-mediamtx).
