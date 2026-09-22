# Indoor Positioning System (IPS)

Camera-based indoor positioning with AprilTags. Several cameras watch the room, each IPS base detects the tags worn by the participants in its camera's view, the coordinates are transformed into one shared frame, and the synchronizer writes positions, orientations and the proximity graph of the group to InfluxDB.

## Pipeline overview

![Indoor positioning system](../img/indoor_positioning_system.png)

1. Video capture from one camera per base: a USB camera, an RTMP stream, LSL, or a recorded file
2. AprilTag detection and pose estimation with the calibrated camera intrinsics
3. Transformation of every camera's coordinates into the main camera's frame, using the matrices produced by camera sync
4. Synchronization of all bases into time buckets, written as `ips_translation`, `ips_rotation` and `ips_relation` events
5. Optional real-time visualization of the positions

| Component | Runs on | Command | Environment |
|---|---|---|---|
| IPS Base, one per camera | base station | `mmla ips-base` | conda env `ips-base` |
| IPS Synchronizer, one per session | base station | `mmla ips-sync` | conda env `ips-base` |
| IPS Visualizer, optional | base station | `mmla ips-vis` | conda env `ips-base` |
| Camera calibrator, once per camera model | any machine with the camera | `mmla ips-ccal` | conda env `ips-base` |
| Camera tag detector and sync manager, once per camera arrangement | base stations | `mmla ips-ctag`, `mmla ips-csync` | conda env `ips-base` |

IPS needs no AI server. Create the `ips-base` environment from the TUI's Environment tab or by hand (`conda create -n ips-base python=3.10 -y && pip install -e '.[ips-base]'`). For an `lsl` source add `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.

An alternative input path skips the cameras entirely: a Nicla Vision badge running `pipelines/wearables/nicla-vision/ips/onboard_apriltag_detect.py` detects tags on the device and publishes its results over MQTT to the `<session>/ips` topic.

## Configuration

`pipelines/ips-base/config.yml` is the base station config; `config_template.yml` documents every key. Open the **Config** tab of the IPS Base card and press **Save** to create it from the template, or copy the template by hand.

| Section | What it holds |
|---|---|
| `Base` | settings shared by every base: `tag_size` and `families` of the AprilTags, `resolution`, `rotate`, `fps`, the file-replay pacing (`keyframe_interval`, `processing_rate`, `enable_timing_sync`), and `stream_kwargs` |
| `Bases` | one entry per camera position: `id`, `camera` (a calibrated profile from `Cameras`), `source`, `source_index`, and `main` (exactly one `true`). Camera sync, the bases and the transform matrices all key on these ids. |
| `Cameras` | the intrinsic parameters per camera model, written by the calibration tool (or filled in by hand); the template ships profiles for a Logitech C920, a MacBook Air camera and an iPhone |
| `Synchronizer` | `bucket_duration` |
| `Streams` | managed and external streams, see below |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings, see [System Services](../system_services.md) |

```yaml
Bases:
  - id: cam-front            # base id; appears in MQTT and as the transform matrix key
    camera: logitechC920     # a calibrated profile from Cameras
    source: opencv           # opencv | stream | lsl | file (rtmp is the old name of stream)
    source_index: 0          # opencv: camera index; stream: index among the pullable Streams entries; file: the file's full path; lsl: stream name
    main: true               # exactly one base is the main reference frame
  - id: cam-side
    camera: logitechC920
    source: stream
    source_index: 0
    main: false
```

IPS has no capture/analyze/live mode switch: a live source (`opencv`, `stream`, `lsl`) runs in real time and a `file` source replays the recording at the configured pace.

### Input sources

| Source | Description | Setup |
|---|---|---|
| `opencv` | USB camera on the base station or a Raspberry Pi | `source_index` is the device index: the Config tab lists the cameras found on the card's host (`/dev/video<N>` is index N on Linux, a Mac's in AVFoundation's order) to pick from, and the base lists the devices it finds when it starts |
| `stream` | video pulled from the MediaMTX server (`rtmp` is the old name) | a `Streams` entry whose `read_target` (else `target`) is an `rtmp://`, `rtsp://` or `srt://` URL; `source_index` is its position among those entries |
| `lsl` | Lab Streaming Layer | `source_index` is the stream name; needs `pylsl` |
| `file` | replay of a recorded video | `source_index` is the file to replay, by its full path: **Browse…** on the Config tab writes it, and the dropdown lists the other files of its folder (when the card's host is this machine; on another one the path is typed). Files replayed together sit in one folder: the replay starts at the latest start among them, read from the names. A config from before keeps a `file_dir` in `Base`, where a bare file name was looked up: the bases still read it, and the Config tab turns those names into full paths, which the next **Save** writes (the log says what moved). |

### Streams

```yaml
Streams:
  # external stream (already running; the base pulls read_target, else target)
  cam-external:
    target: rtmp://uber-server.local:1935/ips/3
    read_target: rtsp://uber-server.local:8554/ips/3

  # managed stream: the TUI starts/stops ffmpeg on a remote Raspberry Pi over SSH
  cam-1:
    ssh_profile: rpi-living-room    # must match a TUI SSH profile name
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local:1935/ips/1       # published to MediaMTX
    read_target: rtsp://uber-server.local:8554/ips/1  # pulled by the base
    codec: libx264
    resolution: 1920x1080
    fps: 30
    bitrate: 1M
    record: true                    # also keep an mkv on the Pi for later replay
```

Managed streams are started and stopped from the **Streams** tab; see the [Streaming guide](../rtmp_streaming.md) for the FFmpeg commands and the recording layout.

## Camera calibration and synchronization

One-time setup for a camera arrangement, done in this order from the leaves under `Launcher → Pipelines → IPS`. Redo the synchronization whenever a camera moves.

### Calibrate each camera model

**IPS Camera Calibration** runs `mmla ips-ccal`, which films a checkerboard, computes the intrinsic parameters and writes them into the `Cameras` section of `config.yml` under the name you give the camera. The captured images land in `pipelines/ips-base/camera_calib/cameras/<camera>/`; the panel below the card lists them and can delete images or a whole camera (local host only). Cameras of the same model can share one profile.

### Synchronize the cameras

Define the `Bases` entries first (one per camera position, exactly one `main: true`); the sync manager refuses to start without a main base. **IPS Camera Sync** starts `Num Tag Detectors` instances of `mmla ips-ctag` (one per camera, each asks which `Bases` entry it is) and one `mmla ips-csync` sync manager. The manager pairs the main base with one alternative base at a time (switch to the next alternative from its menu): show one AprilTag to both cameras and start the synchronization, and it computes the transform from that camera into the main camera's frame and accumulates the pairs in `pipelines/ips-base/camera_sync/transformation_matrices.json`. Choose **export transformations** in the manager's menu when every camera is done: it writes `transformation_matrices_<main-id>.json`, which is the file the bases load.

### Distribute the matrices

The **Transform Matrix** tab of the IPS Base card shows the exported files as editable JSON. The tab edits the files of the host it is set to, and **Sync to Host** copies them into `pipelines/ips-base/camera_sync/` on the machine picked beside it: from here to a base station, or from a base station back here. **Delete** removes the file on screen from the host the tab edits, after a second press (the other hosts keep theirs). Every base station that runs an IPS base needs the exported file.

### Calibrate from a recorded session

A recorded session calibrates itself: whenever two cameras saw the same tag at the same moment, the tag's position in both camera frames is one sample of the transform between them. `mmla ses-calibrate` reads the session's videos (from `artifacts/<session>/manifest.json`), detects the tags on a frame every `-st` seconds with the same detector, intrinsics (`Cameras`) and tag size as the IPS base, pairs the sightings of the main camera with each other camera's, fits one rigid transform per camera to the paired positions (with the pairs that disagree thrown out) and writes `artifacts/<session>/analysis/calibration/transformation_matrices_<main>.json`, ready for `camera_sync/`, next to a `calibration_report.json` with the residuals. Given matrices are scored on the same pairs with `-v`, which is how a calibration made on another day is checked against a session:

```bash
mmla ses-calibrate -c pipelines/ips-base/config.yml -sid exp_20260603_microbit_group_01_260603T0826Z -v pipelines/ips-base/camera_sync/calibrations/microbit-2025-10-15/transformation_matrices_c920-01.json
```

The report says, per camera, how many paired sightings there were, the residual of the fit (median and p90, in metres), how far the pose-to-pose average that Camera Sync would compute lies from it, and for the given matrices their residuals, the share of pairs within 0.15 m and their difference to the fit. A camera that never saw a tag together with the main one cannot be placed.

## Run from the TUI

1. **System services** running and reachable, and the setup above done: calibrated `Cameras`, `Bases` with one `main: true`, and the exported transform matrices on every base station.
2. **IPS Base**: `Launcher → Pipelines → IPS → IPS Base`, Host set to the base station. Choose the number of bases, synchronizers and visualizers, the **Session** (or `Create MongoDB Session` from an experiment group), and the toggles (`Graphics` shows the annotated frames, `Store` saves frames, `Verbose` prints debug output). Everything the windows used to ask is chosen on the card:
    - which `Bases` entry each base is, one dropdown per base (their number follows **Num Bases**);
    - the synchronizer's **main camera**, the base whose `camera_sync/transformation_matrices_<id>.json` it loads. The dropdown lists the files exported on the card's host and starts on the `Bases` entry with `main: true` when its file is there, else on the first file; **Start** refuses to launch a synchronizer until one is picked (press Refresh on the card once the files are there);
    - the visualizer's **2d** or **3d** plot.

    **Start** opens one terminal window per instance. Each process starts at once with those choices and waits for START; nothing is asked in the windows.
3. **Session Control**: once every window reports that it is waiting, send **START** for the session; send **STOP** at the end. On STOP every base, synchronizer and visualizer ends its run and exits (the visualizer closes its plot window), also when STOP comes before START; start the card again for the next session. A synchronizer whose run ends on an error instead of STOP (the connection to Redis lost, say) says so and shows its menu, where `1` starts it again.

When a choice from the card cannot be used (the synchronizer finds no `camera_sync/transformation_matrices_<id>.json` for its main camera, the visualizer gets a dimension other than 2d or 3d, or a base gets an id that is not in `Bases`), that window says why and what to do, then shows the process's own menu or base prompt, so it can be fixed there, or on the card before the next Start.

Each base notes in the session's MongoDB document which `Bases` entry it is, the stream it pulls (for a `stream` source) and when it joined and left; it notes the leaving first on its way out, before it stops its threads and stream. **Sessions → Export Streams** reads this note, so it takes the session's own streams, from the Stream Server and from the capture hosts, without being told which. A session without this note (one from before the bases wrote it, or one no base joined) has nothing to export, and the console says so.

## Manual CLI

```bash
conda activate ips-base
P=pipelines/ips-base; C=$P/config.yml

mmla ips-ccal  -p $P -c $C                 # camera intrinsic calibration
mmla ips-ctag  -p $P -c $C -b <base-id>    # one tag detector per camera (-hl true for headless)
mmla ips-csync -p $P -c $C                 # sync manager; pairs each alternative base with the main one

mmla ips-base  -p $P -c $C -sid <session-id> -b <base-id>
mmla ips-sync  -p $P -c $C -sid <session-id> -mc <main-base-id>   # -mc/--main_camera
mmla ips-vis   -p $P -c $C -sid <session-id> -d 3d                # -d/--dimension: 2d (default) or 3d
```

With `-sid`, as the console runs them, each command asks nothing: it starts at once, waits for START and exits when the run ends on STOP. `ips-sync` without `-mc` takes the `Bases` entry marked `main: true` when its transformation file is there, else the only `transformation_matrices_<id>.json` in `camera_sync/`; `ips-base` without `-b` takes the only `Bases` entry there is. Without `-sid` the commands ask for the session, and `ips-base` for its base when `-b` is omitted; `ips-sync` and `ips-vis` show their menus as before (start, set main camera or switch 2d/3d, exit) and go back to them after each run. `pipelines/ips-base/apriltag/` contains printable tag36h11 tags and a resize script; `pipelines/ips-base/docs/clock.html` is a browser clock you can film to check the timing of recordings.

## Post-time processing

Record with **Collection → Collection Session**, then set each base's `source` to `file` and its `source_index` to its file in the `video/` directory from the collection manifest, by its full path (**Browse…** picks it). `keyframe_interval` and `processing_rate` control how fast the recording is replayed.
