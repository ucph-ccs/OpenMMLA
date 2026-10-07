# Configuration

IPS reads one config file, `pipelines/ips-base/config.yml`, on every machine that runs an IPS base, the synchronizer or a camera tool. You edit it on the **Config** tab of the IPS Base card; this page lists its keys, what the synchronizer stores, and how the dashboard lays the positions on the floor.

## Base config

**Save** on the IPS Base card's **Config** tab writes the file on the card's host, from `config_template.yml` the first time; by hand, copy the template. The bases read the config of the machine they run on.

| Section | What it holds |
|---|---|
| [`Base`](#base) | settings shared by every base |
| [`Bases`](#bases) | one entry per camera position |
| [`Cameras`](#cameras) | the intrinsics of each camera model |
| [`Synchronizer`](#synchronizer) | the time buckets and the fusion of the cameras |
| [`Streams`](#streams) | managed and external video streams |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings ([System services](../../system_services.md)); the IPS bases call no AI service, so `Gateway` goes unused |

### Base

| Key | Default | What it does |
|---|---|---|
| `tag_size` | `0.08` | the side of the tag's printed black square, in metres ([AprilTag badges](index.md#apriltag-badges)); `0.061` when the key is left out |
| `families` | `tag36h11` | the AprilTag family |
| `resolution` | `[1920, 1080]` | the capture resolution, width and height |
| `rotate` | `0` | turns the frames in the base, clockwise, for a source the console does not capture ([Turning the picture](#turning-the-picture)) |
| `fps` | `30` | the capture frame rate |
| `display_smoothing` | `0.7` | the base's window only: the weight of a tag's history in the drawn pose, rotations smoothed along the shorter arc; `0` draws the raw pose. The base publishes, and the database and the dashboard keep, the raw poses |
| `display_reset_seconds` | `2.0` | the base's window only: a tag not seen for longer, or turned by more than 45 degrees, is drawn from its raw pose again |
| `keyframe_interval` | `1.0` | `file` sources: seconds of video between two analyzed frames |
| `processing_rate` | `1.0` | `file` sources: the replay's speed against real time; `0.5` gives each frame twice the time |
| `enable_timing_sync` | `true` | `file` sources: keep the replay on its accumulated schedule, so the bases do not drift apart |
| `initial_sync_time` | not set | `file` sources: the unix time the replay starts at, instead of the latest start among the files; not in the template |
| `stream_kwargs` | required | the video stream's options: `buffer_duration` (0.08), `format` (`MJPG`), `resample_method`, `timestamp_offset` (0.0, see [Timestamps](../../streaming/index.md#timestamps)), `connect_wait` (30) and `reconnect_wait` (3600) (see [How the bases pull a stream](../../streaming/index.md#bases-pulling-a-stream)) |

### Bases

| Key | Default | What it does |
|---|---|---|
| `id` | required | the base id; it appears in MQTT and keys the transform matrices |
| `camera` | the first `Cameras` entry by name | an entry of `Cameras`; a name `Cameras` lacks falls back to the default, and the base's log says which it took |
| `source` | required | `opencv`, `stream`, `lsl` or `file` ([Input sources](#input-sources)) |
| `source_index` | | which device, stream or file, by source |
| `main` | `false` | `true` for the main camera, whose frame the room's positions are in: exactly one base, or one per room |
| `room` | empty | the room the camera is in (`A`, `B`, ...), when one config serves several rooms; empty for one room ([Several rooms](calibration.md#several-rooms)) |
| `capture_rotate` | `0` | `file` sources, and a stream no `Streams` entry describes: how far the picture was turned clockwise before it reached the base ([Turning the picture](#turning-the-picture)) |

```yaml
Bases:
  - id: cam-front            # appears in MQTT and keys the transform matrices
    camera: logitechC920     # an entry of Cameras
    source: opencv
    source_index: 0          # the camera's device index
    main: true
  - id: cam-side
    camera: logitechC920
    source: stream
    source_index: cam-side   # the Streams entry it pulls
    main: false
```

An entry whose `id` is still a `<...>` placeholder is ignored. IPS has no capture, analyze or live mode: a live source (`opencv`, `stream`, `lsl`) runs in real time, and a `file` source replays its recording at the pace of `Base.keyframe_interval` and `Base.processing_rate`.

### Cameras

One entry per camera model, named as a base's `camera` names it. **IPS Intrinsics** writes it ([Calibrate each camera model](calibration.md#calibrate-each-camera-model)), and **+ Add Camera** on the **Config** tab adds one by hand.

| Key | Default | What it does |
|---|---|---|
| `fisheye` | required | `true` for a fisheye lens |
| `params` | required | `[fx, fy, cx, cy]`, what the tag detector uses |
| `calibration_resolution` | from the principal point ([Frame size](#frame-size)) | the frame size, `[width, height]`, the intrinsics were calibrated at |
| `K` | required for a fisheye lens | the 3 × 3 intrinsic matrix |
| `D` | required for a fisheye lens | the distortion coefficients, `[k1, k2, p1, p2, k3]`; a fisheye camera's frames are remapped with them, and a pinhole camera's are not used |

An entry holds the intrinsics of the picture as the sensor gives it. Nothing in it changes when a camera is turned: the base turns the intrinsics with the picture ([Turning the picture](#turning-the-picture)).

#### Frame size

The intrinsics hold for the frame size they were calibrated at. For a frame of another size, such as a 960 × 540 recording of a 1920 × 1080 calibration, the base scales `fx` and `cx` by the width ratio and `fy` and `cy` by the height ratio, and says so once in its log. Unscaled, every tag of such a recording would come out twice as far away and off to one side.

??? info "Details: how the base scales the intrinsics"
    - The size compared is the frame's as the sensor gives it, before a file's `capture_rotate` or `Base.rotate` turned it. On a turned picture the scaled intrinsics are turned with it: a half turn takes (cx, cy) to (W - 1 - cx, H - 1 - cy), and a quarter turn swaps `fx` and `fy` as well.
    - An entry without `calibration_resolution` goes by its principal point, which lies near the centre of the calibration images. Frames within 10% of (2cx, 2cy) are read with the intrinsics as they are; frames of another size are scaled from the common frame size nearest (2cx, 2cy). The base's log says which size it took.
    - A frame of another aspect than the calibration's is scaled all the same, with a warning that the result is only approximate. A principal point that lands more than 10% off the frame's centre is warned of too: the intrinsics were then calibrated at another size than the entry says.
    - A fisheye camera's frames are remapped with its own `K` and are not scaled.

### Synchronizer

| Key | Default | What it does |
|---|---|---|
| `bucket_duration` | `1` | seconds per time bucket; the synchronizer writes its three events once per bucket; required |
| `max_lateness` | `5.0` | seconds of frame time a base may fall behind the newest frame before the buckets are written without it |
| `fusion_gate` | `0.25` | metres a camera's position of a tag may lie from the cameras' median and still be averaged |

[What is stored](#what-is-stored) says how the synchronizer uses them.

## Input sources

| `source` | What it reads | `source_index` |
|---|---|---|
| `opencv` | a USB camera on the base station or a Raspberry Pi | the device index: the **Config** tab lists the cameras found on the card's host (`/dev/video<N>` is index N on Linux, a Mac's are in AVFoundation's order), and the base lists the devices it finds when it starts |
| `stream` | video pulled from the Stream Server; `rtmp` is accepted as another name | the name of a `Streams` entry whose `read_target` (else `target`) is an `rtmp://`, `rtsp://` or `srt://` URL; a number is read as its position among them. Left empty, the base takes the only one there is, and stops with the list of names when there are several. |
| `lsl` | a Lab Streaming Layer stream | the stream name; needs `pylsl` ([What you need](index.md#what-you-need)) |
| `file` | a recorded video, replayed | the file's full path: **Browse…** on the **Config** tab writes it, and the dropdown lists the other files of its folder when the card's host is this machine; on another host, type the path |

Files replayed together sit in one folder, named `<prefix>_<timestamp>.<extension>`. The replay starts at the latest start among them, unless `Base.initial_sync_time` is set. [Replay recordings](run.md#post-time-processing) gives the steps.

??? info "Details: configs with `Base.file_dir`"
    A `Base.file_dir` key names the folder where a bare file name in `source_index` is looked up. The bases read it, and the **Config** tab turns such names into full paths, which the next **Save** writes, and the log says what moved.

??? info "Details: badges that detect tags themselves"
    Besides the bases, the synchronizer takes the sightings of a badge with its own camera that detects the tags on board. Such a badge publishes to the session's MQTT topic `<session>/ips` a JSON message with `base_id`, a number above 50000 that stands for the badge, `acquired_time` and `detected_tags`, the tag ids it sees. Its sightings go into the `ips_relation` graph as the badge facing those tags; they give no position and never hold a bucket up. When the session filters tags by its participants, the badge's `base_id` must be its wearer's **Tag ID**. The repository ships no firmware for such a badge.

## Streams

```yaml
Streams:
  # external stream (already running; the base pulls read_target, else target)
  cam-external:
    target: rtmp://uber-server.local:1935/ips/3
    read_target: rtsp://uber-server.local:8554/ips/3

  # managed stream: the console starts and stops ffmpeg on a remote Raspberry Pi over SSH
  cam-side:
    ssh_profile: pi-01              # must match an SSH profile name of the console
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local:1935/ips/cam-side       # published to the Stream Server
    read_target: rtsp://uber-server.local:8554/ips/cam-side  # pulled by the base
    codec: libx264
    resolution: 1920x1080
    fps: 30
    bitrate: 1M
    record: true                    # also keep an mkv on the Pi for later replay
```

A stream with an `ssh_profile` is managed: start and stop it from the **Streams** tab. A stream without one is external and taken as already running. The [Streaming guide](../../streaming/index.md) has every key of an entry, the FFmpeg commands and the recording layout.

### Turning the picture

A camera mounted upside down, or on its side, is turned upright once, where it is captured:

- Its `Streams` entry's `rotate` (0, 90, 180 or 270, clockwise; the **Rotate** cell of the **Streams** tab) makes the capture host's FFmpeg turn the picture. The base, the Stream Server's recordings and the dashboard all get it upright, and the session notes the turn as the stream's `sources[].capture.rotate`.
- `Base.rotate` stays 0. It turns the frames in the base, for a source the console does not capture.
- A file turned before it reached the base, such as a recording `mmla ses-tidy` flipped or a recording of a turned stream, says so with its `Bases` entry's `capture_rotate`.

The base finds the `Streams` entry its source pulls and turns the camera's intrinsics with the picture, from the `Cameras` entry as it was calibrated. It reports every pose in the sensor's frame whatever turned the picture, so the camera sync matrices hold whether the picture is turned or not ([What is stored](#what-is-stored)). The base's session parameters note `capture_turn` beside `rotate`.

!!! warning
    A base whose capture turn and `Base.rotate` are both set warns at start that the picture is turned twice.

## What is stored

For every bucket, the synchronizer writes three events that share `window_start_time` and `window_end_time`:

| Event | Field | Holds |
|---|---|---|
| `ips_translation` | `translations` | `{tag: [[x], [y], [z]]}`: the fused position in metres, in the main camera's frame |
| | `detections` | `{tag: {camera: {t, f, d, m, dt, n, out}}}`: every camera's own detection of the tag in the bucket, for analyses that use or fuse the cameras themselves ([Detection fields](#detection-fields)) |
| `ips_rotation` | `rotations` | `{tag: 3x3}`: the rotation of the tag's best detection, in the main camera's frame |
| `ips_relation` | `graph` | `{tag: [tags it faces]}`, every tag of the bucket a key |

[Databases](../../database.md#influxdb) has the tags and the schema of every event.

### Detection fields

| Field | What it is |
|---|---|
| `t` | the camera's position of the tag, in the main camera's frame |
| `f` | the tag's outward normal, in the main camera's frame |
| `d` | the tag's distance from that camera, in metres |
| `m` | the detector's decision margin; null from a base that sent none |
| `dt` | the frame's time after the bucket's start |
| `n` | how many frames the detection stands for; present only when it is more than one, which happens with a live camera |
| `out` | `true` when the fused position left this camera out; absent otherwise |

### How a bucket is made

1. **Raw poses.** For every frame, a base publishes the pose of every tag it reads, as the detector gave it: the rotation and position in its camera's frame, the decision margin, the pose error and the frame's time. Nothing is carried over from earlier frames, so a position is where the tag was on that frame.
2. **Buckets.** The synchronizer files each frame into the bucket of its time: buckets of `bucket_duration` seconds from the first frame it gets. A bucket is written once every base still sending has moved past it.
3. **Fusion.** Each camera's detection is taken into the main camera's frame with its matrix. Per tag, the cameras whose positions lie within `fusion_gate` metres of the cameras' median position are averaged, the nearer cameras weighing more. The stored rotation is the best camera's: the one nearest the tag.
4. **Who faces whom.** A tag faces another when its outward normal points at the other within about 20 degrees on the floor, and the two are close. There is no history from one bucket to the next.

??? info "Details: poses in the sensor's frame"
    - A tag the camera sees faces the camera. A rotation whose z axis points at the camera, a misread of a small tag, is turned half a turn about the tag's y axis, which keeps the badge's y axis (gravity) and makes -column 2 the outward normal, as every reader takes it.
    - A base reports every pose in its camera's frame as the sensor gives the picture, before any turn. A pose found on a picture turned by `Base.rotate`, and for a file by its `capture_rotate`, is turned back about the optical axis: a half turn negates x and y, and a quarter turn clockwise maps the turned frame's (x, y) back to the sensor's (y, -x). `mmla ips-ctag` and `mmla ses-calibrate` report their poses the same way, so the matrices, the synchronizer and the stored data keep one frame per camera.
    - Who faces whom on one camera's frame is judged on the turned, upright picture's poses, whose x-z plane is the floor's.

??? info "Details: late frames"
    - A file replay's frames lie on whole seconds from the shared start, so each bucket gets one frame per camera.
    - A base more than `max_lateness` seconds of frame time behind the newest frame no longer holds the buckets up, and a frame of a bucket already written is left out.
    - For the first `max_lateness` seconds, a base the matrices place that has sent nothing yet holds the buckets up as well, so a base that starts a little later keeps its first frames.
    - The log names a base's first late frame, how far it was behind the newest frame, and why it came late: the base joined after the bucket was written, it was more than `max_lateness` behind when the bucket was written, or the frame is older than one it sent before. At the end of the run, the log counts such frames per base, and the buckets still open are written.

??? info "Details: the fusion and the facing"
    - A live camera that sent several frames in a bucket stands for the one nearest the median of their positions.
    - Each camera within the gate is weighted by 1 / distance⁴ from the camera that saw the tag; a tag nearer its camera than 0.5 m weighs as much as one at 0.5 m. A tag's depth error, the larger part of its error, grows with the square of its distance, so its variance grows with the fourth power, and the weight is its inverse.
    - When no camera lies within the gate, as with two cameras that disagree, the best camera is taken alone. The best camera is the nearest to the tag, the decision margin breaking a tie. Its rotation is stored because averaging two readings of a small tag's rotation can give one neither camera saw.
    - A tag faces another (`is_tag_looking_at_another_2d`) when a camera that saw both saw it facing on at least half of its frames of the bucket that held both: on their raw poses in that camera's frame, within 1 m of each other, or on the fused poses in the main camera's frame, within 1.2 m.
    - The fusion table counts the facing over its 10 s windows ([Window features](../../analytics/window_features.md)).

## The floor plan

The dashboard lays the positions on the floor as a plan of the room seen from above (`openmmla/analytics/report/space.py`). The badges hang, so the mean of their vertical axes is the room's gravity in the main camera's frame, and the floor is the plane at right angles to it. The plan's axes are `u` to the right of the main camera, `v` away from it and `h` up, so the plan is level even when the main camera looks down at the room.

??? info "Details: the plan's axes and its fallback"
    - The positions are in the main camera's frame as the sensor gives it, so a main camera hung upside down has its x axis pointing to the room's left, and one on its side has it up or down. `u` is therefore the right of the picture as it stands upright: the right axis of the quarter turn whose down axis lies nearest the gravity, laid flat on the floor. For a camera hung upright, rolled less than 45 degrees, that is the camera's x axis; for one hung upside down, -x. `v` is the optical axis laid flat, and (`u`, `v`, `h`) is right-handed.
    - A session with fewer than 30 badge rotations falls back on the main camera's x-z plane, which the gravity cannot turn. The analysis report then turns that plane by the main camera's turn, read from the session document (its base's `capture_turn` and `rotate`, else its stream's `sources[].capture.rotate`), and its floor says `method: camera-xz` with that `turn`.
    - The **Room** card of the live view and of a replay does not take that fallback. Until the badges' rotations it reads give it the floor, it lays the positions on the camera's own x-z plane, which is a mirror image for a main camera hung upside down.

## What a session records

The session's MongoDB document notes what produced its IPS data ([Databases](../../database.md#mongodb)):

- **Each base** notes which `Bases` entry it is and, for a `stream` source, the stream it pulls: the `Streams` entry, its URL and path on the Stream Server, and the machine that captures and records it (the session's `sources`). It writes this when it joins the session, and notes when it leaves, first thing on its way out. **Sessions → Export** reads it, so it takes that session's own streams from the Stream Server and the capture hosts without being told which.
- **Each base** also notes its parameters: its camera and `calibration_resolution`, `tag_size`, the turns (`rotate`, `capture_turn`), its source and the main camera.
- **The synchronizer** notes the main camera and its matrices (`file`, `main_id`, `matrices`), `bucket_duration`, `max_lateness`, `fusion_gate` and the tag ids it keeps. The config and the matrix file are copied into the session's folder beside its other configs.

??? info "Details: when a note is missing"
    A session no base joined has nothing to export, and the console says so. A base or synchronizer that cannot write its note (MongoDB down, or a session the console did not create) warns in its log and runs on.
