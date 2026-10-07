# Configuration

VFA reads two config files: `pipelines/vfa-base/config.yml` for the bases and the synchronizer, and `pipelines/vfa-server/config.yml` for the VFA Server. You edit both on the **Config** tab of their card in the management console; this page lists their keys.

## Base config

`pipelines/vfa-base/config.yml` lives on every base station. **Save** on the VFA Base card's **Config** tab writes it on the card's host; by hand, copy it from `config_template.yml`.

| Section | What it holds |
|---|---|
| [`Base`](#base) | settings shared by every base |
| [`Bases`](#bases) | one entry per camera angle |
| [`Synchronizer`](#synchronizer) | how frames are merged, and the outputs asked for |
| [`Streams`](#streams) | managed and external video streams |
| `Server.vfa` | `vllm_frame_analyzer`, the frame analyzer endpoint: the bare name `vllm` goes through the Gateway (`http://<gateway>:8080/vllm`), a full URL goes straight to the server (`http://<server>:5007/vllm`). The action labels go to this URL, the pose and the gaze to it with `/features` appended. |
| `Cameras` | calibrated camera profiles (`fisheye`, `params`, `K`, `D`), in the form IPS uses: **IPS Intrinsics** writes them into the IPS Base config, from which you copy an entry here, or **+ Add Camera** adds one |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings ([System services](../../system_services.md)) |

### Base

| Key | Default | What it does |
|---|---|---|
| `tag_size` | `0.08` | the AprilTag size, in metres |
| `families` | `tag36h11` | the AprilTag family |
| `resolution` | `[1920, 1080]` | the capture resolution, width and height |
| `rotate` | `0` | turns the frames in the base, clockwise; see [Turning the picture](#turning-the-picture) |
| `fps` | `30` | the capture frame rate |
| `angle_config` | `front`, `front-top-45`, `top` | the viewing angles: a name and what a camera at that angle sees, which travels with its frames to the VLM; added and removed on the **Config** tab. The synchronizer looks each frame's angle up in the `angle_config` of its own host's config, and sends `Image from <angle> perspective` for an angle it does not find. |
| `keyframe_interval` | `1.0` | seconds between two analyzed frames, so the pace of the frame sets |
| `processing_rate` | `1.0` | file sources: the replay's speed against real time; `0.5` gives each frame twice the time |
| `enable_timing_sync` | `true` | file sources: keep the replay on its accumulated schedule, so it does not drift |
| `initial_sync_time` | not set | file sources: the unix time the replay starts at, instead of the latest start among the files |
| `stream_kwargs` | | the video stream's options: `buffer_duration` (0.08), `format` (`MJPG`), `resample_method`, `timestamp_offset` (0.0, see [Timestamps](../../streaming/index.md#timestamps)), `connect_wait` (30) and `reconnect_wait` (3600) (see [Bases: pulling a stream](../../streaming/index.md#bases-pulling-a-stream)) |

### Bases

| Key | What it is |
|---|---|
| `id` | the base id; storage paths read `<camera>_<id>` |
| `camera` | a calibrated profile from `Cameras`, shared with IPS |
| `source` | `opencv`, `stream`, `lsl` or `file` ([Input sources](#input-sources)) |
| `source_index` | which device, stream or file, by source |
| `camera_angle` | the viewing angle, by its name in `Base.angle_config` |
| `capture_rotate` | `file` sources: how far the file's picture was turned clockwise before it was saved ([Turning the picture](#turning-the-picture)) |

An entry whose `id` is still a `<...>` placeholder is ignored.

### Synchronizer

| Key | Default | What it does |
|---|---|---|
| `result_expiry_time` | `30` | seconds a frame set waits for missing bases; then it is sent with the frames it has, or dropped with only one |
| `match_tolerance` | `0.5` | seconds within which frames of different bases count as one moment |
| `actions` | `false` | ask for the [action labels](action-labels.md) ([Choosing the outputs](run.md#choosing-the-outputs)) |
| `pose` | `true` | ask for the [pose](pose-and-gaze.md#pose) |
| `gaze` | `true` | ask for the [gaze](pose-and-gaze.md#gaze); turns the pose on too |
| `action_interval` | `30` | the least seconds of frame time between two action-label requests; the frame sets in between get no labels; `0` asks for every frame set; `30` when left out |
| `pose_keypoints` | `true` | keep the skeletons in every `vfa_features` event; `false` keeps the derived fields alone, for smaller events |
| `feature_zones_file` | not set | a JSON file of named polygons a gaze may land in, `{name: [[x, y], ...]}` for every angle or `{angle: {name: polygon}}`, in pixels or in `[0, 1]`; relative to the config's folder. A file that cannot be read is logged and ignored. |

The card's **Action Labels**, **Pose** and **Gaze** override `actions`, `pose` and `gaze` for one start.

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

## Streams

```yaml
Streams:
  # external stream (already running; the base pulls read_target, else target)
  cam-external:
    target: rtmp://uber-server.local:1935/vfa/side
    read_target: rtsp://uber-server.local:8554/vfa/side

  # managed stream: the console starts and stops ffmpeg on a remote Raspberry Pi over SSH
  cam-front:
    ssh_profile: rpi-front          # must match an SSH profile name of the console
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local:1935/vfa/front       # published to the Stream Server
    read_target: rtsp://uber-server.local:8554/vfa/front  # pulled by the base
    codec: libx264
    resolution: 1920x1080
    fps: 30
    bitrate: 1M
    record: true                    # also keep an mkv on the Pi for later replay
```

A stream with an `ssh_profile` is managed: start and stop it from the **Streams** tab. A stream without one is external and taken as already running. The [Streaming guide](../../streaming/index.md) has every key of an entry, the FFmpeg commands and the recording layout.

### Turning the picture

A camera mounted upside down, or on its side, is turned upright once, where it is captured:

- Its `Streams` entry's `rotate` (0, 90, 180 or 270, clockwise; the **Rotate** cell of the **Streams** tab) makes the capture host's FFmpeg turn the picture ([Streaming guide](../../streaming/index.md)). The frames reach the base, the VFA Server and the dashboard upright, and the session notes the turn as the stream's `sources[].capture.rotate`.
- `Base.rotate` stays 0. It turns the frames in the base, for a source the console does not capture.
- A file turned before it reached the base, such as a recording `mmla ses-tidy` flipped, says so with its `Bases` entry's `capture_rotate`.

A fisheye camera's frames that were turned on capture are remapped as they arrive, with the camera's `K` turned with them; its distortion `D` is radial and holds as it is. The base's session parameters note `capture_turn` beside `rotate`.

!!! warning
    A base whose capture turn and `Base.rotate` are both set warns at start that the picture is turned twice.

## Server config

`pipelines/vfa-server/config.yml` lives on the GPU server, under the section `VLLMFrameAnalyzer`. **Save** on the VFA Server card's **Config** tab writes it on the card's host. The server reads it when its container starts, so **Stop** and **Start** the card after a change.

### Keys for every output

| Key | Default | What it does |
|---|---|---|
| `port` | `5007` | the port the card finds the service by; the container listens on 5007 whatever it says |
| `workers` | `1` | the container runs one gunicorn worker whatever it says, and one it must be: the pose's tracks live in the process |
| `april_tag` | `true` | load the AprilTag detector: the pose's identity reads the tags, and the action labels' marks draw them when `action_overlays` allows; not in the template |
| `families` | `tag36h11` | the AprilTag families; required while `april_tag` is on |
| `gaze_detect` | `true` | load the face detector and the gaze model; off, the features get no gaze and the action labels' frames no gaze lines; not in the template |

### Keys for the action labels

| Key | Default | What it does |
|---|---|---|
| `backend` | `vllm` | the VLM ([Model backends](action-labels.md#model-backends)); `vllm` when left out; the server needs this backend's block to start ([Once per deployment](run.md#once-per-deployment)) |
| `end_to_end` | `false` | `false`: a VLM describes and an LLM classifies; `true`: one VLM call ([How labelling works](action-labels.md#how-labelling-works)) |
| `prompt_profile` | `cot` | the one-step templates, with `end_to_end: true` ([Prompt profiles](action-labels.md#prompt-profiles)) |
| `action_overlays` | `auto` | the marks drawn on the frames the VLM sees: `auto`, `all`, `tags`, `gaze` or `none`; the features are not affected ([Marks on the frames](action-labels.md#marks-on-the-frames)) |
| `image_detail` | `auto` | the image detail a vision model is asked for: `low` (faster), `high` (slower) or `auto` |
| `action_schema` | the file's `default_schema` | a schema of `config/vfa/action_schemas.yml` |
| `prompt_templates_dir` | `prompts` | the templates folder, relative to `pipelines/vfa-server` |

### Backend blocks

Each backend has a block of its own, named after it (`vllm:`, `openai:` ...):

| Key | What it is |
|---|---|
| `api_key` | the key; a `vllm`, `ollama` or `llamacpp` block may leave it out, and `EMPTY` is sent; the console stores it encrypted on save |
| `vlm_base_url` | the endpoint of the VLM; a block without it uses the provider's default |
| `llm_base_url` | the endpoint of the classification step, with `end_to_end: false` |
| `vlm_model`, `llm_model` | the models of the two steps; required |
| `temperature`, `top_p` | optional; leave them out for a model that does not take them |
| `VLMExtraBody`, `LLMExtraBody` | extra fields for each step's request body |

??? info "Details: the shipped blocks"
    | Backend | Endpoint | Model |
    |---|---|---|
    | `vllm` | the MLLM Server ([Local VLM with the MLLM Server](action-labels.md#local-vlm-with-the-mllm-server)) | `Qwen/Qwen3-VL-8B-Instruct` |
    | `ollama` | `http://host.docker.internal:11434/v1` | `llava:13b`, `llama3.1` for the LLM step |
    | `llamacpp` | `http://host.docker.internal:8001/v1` | `ggml-org/SmolVLM-500M-Instruct-GGUF` |
    | `openai` | the provider's default | `gpt-4o` |
    | `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-pro-exp-02-05` |
    | `grok` | `https://api.x.ai/v1` | `grok-4-latest` |
    | `deepseek` | `https://api.deepseek.com` | `deepseek-reasoner` |
    | `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen2.5-72b-instruct` |
    | `zhipuai` | the provider's default | `glm-4.5v` |
    | `intern` | `https://chat.intern-ai.org.cn/api/v1/` | `internvl3.5-241b-a28b` |

    `host.docker.internal` is the frame analyzer's own host as seen from its container; without Docker it is `localhost` ([Run the server without Docker](run.md#run-the-server-without-docker)).

### Keys for the pose

The `features` block:

| Key | Default | What it does |
|---|---|---|
| `enabled` | `true` | answer `/vllm/features`; `false` makes it answer 503 |
| `pose_model` | `yolo26n-pose.pt` | an Ultralytics pose weights file: a bare name or relative path lives under `weights_dir`, and a bare name is fetched there once, at start; an absolute path is used as it is |
| `weights_dir` | `weights` | where the pose weights live, relative to `pipelines/vfa-server` (the container's `/project`, so they survive a rebuild) |
| `pose_confidence` | `0.25` | a person below this detection confidence is left out |
| `keypoint_confidence` | `0.3` | a keypoint below this confidence counts as hidden (head yaw, torso, hands, head boxes) |
| `tracking` | | the tracker and its appearance checks ([Tracking settings](pose-and-gaze.md#tracking-settings)) |

### Keys for the gaze

| Key | Default | What it does |
|---|---|---|
| `gaze_backend` | `page` | the gaze model: `page` (PaGE) or `gazelle` (Gaze-LLE) ([Gaze models](pose-and-gaze.md#gaze-models)) |
| `gaze_model` | the backend's default | its checkpoint |
| `gaze_head_scale` | `1.3` | PaGE only: how far the face box is widened into the head crop it looks at; 1 crops the face box as it is |
| `gaze_head_box_fallback` | `true` | a person the face detector missed gets a head box from the pose's nose, eyes and ears ([Head boxes from the pose](pose-and-gaze.md#head-boxes-from-the-pose)) |
| `gaze_face_detector_bgr` | `true` | hands RetinaFace the frame in BGR, the order it expects ([Face detector input](pose-and-gaze.md#face-detector-input)) |
| `features.gaze` | `true` | whether the features carry gazes when a request does not say; a synchronizer with **Gaze** on does not say, so `false` here leaves its features without gazes |
| `features.inout_threshold` | `0.5` | below this probability a gaze is `out_of_frame`; a request may name its own |

The gaze lines of the action labels use the same gaze model.

### Config tab dropdowns

On the VFA Server card's **Config** tab, these keys are dropdowns:

| Key | Choices |
|---|---|
| `backend` | `vllm`, `ollama`, `llamacpp`, `openai`, `gemini`, `qwen`, `deepseek`, `grok`, `zhipuai`, `intern` |
| `prompt_profile` | `cot`, `baseline`, `baseline_no_pre` |
| `action_overlays` | `auto`, `all`, `tags`, `gaze`, `none` |
| `image_detail` | `auto`, `low`, `high` |
| `gaze_backend` | `page`, `gazelle` |
| `gaze_model` | the six checkpoints of [Gaze models](pose-and-gaze.md#gaze-models) |
| `features.pose_model` | `yolo26`, `yolo11` and `yolov8` pose weights in sizes `n`, `s`, `m`, `l` and `x`, such as `yolo26s-pose.pt` |

A value that is not in the list, such as a local checkpoint, is shown, marked and kept. The blank entry keeps the template's default.

## What a session records

Each session's MongoDB document notes what produced its VFA data ([Databases](../../database.md#mongodb)):

- **Each base** notes which `Bases` entry it is and the stream it pulls: the `Streams` entry, its URL and path on the Stream Server, and the machine that captures and records it (the session's `sources`). It writes this when it joins the session, and notes when it leaves, first thing on its way out. **Sessions → Export** reads it, so it takes that session's own streams from the Stream Server and the capture hosts without being told which.
- **The synchronizer** notes what it runs with: the number of bases, its tolerances, the participant descriptions, the outputs asked for, the zones and `action_interval`.
- **The VFA Server** is asked `GET /vllm/info` (also through the Gateway) by the synchronizer, which notes the answer. It holds the backend, its VLM and LLM models and their addresses, the prompt profile, the action schema, the temperature, the gaze backend and model, and under `features` the pose model, its thresholds, the hand circle and the tracking settings. A server without `/info` is noted with an error instead.

So a session's action labels and features can be traced to the models, prompts and settings that produced them.

??? info "Details: when a note is missing"
    A session no base joined has nothing to export, and the log says so. A base that cannot write its note (MongoDB down, or a session the console did not create) warns in its log and runs on.
