# Video Frame Analyzer (VFA)

Gaze-augmented collaborative action recognition with vision-language models. Bases capture a frame per camera angle at a fixed interval, the synchronizer bundles the angles of one moment into a single request, and the VFA server grounds each participant by their AprilTag, overlays gaze cues, and asks a VLM to describe what each person is doing and to classify it into one of five collaborative actions with a transparent, step-by-step justification.

The pipeline, its prompt design and its evaluation against human coders are described in the ICALT 2026 paper *Designing for Transparency: Gaze-Augmented Collaborative Action Recognition with Vision-Language Models*.

## Pipeline overview

![Video frame analysis pipeline: frame capture, frame synchronization, frame analysis, vision-language processing](../../img/video_frame_analyzer.png)

The pipeline has four stages:

1. **Frame Capture (FC)**: each VFA base captures a frame from its video stream every `keyframe_interval` seconds (30 s by default) and broadcasts the frame metadata (camera angle description, frame path) over MQTT.
2. **Frame Synchronization (FS)**: the VFA synchronizer listens on the MQTT channel, aligns the messages of all bases in time, and sends the synchronized multi-angle frames together with the participant and angle descriptions to the analyzer in one HTTP request.
3. **Frame Analysis (FA)**: the VFA analyzer (the VFA server) runs AprilTag detection and gaze detection (RetinaFace for faces, [Gaze-LLE](https://github.com/fkryan/gazelle) for gaze targets), renders the frames with the overlays below, builds the structured prompt, and sends it to the VLM.
4. **Vision-Language Processing (VLP)**: the VLM, local or in the cloud behind an OpenAI-compatible API, performs grounding, captioning and classification and returns JSON. The result goes back to the synchronizer, which writes it to InfluxDB as `vfa_action` events (see the [Database Reference](../../database.md)).

![A rendered frame with face boxes, gaze lines, in-frame probabilities and a repainted AprilTag](../../img/vfa_rendered_frame.png)

A rendered frame carries the cues the prompt refers to: each detected AprilTag is repainted as a black square with its id in white, each detected face gets a coloured box, a gaze line points at the estimated gaze target, and `in: 0.98` is the probability that the gaze target lies inside the frame.

| Component | Runs on | Command | Environment |
|---|---|---|---|
| VFA Base, one per camera angle | base station | `mmla vfa-base` | conda env `vfa-base` |
| VFA Synchronizer, one per session | base station | `mmla vfa-sync` | conda env `vfa-base` |
| VFA Server: the multi-angle frame analyzer | GPU base server | docker compose | image in `docker/` |
| MLLM Server, optional local vision-language model | GPU server | `vllm serve` | conda env `vfa-vllm` |

Create the `vfa-base` environment from the TUI's Environment tab or by hand (`conda create -n vfa-base python=3.10 -y && pip install -e '.[vfa-base]'`). For an `lsl` source add `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.

## Action coding scheme

Every participant in every frame is assigned one of five mutually exclusive actions. Human coders work from the descriptive definitions below; the VLM receives a rule-based formalization of the same definitions, written as explicit conditions on gaze and hands plus a hierarchical decision process (`config/vfa/action_schemas.yml`, schema `collaboration_v1`, editable from the **Action Schema** tab of the VFA Server card).

| Action | Definition |
|---|---|
| Communicating | Actively communicating with others: looking at work items (screen, documents, hardware) while clearly pointing at them, or looking at another person while gesturing or pointing. |
| Observing | Watching or monitoring without communicating or manipulating: looking at work items or people while the hands are resting, hovering without touching, or touching something other than what is being looked at. |
| Manipulating | Directly working with something: looking at a work item while at least one hand clearly touches the same item. |
| Idle-OffTask | Not engaged in the task: looking at personal items (phone, snacks) or outside the camera frame, whatever the hands do. |
| Unclear | The gaze or the hands cannot be seen well enough to tell, because of occlusion, blur or poor visibility. |

The same scheme ships as the default template of the [human coding interface](coding_interface.md), so machine and human codings use identical labels.

## Prompt engineering

![Structure of the chain-of-thought prompt: persona, context, grounding, captioning, classifying, formulating](../../img/vfa_prompt_structure.png)

The prompt mirrors the human annotation process as a stepwise reasoning chain:

1. **Persona assignment** (system prompt): the VLM acts as an expert multi-perspective lab activity analyst.
2. **Context**: a legend of the visual overlays, the camera setup (`{{num_perspectives}}`, `{{angle_descriptions}}`) and the participant descriptions (`{{participant_descriptions}}`).
3. **Grounding**: identify each person by the AprilTag id, else by matching the appearance to the participant descriptions, else assign an id from 100 upwards.
4. **Captioning**: describe gaze focus, hand status (contact, hovering, inactive), position and clothing per person, citing the camera view that supports each observation.
5. **Classifying**: apply `{{action_definitions}}` and `{{decision_process}}` top down, stopping at the first match, with contact evidence when a rule requires contact.
6. **Formulating**: answer in a fixed JSON layout with `observations`, `classifications` and `justifications` per id.

The templates are plain text files under `pipelines/vfa-server/prompts/` (`prompt_templates_dir`), edited from the **Prompts** tab of the VFA Server card, which marks the templates in use. `prompt_profile` selects the end-to-end variant: `cot` (the chain-of-thought prompt above, the paper's main condition), `baseline` (direct classification without the reasoning steps) and `baseline_no_pre` (baseline without the pre-context block). `end_to_end: false` switches to the older two-step mode with the fixed `multi_angle_vlm_*` (vision) and `multi_angle_llm_*` (classification) templates.

Templates substitute `{{variable}}` placeholders, each using only its own subset: `{{num_perspectives}}`, `{{angle_descriptions}}`, `{{participant_descriptions}}`, `{{action_definitions}}`, `{{decision_process}}` (not in the baseline variants) and `{{image_description}}` (two-step LLM prompts only). The participant descriptions come from the experiment selected for the session (`config/experiments.yaml`, edited under **System Settings → Study → Experiments**). A missing templates directory is an error at startup; a missing single template is skipped silently and leaves that prompt empty, so check the path if the model starts receiving bare input.

## Model backend

The server talks to an OpenAI-compatible endpoint. Choose it with `VLLMFrameAnalyzer.backend` in `pipelines/vfa-server/config.yml` and fill in the matching block; the template lists every supported one. In the paper, GLM-4.5V, InternVL-3.5, Gemini-2.5-Pro and GPT-5 were evaluated with the `cot` profile.

Local:

- **vLLM**: the **MLLM Server** card runs `vllm serve` with the model, port and limits from `config/mllm_server.yml` (Qwen3-VL-8B-Instruct by default) in the `vfa-vllm` environment (`pip install -e '.[vfa-vllm-runtime]'`, Python 3.12). Point `vllm.vlm_base_url` at it. Alternatively the `mllm` profile of the VFA compose file runs the official vLLM image; see the [Docker guide](../../docker.md#local-vllm-vlm-backend).
- **Ollama**: install from https://ollama.com/download and pull a multimodal model (`ollama pull llava`); backend `ollama`.
- **llama.cpp**: backend `llamacpp` against a llama-server endpoint.

Cloud, each needing an API key in its block (stored encrypted on save):

- **OpenAI** (`openai`), **Google Gemini** (`gemini`), **xAI Grok** (`grok`), **Zhipu** (`zhipuai`), **InternLM** (`intern`): vision-capable models for both steps.
- **DeepSeek** (`deepseek`): text models only, so usable for the classification step of the two-step mode; the shipped block points `vlm_model` at `deepseek-reasoner`, which does not accept images.
- **Qwen** (`qwen`) through DashScope's OpenAI-compatible endpoint: the template default `qwen2.5-72b-instruct` is text-only, so set `vlm_model` to a `-vl` model.

Because the frame analyzer runs in a container, a backend on the same machine is reached as `http://host.docker.internal:<port>/v1`, not `localhost`.

## Configuration

`pipelines/vfa-base/config.yml` for the bases (created from the Config tab with **Save**, or copied from `config_template.yml`):

| Section | What it holds |
|---|---|
| `Base` | shared settings: `tag_size` and `families`, `resolution`, `rotate`, `fps`, `keyframe_interval` (seconds between analyzed frames), `angle_config` (the viewing angles: a name and what a camera at that angle sees, added and removed on the Config tab), the file-replay pacing (`processing_rate`), `stream_kwargs` |
| `Bases` | one entry per camera angle: `id`, `camera` (a calibrated profile from `Cameras`, shared with IPS), `source`, `source_index`, `camera_angle` (picked from the names of `angle_config`) |
| `Synchronizer` | `result_expiry_time`, `match_tolerance` |
| `Streams` | managed and external streams |
| `Server.vfa` | the frame analyzer endpoint, through the gateway (`http://<gateway>:8080/vllm`) or direct (`http://<server>:5007/vllm`) |
| `Cameras` | calibrated camera profiles; calibrate with the IPS camera calibration tool |
| `InfluxDB`, `MongoDB`, `MQTT`, `Redis`, `Gateway` | mirrored from System Settings |

`pipelines/vfa-server/config.yml` for the server: `backend` and its block, `end_to_end`, `prompt_profile`, `image_detail`, the AprilTag `families`, optionally `action_schema`, and the `features` block of the [features endpoint](#features-endpoint-skeletons-and-gazes) (`enabled`, `pose_model`, `weights_dir`, `pose_confidence`, `keypoint_confidence`, `inout_threshold`).

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
    target: rtmp://uber-server.local:1935/vfa/side
    read_target: rtsp://uber-server.local:8554/vfa/side

  # managed stream: the TUI starts/stops ffmpeg on a remote Raspberry Pi over SSH
  cam-front:
    ssh_profile: rpi-front          # must match a TUI SSH profile name
    device: /dev/video0             # camera device on the remote machine
    target: rtmp://uber-server.local:1935/vfa/front       # published to MediaMTX
    read_target: rtsp://uber-server.local:8554/vfa/front  # pulled by the base
    codec: libx264
    resolution: 1920x1080
    fps: 30
    bitrate: 1M
    record: true                    # also keep an mkv on the Pi for later replay
```

Managed streams are started and stopped from the **Streams** tab; see the [Streaming guide](../../rtmp_streaming.md) for the FFmpeg commands and the recording layout.

## Run from the TUI

1. **VFA Server**: `Launcher → Pipelines → VFA → VFA Server`, Host set to the GPU server. Set the backend, models, `end_to_end` and `prompt_profile` on the Config tab, review the Prompts and Action Schema tabs, then **Start**: the card runs `docker compose -f docker/docker-compose.vfa.yml up -d --build frame-analyzer`. Start the **MLLM Server** card first if you use a local vLLM model.
2. **System services** running, and `Server.vfa` pointing at the server or the gateway.
3. **VFA Base**: Host set to the base station. Choose the number of bases and synchronizers, which `Bases` entry each base is (one dropdown per base, as many as the base count), the **Session**, the **Mode**, the toggles (`Graphics`, `Store Frames`, `Verbose`) and what the synchronizer asks the frame analyzer for: **Action Labels** (`-a`, the VLM's per-window labels, the `vfa_action` event), **Pose** (`-pose`, the [features endpoint](#features-endpoint-skeletons-and-gazes): skeletons, AprilTags and head yaws, one `vfa_features` event per synchronized frame set) and **Gaze** (`-gaze`, where each person looks, with the pose: a gaze lands on someone's face or hands, so Gaze on turns Pose on). Each is a three-way choice: `as the config says` passes nothing and the config's `Synchronizer.actions` (true by default), `Synchronizer.pose` and `Synchronizer.gaze` (false by default) decide, `on` and `off` override them for this start. The three are independent requests to the same server: the action labels run the VLM on frames the server first annotates with AprilTags and gaze lines (its own `april_tag` and `gaze_detect` settings), so they use the gaze model as a hint for the VLM but return no gaze and run no pose model; Pose and Gaze return data and run no VLM. With Action Labels and Gaze both on, the gaze model runs twice per frame set. The labels and the pose run at different paces: for a pose run set `Base.keyframe_interval` to about 1 second and `Synchronizer.action_interval` to 30, so the frame sets come every second and the VLM is asked at most every 30 s; for a labels-only run keep `keyframe_interval` at 30 and `action_interval` at 0. The synchronizer waits for as many bases as **Sync Waits For** says (`--num_bases`): it starts on the number of `Bases` entries, and is set by hand when bases of the session run on other hosts or fewer run than the list has. **Start** opens one terminal window per instance, and nothing in them asks anything: each base runs as its `Bases` entry, and the synchronizer starts on the session at once and waits for START.
4. **Session Control**: send **START**, and **STOP** at the end. STOP ends the run of every base and synchronizer of the session, and each of them exits, also when STOP comes before START; start them again from the card for the next session.

Modes: `live` analyzes frames as they are captured (the TUI default); `capture` only stores frames, which is how frames for human coding are collected; `analyze` re-runs the analysis on the frames this session stored earlier.

A process that cannot start says why in its window. A synchronizer with no number of bases to use (`-nb 0`, or no `-nb` and an empty `Bases` list), one that cannot reach Redis, or one whose run ends on an error rather than STOP, prints the reason and falls back to its menu (`1: start`, `2: reinitialize` to reload the config, `0: exit`), which keeps the session it was started for. MQTT and MongoDB are connected to as the synchronizer starts, before there is a menu: when either cannot be reached it says so in plain words, names what to check, and exits; start them from **System Services** and press **Start** on the card again. A base launched without an id takes the only `Bases` entry there is; given an id that the `Bases` list does not have, or no id while there are several entries, it lists the ids there are and asks which one it is.

### What a session records

Each base notes in the session's MongoDB document which `Bases` entry it is and the stream it pulls: the `Streams` entry, its URL and path on the Stream Server, and the machine that captures and records it (the session's `sources`, see the [Database Reference](../../database.md#mongodb)). It writes this when it joins the session and notes when it leaves, first thing on its way out, before it stops its threads and stream. **Sessions → Export Streams** reads it, so it takes that session's own streams, from the Stream Server and from the capture hosts, without being told which. A session recorded before bases did this, or one no base joined, has nothing to export, and the log says so. A base that cannot write the note (MongoDB down, or a session the console did not create) warns in its log and runs on.

## Manual CLI

```bash
conda activate vfa-base
mmla vfa-base -p pipelines/vfa-base -c pipelines/vfa-base/config.yml -m live -sid <session-id> -b <base-id>
mmla vfa-sync -p pipelines/vfa-base -c pipelines/vfa-base/config.yml -sid <session-id> -nb <number-of-bases>
# pose and gaze instead of (or next to) the action labels; the bases' keyframe_interval sets the rate
mmla vfa-sync -p pipelines/vfa-base -c pipelines/vfa-base/config.yml -sid <session-id> -nb 2 -a False -pose True -gaze True
```

With `-sid`, as the card runs them, both start at once and exit when the session is stopped; `-nb` (`--num_bases`) defaults to the number of entries in `Bases`, and `vfa-base` without `-b` takes the only `Bases` entry there is (with several it asks). Without `-sid` the synchronizer opens its menu, where `1: start` asks for the session and, unless `-nb` is given, the number of bases; the base asks for the session, and without `-b` for its `Bases` entry.

To run the server without Docker (`pip install -e '.[vfa-server]'`):

```bash
export PROJECT_DIR=pipelines/vfa-server CONFIG_PATH=pipelines/vfa-server/config.yml
gunicorn -k gevent -w 1 -b 0.0.0.0:5007 openmmla.services.vfa.apps.serve_multi_angle_vllm_frame_analyzer:app
```

The server also answers `GET /vllm/info` (through the gateway too) with what it runs — the backend, its VLM and LLM models and their addresses, the prompt profile, the action schema, the temperature — and the VFA synchronizer asks and notes the answer in the session's document, so that a session's actions can be traced to the models and prompts that produced them (see [Databases](../../database.md#mongodb)). A server running an older openmmla has no `/info`, and the session says so instead.

## Smoke test

With the VFA Server running, send still frames straight to it without starting any base:

```bash
python pipelines/vfa-base/examples/analyze_video_frame.py front.jpg side.jpg \
  --angles front,side \
  --participant-descriptions '{"1": "person with red shirt"}'
```

## Features endpoint: skeletons and gazes

Next to the action classification, the same server answers `POST /vllm/features` (through the gateway too) with the persons of each frame as **geometry**, and no VLM is involved: it is the continuous, low-level signal source for interaction analysis, where `/vllm` gives one semantic label per window.

For every image sent (`images`, with `angles` naming each) the answer holds one frame:

```json
{"frames": [{"angle": "front", "width": 1920, "height": 1080,
             "tags": {"0": [1010.5, 560.0]},
             "persons": [{"person_id": "0", "tag_id": 0, "tag_match": "torso", "bbox": [775, 70, 1359, 938], "score": 0.93,
                          "keypoints": [[x, y, confidence], "... 17 in COCO order"],
                          "head_yaw": -3.6,
                          "face_bbox": [1000, 230, 1210, 400],
                          "gaze": {"point": [1080, 760], "inout": 0.91,
                                   "target": {"category": "zone", "person_id": null, "zone": "table"}}}],
             "pairs": {"0|1": {"gaze_distance": 412.0, "hand_distance": 254.4}}}],
 "pose_model": "yolo11n-pose.pt", "gaze": true}
```

- **Skeletons** come from an Ultralytics YOLO pose model (`features.pose_model`, `yolo11n-pose.pt` by default): a box, a score and the 17 COCO keypoints per person, in pixels from the top-left corner. The weights are fetched once, at the first start, into `pipelines/vfa-server/weights/` (the container's `/project/weights`); put the file there yourself on a host without internet. A model that cannot be loaded, or one that does not answer 17 keypoints, makes the endpoint answer 503 with the reason. **Licence**: `ultralytics` is AGPL-3.0 while openmmla is MIT; serving the frame analyzer image over a network with it inside carries the AGPL source-offer obligation for the combined work (see the [Docker guide](../../docker.md)). A deployment that must stay AGPL-free leaves `ultralytics` out and sets `features.enabled: false`.
- **Identity** is the AprilTag on the chest: a tag inside a person's torso (the shoulders and hips, or, seated with the hips hidden, a box hanging from the shoulder line) beats one merely inside their box, the nearest such tag wins, and tags and persons are matched one to one; `tag_match` says which (`torso`, `box`). The others are `unknown_1`, `unknown_2` ... from left to right, and are bystanders a client should drop.
- **Head yaw** is read from where the nose sits between the ears (between the eyes when an ear is hidden, scaled for their narrower arc): 0 facing the camera, positive turned towards the right of the image, past ±90 seen from behind, about ±70 when only one ear is seen, `null` when the nose is hidden.
- **Gaze** comes from the gaze model the server already runs for the overlays (RetinaFace and Gaze-LLE): the face box, the point the gaze lands on, the probability it lands in the frame at all, and what it lands on. Every target within reach is scored and the best taken: a partner's face (`partner_face`, with their id) beats any hands, the nearest hands (`own_hands` or `partner_hands`) beat a zone, and a named zone beats `elsewhere`; `out_of_frame` below `inout_threshold` (the request's `inout_threshold`, else the config's), and `unknown` for a person the gaze model found no face for. Every target is widened by what the model cannot resolve, one cell of its 64 by 64 heatmap (30 px at 1920 wide). Zones are polygons the request names in `zones`, `{"table": [[x, y], ...]}` for every frame or `{"front": {"table": [...]}}` per angle, in pixels or in `[0, 1]`.
- **Pairs** give, for every two persons, how far apart their gazes land (a joint-attention proxy; `null` when either gaze is out of the frame or missing) and how close their hands come, in pixels. The frame also echoes the `zones` as resolved, in pixels, so a zone sent in the wrong units shows up at once; a zone that is not a polygon, or a frame that is not an image, is answered with 400 naming it. A gaze model that fails on a frame leaves a `gaze_error` on that frame and the skeletons stand.
- **Privacy**: the answer holds boxes, keypoints and scalars, never pixels, and `/features` writes no frame to the server's `temp/` (where `/vllm` keeps its overlays); `keypoints=false` in the request leaves even the skeletons out.

The endpoint is stateless: one request, one set of synchronized frames. The synchronizer sends every frame set to it when its **Pose** (or **Gaze**) is on and writes the answer as one `vfa_features` event (see the [Database Reference](../../database.md#influxdb)); with **Gaze** off the request carries `gaze=false` and the server answers the pose alone (its own `features.gaze` sets its default); `Synchronizer.pose_keypoints: false` keeps the skeletons out of the events, and `Synchronizer.feature_zones_file` names a JSON file of zones. It also publishes each answer to the bases (`<session>/vfa/features` on MQTT): a base with **Graphics** on draws its own angle's boxes, tags, head yaws, skeletons and gaze lines on its live window, with a note of how old they are (the window runs at the camera's rate, the pose at the bases' `keyframe_interval`). Tracking across frames, joining to the IPS positions and windowing into 10 s features belong to the analytics ([window features](#window-features-the-fusion-table)). The `features` block of the server config turns it off (`enabled: false`), picks the model and the confidence thresholds; the client helper is `request_frame_features` in `openmmla/services/vfa/requests.py`, and the example below sends still frames to it.

```bash
python pipelines/vfa-base/examples/frame_features.py front.jpg side.jpg --angles front,side \
  --zones '{"table": [[0, 0.55], [1, 0.55], [1, 1], [0, 1]]}'
```

## Window features: the fusion table

`mmla ses-fuse` turns a session's events into one table, one row per window (10 s by default), with the speech, space, body, gaze and action features side by side and a coverage count per modality, so that interaction detection, modality ablations and human codings can be joined on the window:

```bash
mmla ses-fuse -c pipelines/vfa-base/config.yml -sid <session-id>                       # from InfluxDB
mmla ses-fuse --measurements artifacts/<session-id>/measurements -w 10 -st 10 -o features.csv   # from an export
```

| Columns | From | What |
|---|---|---|
| `window_index`, `window_start`, `window_end` | the session's span | the window this row covers: an index from 0 and its unix start and end in seconds, the keys a human coding joins on |
| `speech_ratio`, `silence_ratio`, `n_speakers_named`, `spk_<name>_ratio`, `n_spurts`, `mean_spurt_seconds`, `words` | `asr_recognition`, `asr_transcription` | how much of the window held speech (a speaker several microphones heard in one bucket counts once, and no bucket counts for more than it covers), how many named speakers and how much each, the talk spurts and words that started in it |
| `dia_speakers`, `dia_switches`, `dia_overlap_ratio`, `dia_share_entropy` | `asr_transcription.diarization` | the anonymous speaker turns of the diarized chunks (labels hold within a chunk, so these are per chunk: the most speakers, the switches and the overlapped time summed, the evenness of the shares averaged) |
| `p<tag>_present_ratio`, `p<tag>_path_m`, `pair<a>_<b>_dist_mean_m`, `_dist_min_m`, `_face_ab_ratio`, `_face_ba_ratio`, `_face_mutual_ratio` | `ips_translation`, `ips_relation` | presence, movement, distance, who faced whom |
| `p<tag>_yaw_mean`, `_yaw_abs_mean`, `_yaw_std`, `_wrist_speed`, `_gaze_switches`, `_gaze_<category>_ratio`; `pair<a>_<b>_hand_dist_min`, `_hand_dist_mean`, `_gaze_dist_mean`, `_joint_attention_ratio`, `_mutual_gaze_ratio` | `vfa_features` | head turn, hand activity (each hand followed from frame to frame, in frame widths per second), where gazes landed, joint attention (gazes within 5% of the frame's width), mutual gaze, hand distance; every frame of every camera angle that saw the person or the pair counts once, and a frame set belongs to the one window its moment falls in |
| `p<tag>_action`, `pair<a>_<b>_co_manipulating` | `vfa_action` | the VLM's label for the window: the newest one up to the window's end, remembered for at most 60 s before its start |
| `n_asr_recognition`, `n_asr_transcription`, `n_ips`, `n_ips_relation`, `n_vfa_features`, `n_vfa_angles`, `n_vfa_action`, `p<tag>_frames`, `pair<a>_<b>_frames` | all | coverage: how many records each modality contributed to the window |

Persons are tag ids (`-tags 0,1` names them; else the tags the events hold), pairs are sorted tag pairs; persons the pose model saw without a tag are left out. A cell is empty when its modality contributed nothing to the window. The table lands in `artifacts/<session>/analysis/features/` with an analysis record next to it saying which events it came from; the export folder's `<session>_features.json` is the `vfa_features` events.

## Human coding and evaluation

Ground truth for VFA is produced with the [human coding interface](coding_interface.md), a single HTML page in `pipelines/vfa-base/coding-interface/`. Run a base in `capture` mode (or `live` with `Store Frames` on) to collect frames; they land under `artifacts/runtime/pipelines/vfa-base/<host>/real-time/runtime/<camera>_<base-id>/` named `<unix-timestamp>.jpg`. Coders load the same frames and the action template, code every participant in every frame, and export a JSON file whose windows mirror the pipeline's `action_recognition` output, so human and machine codings can be joined on the frame timestamp and the participant id.

In the paper, three researchers coded two pilot sessions this way (Cohen's κ 0.73 to 0.84), a majority vote formed the gold standard, and each VLM was run five times over 214 person-frame codings. Manipulating and Observing were recognised most reliably; Communicating was the hardest class.

## Post-time processing

Record with **Collection → Collection Session**, then set each base's `source` to `file` and its `source_index` to its file in the `video/` directory from the collection manifest, by its full path (**Browse…** picks it), and run in `live` mode; `keyframe_interval` and `processing_rate` control the replay pace.
