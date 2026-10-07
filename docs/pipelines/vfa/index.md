# Video frame analyzer (VFA)

The video frame analyzer watches a group at work through cameras at several angles and turns each moment into data: where each person is, where they look, and what they are doing. Use it when a study needs body, gaze or activity measures from video next to speech ([ASR](../asr/index.md)) and positions ([IPS](../ips/index.md)).

## What VFA produces

| Output | What it gives | What it is for | Event |
|---|---|---|---|
| [Action labels](action-labels.md) | per person, one of five collaborative actions, with the observations and a step-by-step justification | a coder-like account of what each person is doing | `vfa_action` |
| [Pose](pose-and-gaze.md#pose) | per person, a box, 17 keypoints, the AprilTag they wear, a track id and the head yaw; per pair, how close their hands come | who is where, head turns, hand activity | `vfa_features` |
| [Gaze](pose-and-gaze.md#gaze) | per person, the face box, the point the gaze lands on, the probability it lands in the frame, and what it lands on; per pair, how far apart the gazes land | who looks at whom or at what, joint attention | `vfa_features` |

The action labels come from a vision-language model (VLM), one label per person each time the VLM is asked. The pose and the gaze are geometry from a pose model and a gaze model, one answer per frame set and no VLM. Each output is switched on and off on its own: see [Choosing the outputs](run.md#choosing-the-outputs).

![A classroom frame with the pose and gaze VFA computes drawn on it: each badge wearer's skeleton and box in their tag colour, labelled Tag 0 and Tag 1, a dashed gaze line from each face to what it lands on (partner_face, partner_hands), and an adult without a badge in grey; every head is pixelated](../../img/vfa/overlay.png)

## How it works

![The VFA path for action labels: VFA bases capture frames at an interval and broadcast them over MQTT (FC); the VFA synchronizer aligns the frames of all bases (FS) and sends them by HTTP to the VFA analyzer, which detects AprilTags and gazes and renders them on the frames (FA); a VLM server grounds, captions and classifies each person (VLP); the synchronizer uploads the results to InfluxDB](../../img/video_frame_analyzer.png)

1. **Frame capture**: each VFA base takes a frame from its camera every `keyframe_interval` seconds and announces it over MQTT, with its base id, angle name, frame path and capture time.
2. **Frame synchronization**: the VFA synchronizer aligns the frames of all bases in time into one frame set. It sends the set to the VFA server for the action labels (`/vllm`, with the participant and angle descriptions), for the pose and the gaze (`/vllm/features`, one request for both), or both.
3. **Frame analysis**: the VFA server answers each request. For the action labels it draws the AprilTags and gaze lines on the frames and asks a VLM; for the pose and the gaze it runs the pose and gaze models.
4. **Storage**: the synchronizer writes the answers to InfluxDB as `vfa_action` and `vfa_features` events ([Databases](../../database.md#influxdb)). It also sends the features to the bases, which draw them on their live windows.

Joining these events with speech and positions, window by window, is analysis rather than capture: see [Window features](../../analytics/window_features.md).

## Components

| Component | Runs on | Started with | Environment |
|---|---|---|---|
| VFA Base, one per camera angle | base station | `mmla vfa-base` | conda env `vfa-base` |
| VFA Synchronizer, one per session | base station | `mmla vfa-sync` | conda env `vfa-base` |
| VFA Server, the frame analyzer | GPU server | `docker compose` | image built from `docker/` |
| MLLM Server, a local VLM for the action labels (optional) | GPU server | `vllm serve` | conda env `vfa-vllm` |

The management console starts all four from cards under `Launcher → Pipelines → VFA`; see [Run VFA](run.md).

## What you need

- **Cameras**: one per viewing angle, as a USB camera, a stream through the Stream Server, a Lab Streaming Layer stream or a recorded file ([Input sources](configuration.md#input-sources)).
- **AprilTag badges**: each participant wears one on the chest. The tag is how the pose and the action labels know who is who.
- **A GPU server** for the VFA Server, with the NVIDIA driver and the NVIDIA container toolkit ([Docker](../../docker.md)).
- **A VLM**, only for the action labels: a local one served by the **MLLM Server** card (the default), or a cloud API ([Model backends](action-labels.md#model-backends)).
- **The system services**: MQTT, Redis, MongoDB and InfluxDB, and the Stream Server for streamed cameras ([Quickstart](../../quickstart.md#one-time-setup)).
- **The `vfa-base` environment** on every base station. Create it from the console's **Environment** tab, or by hand:

    ```bash
    conda create -n vfa-base python=3.10 -y
    conda activate vfa-base
    pip install -e '.[vfa-base]'
    ```

    For an `lsl` source, add `pip install pylsl==1.17.6` and `conda install -c conda-forge liblsl=1.16.2`.

## Pages in this guide

- [Run VFA](run.md): set up once, run every session, run from the command line, replay recordings.
- [Action labels](action-labels.md): the action coding scheme, the prompts and the model backends.
- [Pose and gaze](pose-and-gaze.md): the features endpoint, skeletons and identity, tracking, gaze targets.
- [Configuration](configuration.md): every key of the base and server configs.
- [Human coding interface](coding_interface.md): code frames by hand with the same labels.
