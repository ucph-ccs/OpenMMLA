# Action labels

The VFA server can label what each person in a frame set is doing with one of five collaborative actions, using a vision-language model (VLM). Use the action labels when you want a semantic, coder-like account of activity; they are off by default ([Choosing the outputs](run.md#choosing-the-outputs)).

## How labelling works

For every frame set it is asked about, the server does three things:

1. **Marks the frames.** Each detected AprilTag is repainted as a black square with its id in white, each detected face gets a coloured box, and a gaze line points at the estimated gaze target. `in: 0.98` next to a face is the probability that its gaze target lies inside the frame.
2. **Builds a structured prompt** from templates, with the camera angles, the participant descriptions and the [action coding scheme](#action-coding-scheme).
3. **Asks a VLM**, local or in the cloud behind an OpenAI-compatible API, which identifies each person, describes them and classifies their action. It answers in JSON with `observations`, `classifications` and `justifications` per person id.

The synchronizer stores the answer as a `vfa_action` event, in its `action_recognition` field ([Databases](../../database.md#influxdb)). The method is described in the ICALT 2026 paper *Designing for Transparency: Gaze-Augmented Collaborative Action Recognition with Vision-Language Models*.

The server asks the model in one of two modes, set by `end_to_end` in the server config:

| `end_to_end` | Mode | Templates |
|---|---|---|
| `false` (default) | two steps: a VLM describes each person (`observations`), then an LLM classifies them from that description | `multi_angle_vlm_*` and `multi_angle_llm_*` |
| `true` | one step: the VLM describes and classifies in one answer | the `multi_angle_end_*` templates of the `prompt_profile` |

The marks need the server's `april_tag` and `gaze_detect` settings on, which they are unless set otherwise ([Server config](configuration.md#server-config)).

## Action coding scheme

Every participant in every frame set gets one of five mutually exclusive actions:

| Action | Definition |
|---|---|
| Communicating | Actively communicating with others: looking at work items (screen, documents, hardware) while clearly pointing at them, or looking at another person while gesturing or pointing. |
| Observing | Watching or monitoring without communicating or manipulating: looking at work items or people while the hands are resting, hovering without touching, or touching something other than what is being looked at. |
| Manipulating | Directly working with something: looking at a work item while at least one hand clearly touches the same item. |
| Idle-OffTask | Not engaged in the task: looking at personal items (phone, snacks) or outside the camera frame, whatever the hands do. |
| Unclear | The gaze or the hands cannot be seen well enough to tell, because of occlusion, blur or poor visibility. |

Human coders work from these descriptive definitions. The VLM receives a rule-based form of the same definitions: explicit conditions on gaze and hands, and a decision process applied top down.

- The rules live in `config/vfa/action_schemas.yml`, schema `collaboration_v1`, which is the file's `default_schema`.
- Edit them on the **Action Schema** tab of the VFA Server card.
- `action_schema` in the server config picks another schema of the file.

The same scheme is the default template of the [human coding interface](coding_interface.md), so machine and human codings use identical labels.

## Prompts

![Structure of the chain-of-thought prompt: the system prompt assigns the expert analyst persona; the user prompt gives the context (visual legend, camera setup, participants), then grounding (person identification), captioning (gaze, hands, position, clothing), classifying (definitions, decision process, rules) and the JSON response format](../../img/vfa_prompt_structure.png)

The prompt mirrors how a human coder works, as a chain of steps:

1. **Persona** (system prompt): the VLM acts as an expert multi-perspective lab activity analyst.
2. **Context**: a legend of the marks on the frames, the camera setup (`{{num_perspectives}}`, `{{angle_descriptions}}`) and the participant descriptions (`{{participant_descriptions}}`).
3. **Grounding**: identify each person by their AprilTag id, else by matching their appearance to the participant descriptions, else give them an id from 100 upwards.
4. **Captioning**: describe gaze focus, hand status (contact, hovering, inactive), position and clothing per person, citing the camera view that supports each observation.
5. **Classifying**: apply `{{action_definitions}}` and `{{decision_process}}` top down, stopping at the first match, with contact evidence where a rule requires contact.
6. **Formulating**: answer in a fixed JSON layout with `observations`, `classifications` and `justifications` per id.

In the two-step mode the VLM's templates carry steps 1 to 4, and the LLM's templates carry steps 5 and 6, with the VLM's observations as `{{image_description}}`.

### Prompt profiles

With `end_to_end: true`, `prompt_profile` picks the one-step templates:

| `prompt_profile` | What the VLM is asked |
|---|---|
| `cot` (default) | the chain-of-thought prompt above |
| `baseline` | direct classification, without the reasoning steps; the answer holds `classifications` only |
| `baseline_no_pre` | `baseline` without the legend of the marks: each person is identified by appearance alone, not by their AprilTag |

### Templates and placeholders

The templates are plain text files in `pipelines/vfa-server/prompts/` (the server's `prompt_templates_dir`). Edit them on the **Prompts** tab of the VFA Server card, which marks the templates in use.

| Placeholder | Filled with | Used by |
|---|---|---|
| `{{num_perspectives}}` | the number of frames in the set | VLM and one-step templates |
| `{{angle_descriptions}}` | what each camera's angle sees, from `Base.angle_config` | VLM and one-step templates |
| `{{participant_descriptions}}` | the descriptions of the session's participants | VLM and one-step templates |
| `{{action_definitions}}` | the labels of the action schema | LLM and one-step templates |
| `{{decision_process}}` | the decision process of the action schema | LLM templates, `cot` |
| `{{image_description}}` | the VLM's observations | LLM templates |

The participant descriptions come from the experiment group selected for the session (`config/experiments.yaml`, edited under **System Settings → Study → Experiments**).

!!! warning "Check the templates path"
    A missing templates directory stops the server at start. A missing single template is skipped silently and leaves that prompt empty, so check the path if the model starts receiving bare input.

## Model backends

The server talks to an OpenAI-compatible endpoint. Choose it with `VLLMFrameAnalyzer.backend` in `pipelines/vfa-server/config.yml` (a dropdown on the **Config** tab) and fill in the block of the same name ([Backend blocks](configuration.md#backend-blocks)). The template carries a block for every backend.

| `backend` | Where it runs | Notes |
|---|---|---|
| `vllm` (default) | the **MLLM Server** card, on your GPU server | one model for both steps; also what a config without `backend` gets |
| `ollama` | Ollama on your machine | install from <https://ollama.com/download> and pull a multimodal model, such as `ollama pull llava` |
| `llamacpp` | a llama.cpp `llama-server` | any multimodal GGUF model it serves |
| `openai`, `gemini`, `grok`, `zhipuai`, `intern` | cloud | vision-capable models for both steps; `zhipuai` also needs the `zai` package |
| `deepseek` | cloud | text models only, so only for the classification step of the two-step mode; the shipped `vlm_model`, `deepseek-reasoner`, does not accept images |
| `qwen` | cloud, through DashScope's OpenAI-compatible endpoint | the shipped `qwen2.5-72b-instruct` is text-only: set `vlm_model` to a `-vl` model |

- A `vllm`, `ollama` or `llamacpp` block may leave `api_key` out: `EMPTY` is sent, the key the MLLM Server card starts `vllm serve` with.
- A cloud backend needs its `api_key`, which the console stores encrypted on save, and it uploads the frames.

!!! note "A backend on the same machine"
    The frame analyzer runs in a container, so a VLM on the same machine is reached as `http://host.docker.internal:<port>/v1`, not `localhost` ([Docker](../../docker.md#a-vlm-server-on-the-same-host)).

### Local VLM with the MLLM Server

The **MLLM Server** card runs `vllm serve` in the `vfa-vllm` environment (`pip install -e '.[vfa-vllm-runtime]'`, Python 3.12), with the settings below. [Run VFA](run.md#once-per-deployment) gives the steps.

The server config's `vllm` block points at it as shipped:

| Key | Shipped value |
|---|---|
| `vlm_base_url`, `llm_base_url` | `http://host.docker.internal:8010/v1` |
| `vlm_model`, `llm_model` | `Qwen/Qwen3-VL-8B-Instruct` |
| `api_key` | `EMPTY` |

Change the block on the VFA Server card's **Config** tab, and **Save**, when the MLLM Server runs on another machine (`http://<host>:<port>/v1`), on another port, or serves another model. With `end_to_end: false` the classification goes to `llm_base_url` and `llm_model`, so keep them on the same server unless another one serves the LLM.

### MLLM Server settings

The MLLM Server card's **Config** tab edits `config/mllm_server.yml` on the console, whatever its **Host** selector says. Section `server`:

| Key | Shipped value | What it does |
|---|---|---|
| `model` | `Qwen/Qwen3-VL-8B-Instruct` | the Hugging Face model id `vllm serve` loads |
| `port` | `8010` | the OpenAI-compatible API port |
| `host` | `0.0.0.0` | the address vLLM binds |
| `dtype` | `auto` | vLLM's `dtype` |
| `max_model_len` | `16384` | the model's maximum context length |
| `limit_mm_per_prompt` | `{"image":4}` | images per request; must cover the cameras of a frame set, which go in one request |
| `gpu_memory_utilization` | `0.95` | the share of GPU memory vLLM takes |
| `api_key` | `EMPTY` | the key clients must send |

## Human coding

Ground truth for the action labels is coded by hand in the [human coding interface](coding_interface.md), a single HTML page in `pipelines/vfa-base/coding-interface/`. Coders load the frames the pipeline analyzed and the action template, and code every participant in every frame with the same labels. They export a JSON file whose windows mirror the pipeline's `action_recognition` output, so human and machine codings join on the frame timestamp and the participant id.

To collect frames, run the bases in `capture` mode, which always stores its frames, or in `live` mode with **Store Frames** on. The frames land in `artifacts/<session-id>/pipelines/vfa-base/<host>/real-time/runtime/<camera>_<base-id>/`, named `<unix-timestamp>.jpg`.
