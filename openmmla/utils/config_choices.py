"""The values a few model and backend settings can take, which the services
read and the console's Config tab offers as dropdowns (openmmla/tui/schema/
loader.py, KEY_CHOICES). A list of plain values, or of (label, value) pairs.
Kept free of other imports, so that both sides can read it."""

# the VLMs behind the VFA server's action labels (VLLMFrameAnalyzer.backend), each with a
# block of its own in the config; the first is the default
VFA_BACKENDS = ('vllm', 'ollama', 'llamacpp', 'openai', 'gemini', 'qwen', 'deepseek', 'grok', 'zhipuai', 'intern')
DEFAULT_VFA_BACKEND = 'vllm'
# the self-hosted backends: a block of theirs may leave the api key out, and EMPTY is sent,
# the key the console's MLLM Server card starts vllm serve with (Ollama and a llama.cpp
# server started without a key take any). The others are cloud APIs, which upload the frames
LOCAL_VFA_BACKENDS = ('vllm', 'ollama', 'llamacpp')
LOCAL_API_KEY = 'EMPTY'

# the gaze model behind the gaze lines and the features' gazes (VLLMFrameAnalyzer.gaze_backend)
GAZE_BACKENDS = ('page', 'gazelle')
# their checkpoints (VLLMFrameAnalyzer.gaze_model): PaGE's on Hugging Face, Gaze-LLE's
# torch.hub entry points with the in/out-of-frame head the features threshold. A checkpoint
# loads with its own backend only: a gazelle_* one needs gaze_backend gazelle
GAZE_MODELS = (
    ('Octopus1/page-vitb  (page, its default)', 'Octopus1/page-vitb'),
    ('Octopus1/page-vits  (page, a third of the compute)', 'Octopus1/page-vits'),
    ('Octopus1/page-vitsplus  (page, between vits and vitb)', 'Octopus1/page-vitsplus'),
    ('Octopus1/page-vithplus  (page, the 840M teacher)', 'Octopus1/page-vithplus'),
    ('gazelle_dinov2_vitl14_inout  (gazelle, its default)', 'gazelle_dinov2_vitl14_inout'),
    ('gazelle_dinov2_vitb14_inout  (gazelle)', 'gazelle_dinov2_vitb14_inout'),
)

# the Ultralytics pose weights the frame analyzer's ultralytics (8.4.157) fetches by name
# (VLLMFrameAnalyzer.features.pose_model); a local .pt of 17 COCO keypoints works too
POSE_MODELS = tuple(f'{family}{size}-pose.pt' for family in ('yolo26', 'yolo11', 'yolov8') for size in 'nsmlx')

# the end-to-end prompt variants (the keys of PROMPT_PROFILES in openmmla/services/vfa/
# prompt_profiles.py, kept in step by a test) and the image detail a vision model is asked for
PROMPT_PROFILES = ('cot', 'baseline', 'baseline_no_pre')
IMAGE_DETAILS = ('auto', 'low', 'high')

# the pyannote pipelines the speech transcriber's WhisperX diarizes with
# (SpeechTranscriber.local.diarize_model); both gated on huggingface.co. community-1 needs
# pyannote.audio 4: it is the default of the speech transcriber image's WhisperX 3.8
DIARIZE_MODELS = (
    ('pyannote/speaker-diarization-community-1  (the image\'s WhisperX default)',
     'pyannote/speaker-diarization-community-1'),
    ('pyannote/speaker-diarization-3.1', 'pyannote/speaker-diarization-3.1'),
)
