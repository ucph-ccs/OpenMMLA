import json
import os
import re
import threading
from io import BytesIO
from PIL import Image
from typing import Dict, Any, cast

import cv2
import numpy as np
import torch
from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.services.vfa.features import frame_features
from openmmla.services.vfa.prompt_profiles import DEFAULT_PROMPT_PROFILE, profile_template_files
from openmmla.services.vfa.schema_loader import load_vfa_action_schema
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.gaze import detect_gaze, gaze_backend_name, load_gaze_backend
from openmmla.utils.video.image import encode_image_base64, load_image

SUPPORTED_BACKENDS = (
    'ollama', 'vllm', 'openai', 'qwen', 'gemini',
    'deepseek', 'llamacpp', 'grok', 'zhipuai', 'intern',
)

# what the console tells the user to do about a config field it cannot supply
_FILL_IN_HINT = (
    "Fill it in the management console (Launcher -> VFA Server -> Config, with Host set to the "
    "machine that runs the service), then start VFA Server again. The console strips unfilled "
    "<...> placeholders when it saves, so a field nobody filled leaves no line in the file."
)


def _mask_secret(value) -> str:
    """what a secret may look like in a log: docker logs of this container are
    shown in the console's log pane, where the whole key would be readable."""
    text = str(value or "")
    return f"{text[:4]}...{text[-4:]}" if len(text) >= 12 else "(set)" if text else "(unset)"


def _is_unfilled(value) -> bool:
    """an <...> placeholder the user never replaced counts as no value at all."""
    text = str(value or "").strip()
    return text.startswith("<") and text.endswith(">")


def _text(value, default: str) -> str:
    """a config text, or `default` when it is missing, empty or an unfilled placeholder."""
    text = str(value or "").strip()
    return default if not text or _is_unfilled(text) else text


def _number(value, default: float) -> float:
    """a config number, or `default` when it is missing, unfilled or not a number."""
    if value is None or _is_unfilled(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool) -> bool:
    """a config flag: true/false, 1/0, yes/no, on/off; `default` for none or a placeholder."""
    if value is None or _is_unfilled(value) or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _json_value(text, default):
    """a JSON form field, or `default` when it is missing or not JSON."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _zones_for(zones, angle: str) -> dict:
    """the zones a frame's gaze targets are looked for in: `zones` holds {name: polygon}
    entries for every angle and {angle: {name: polygon}} entries for one; a frame gets the
    former plus its own angle's."""
    if not isinstance(zones, dict) or not zones:
        return {}
    every = {name: polygon for name, polygon in zones.items() if not isinstance(polygon, dict)}
    own = zones.get(angle)
    return {**every, **own} if isinstance(own, dict) else every


class MultiAngleVLLMFrameAnalyzer(Server):
    """Multi-angle VLLM frame analyzer that processes multiple images captured simultaneously from different angles.
    It combines information from different views for more comprehensive analysis of individuals' activities,
    gaze direction, and hand positions with enhanced accuracy."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the Multi-angle VLM frame analyzer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()
        self._load_prompt_templates()

    def _config_where(self, *keys) -> str:
        """name a config field the way the user sees it: its path and its file."""
        path = ".".join(("VLLMFrameAnalyzer",) + tuple(str(k) for k in keys))
        return f"{path} in {self.config_path}"

    def _required_backend_value(self, backend_config: dict, key: str):
        """a backend field the analyzer cannot start without.

        Missing and unfilled are the same thing here, and both stop the start
        with a message naming the field: retrying cannot fix a config, so the
        container must fail on its own terms rather than die of a KeyError."""
        value = backend_config.get(key)
        if value is None or str(value).strip() == "" or _is_unfilled(value):
            raise ValueError(f"{self._config_where(self.backend, key)} is not set. {_FILL_IN_HINT}")
        return value

    def _setup_yaml(self):
        analyzer_config = self.config['VLLMFrameAnalyzer']  # type: ignore

        # Load processing options from config first (needed for conditional setup)
        self.april_tag_enabled = analyzer_config.get('april_tag', True)
        self.gaze_detect_enabled = analyzer_config.get('gaze_detect', True)
        
        # the gaze model: which backend (page = PaGE, the default; gazelle = Gaze-LLE from
        # torch.hub, also what a gazelle_* checkpoint name alone means) and which of its
        # checkpoints (empty: the backend's default)
        gaze_model = _text(analyzer_config.get('gaze_model'), '')
        self.gaze_model = gaze_model or None
        self.gaze_backend = gaze_backend_name(_text(analyzer_config.get('gaze_backend'), ''), gaze_model)
        # PaGE crops the head its second branch looks at from the face box widened by this much
        self.gaze_head_scale = _number(analyzer_config.get('gaze_head_scale'), None)

        # Only load families if AprilTag detection is enabled
        if self.april_tag_enabled:
            if 'families' not in analyzer_config:
                raise ValueError("AprilTag detection is enabled but 'families' parameter is missing from config")
            self.families = analyzer_config['families']
        else:
            self.families = None
            
        backend = analyzer_config.get('backend')
        if backend is None or _is_unfilled(backend):
            raise ValueError(f"{self._config_where('backend')} is not set. {_FILL_IN_HINT}")
        self.backend = str(backend).strip()
        self.end_to_end = analyzer_config.get('end_to_end', False)
        
        # Image detail setting for vision models (low/high/auto)
        self.image_detail = analyzer_config.get('image_detail', 'auto')

        # Get prompt templates directory (treat an unfilled <...> placeholder as unset)
        self.prompt_templates_dir = analyzer_config.get('prompt_templates_dir') or 'prompts'
        if str(self.prompt_templates_dir).strip().startswith('<') and \
                str(self.prompt_templates_dir).strip().endswith('>'):
            self.prompt_templates_dir = 'prompts'
        if not os.path.isabs(self.prompt_templates_dir):
            self.prompt_templates_dir = os.path.join(self.project_dir, self.prompt_templates_dir)
        if not os.path.exists(self.prompt_templates_dir):
            raise FileNotFoundError(f"Prompt templates directory not found: {self.prompt_templates_dir}")
        self.logger.info(f"Prompt templates directory: {self.prompt_templates_dir}")

        # Prompt profile selects which end-to-end template variant is loaded
        # (cot | baseline | baseline_no_pre); see openmmla/services/vfa/prompt_profiles.py
        self.prompt_profile = str(analyzer_config.get('prompt_profile', DEFAULT_PROMPT_PROFILE))
        self.logger.info(f"Prompt profile: {self.prompt_profile}")


        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{self.backend}' in {self.config_path}. "
                f"Use one of: {', '.join(SUPPORTED_BACKENDS)}."
            )
        backend_config = analyzer_config.get(self.backend)
        if not isinstance(backend_config, dict):
            raise ValueError(
                f"{self._config_where(self.backend)} has no settings. "
                f"{_FILL_IN_HINT}"
            )

        self.api_key = self._required_backend_value(backend_config, 'api_key')
        self.vlm_model = self._required_backend_value(backend_config, 'vlm_model')
        self.llm_model = self._required_backend_value(backend_config, 'llm_model')
        self.vlm_base_url = backend_config.get('vlm_base_url', None)
        self.llm_base_url = backend_config.get('llm_base_url', None)

        schema = load_vfa_action_schema(analyzer_config, self.project_dir)
        self.action_schema_name = schema.schema_name
        self.action_schema_path = schema.schema_path
        self.action_definitions_dict = schema.action_definitions
        self.action_definitions = '\n'.join(
            [f"'{key}': {value}" for key, value in self.action_definitions_dict.items()])
        self.decision_process = schema.decision_process

        # Load VLM extra body from config
        self.vlm_extra_body = backend_config.get('VLMExtraBody', {})
        self.llm_extra_body = backend_config.get('LLMExtraBody', {})
        
        self.temperature = backend_config.get('temperature', None)
        self.top_p = backend_config.get('top_p', None)
        self.temperature = float(self.temperature) if self.temperature is not None else None
        self.top_p = float(self.top_p) if self.top_p is not None else None

        self.logger.info(f"API Key: {_mask_secret(self.api_key)}")
        self.logger.info(f"VLM Model: {self.vlm_model}")
        self.logger.info(f"LLM Model: {self.llm_model}")
        self.logger.info(f"Temperature: {self.temperature if self.temperature is not None else 'not configured'}")
        self.logger.info(f"Top-p: {self.top_p if self.top_p is not None else 'not configured'}")
        self.logger.info(f"VLM Base URL: {self.vlm_base_url}")
        self.logger.info(f"LLM Base URL: {self.llm_base_url}")
        self.logger.info(f"End-to-End: {self.end_to_end}")
        self.logger.info(f"Action Schema: {self.action_schema_name} ({self.action_schema_path})")
        self.logger.info(f"AprilTag Detection: {self.april_tag_enabled}")
        self.logger.info(f"Gaze Detection: {self.gaze_detect_enabled}")

        # /features: the persons of a frame as skeletons (a pose model), the AprilTag each
        # wears, their head yaw and their gaze (the gaze model above), as geometry
        features_config = analyzer_config.get('features') or {}
        if not isinstance(features_config, dict):
            features_config = {}
        self.features_enabled = _as_bool(features_config.get('enabled'), True)
        self.pose_model = _text(features_config.get('pose_model'), 'yolo26n-pose.pt')
        weights_dir = _text(features_config.get('weights_dir'), 'weights')
        self.pose_weights_dir = weights_dir if os.path.isabs(weights_dir) else os.path.join(self.project_dir, weights_dir)
        self.pose_confidence = _number(features_config.get('pose_confidence'), 0.25)
        self.keypoint_confidence = _number(features_config.get('keypoint_confidence'), 0.3)
        self.features_inout_threshold = _number(features_config.get('inout_threshold'), 0.5)
        # whether the features come with the gaze model's gazes (the pose alone is faster)
        self.features_gaze = _as_bool(features_config.get('gaze'), True)
        self.logger.info(f"Features endpoint: {self.features_enabled}"
                         f"{f' (pose model {self.pose_model} under {self.pose_weights_dir})' if self.features_enabled else ''}")

    def _load_prompt_templates(self):
        """Load prompt templates from files in the prompt_templates_dir."""
        # Initialize with default values in case files are not found
        self.multi_angle_end_system_prompt_template = ""
        self.multi_angle_end_user_prompt_template = ""
        self.multi_angle_vlm_system_prompt_template = ""
        self.multi_angle_vlm_user_prompt_template = ""
        self.multi_angle_llm_system_prompt_template = ""
        self.multi_angle_llm_user_prompt_template = ""

        # Template files are selected by the configured prompt profile
        # (cot | baseline | baseline_no_pre) plus the fixed two-step templates.
        template_files = profile_template_files(self.prompt_profile)

        try:
            if os.path.exists(self.prompt_templates_dir):
                for filename, attr_name in template_files.items():
                    filepath = os.path.join(self.prompt_templates_dir, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            setattr(self, attr_name, f.read())
                            self.logger.info(f"Loaded prompt template from {filepath}")
            else:
                self.logger.warning(f"Prompt templates directory not found: {self.prompt_templates_dir}")
        except Exception as e:
            self.logger.error(f"Error loading prompt templates: {e}")

    @staticmethod
    def _load_face_detector():
        """RetinaFace built and warmed up now, before any request, and answering one call at a
        time. Its model is a lazy singleton that the first call builds, downloading the weights
        on the way; two requests arriving together (the action overlay and a features request,
        the moment a session starts) each built one in the gevent worker, and the model that
        survived answered symbolic tensors for the life of the worker: no faces, no gazes,
        nothing but a line in the server log."""
        RetinaFace.build_model()
        # the first call traces the graph and claims the GPU memory; better now than in a request
        RetinaFace.detect_faces(np.zeros((320, 320, 3), dtype=np.uint8))
        lock = threading.Lock()

        def detect_faces(image):
            with lock:
                return RetinaFace.detect_faces(image)

        return detect_faces

    def _setup_gaze(self):
        """the face detector and the gaze backend, both before the first request"""
        # Conditionally setup gaze detection objects: the face detector and the gaze backend
        self.gaze = None
        self.face_detector = None
        if self.gaze_detect_enabled:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.face_detector = self._load_face_detector()
                self.logger.info("Face detector loaded: RetinaFace")
            except Exception as e:
                self.logger.error(f"Error loading the face detector (RetinaFace): {e}")
            try:
                self.gaze = load_gaze_backend(self.gaze_backend, self.gaze_model, device, self.gaze_head_scale)
                self.gaze_model = self.gaze.model_name or self.gaze_model
                self.device = device
                self.logger.info(f"Gaze model loaded on {device}: {self.gaze_backend} {self.gaze_model}")
            except Exception as e:
                self.logger.error(f"Error loading the gaze model ({self.gaze_backend} {self.gaze_model or ''}): {e}")
                self.gaze = None
                self.device = None
            if self.face_detector is None and self.gaze is not None:
                # a gaze needs a face first; without the detector the gaze model would only ever see nothing
                self.logger.error("Gaze detection is off: the face detector did not load")
                self.gaze = None
        else:
            self.device = None
            self.logger.info("Gaze detection disabled - models not initialized")

    def _setup_objects(self):
        # Conditionally setup AprilTag detector
        if self.april_tag_enabled and self.families:
            self.detector = Detector(families=self.families, nthreads=4)
            self.logger.info(f"AprilTag detector initialized with families: {self.families}")
        else:
            self.detector = None
            self.logger.info("AprilTag detection disabled - detector not initialized")
        
        self._setup_gaze()

        # the pose model of /features, fetched once at start and never during a request; a
        # model that cannot be loaded makes /features answer 503 with the reason
        self.pose_estimator = None
        self.pose_error = None
        if self.features_enabled:
            try:
                from openmmla.utils.video.pose import PoseEstimator
                self.pose_estimator = PoseEstimator(
                    self.pose_model, weights_dir=self.pose_weights_dir,
                    device='cuda' if torch.cuda.is_available() else 'cpu', confidence=self.pose_confidence)
                count = self.pose_estimator.keypoints_count
                if count is not None and count != 17:
                    raise ValueError(f"pose model {self.pose_model} answers {count} keypoints per person, "
                                     f"not the 17 of COCO that /features reads")
                self.logger.info(f"Pose model loaded: {self.pose_estimator.weights_path}")
            except Exception as e:
                self.pose_error = f"{type(e).__name__}: {e}"
                self.logger.error(f"Pose model {self.pose_model} could not be loaded ({self.pose_error}): "
                                  f"/features answers 503 until it can. The vfa-server extra installs "
                                  f"ultralytics, and the weights are fetched into {self.pose_weights_dir}.")

        # Setup VLM/LLM clients (required for analysis)
        if self.backend == 'zhipuai':
            try:
                from zai import ZhipuAiClient
            except ImportError as e:
                raise ImportError(
                    "ZhipuAI backend requires the optional 'zai' package. "
                    "Install it before using backend: zhipuai."
                ) from e

            self.vlm_client = ZhipuAiClient(
                api_key=self.api_key
            )
            self.llm_client = ZhipuAiClient(
                api_key=self.api_key
            )
        else:
            self.vlm_client = OpenAI(
                api_key=self.api_key,
                base_url=self.vlm_base_url
            )
            self.llm_client = OpenAI(
                api_key=self.api_key,
                base_url=self.llm_base_url
            )


    def describe(self) -> dict:
        """What this analyzer runs with, for GET /vllm/info: the backend, its models and
        their addresses, the prompt profile and the action schema, as resolved at start
        (see openmmla.utils.session_provenance). The api key stays here."""
        import os
        from openmmla.utils.session_provenance import plain
        info = {
            'service': self.__class__.__name__,
            'backend': self.backend,
            'vlm_model': self.vlm_model,
            'llm_model': self.llm_model,
            'vlm_base_url': self.vlm_base_url,
            'llm_base_url': self.llm_base_url,
            'end_to_end': bool(self.end_to_end),
            'image_detail': self.image_detail,
            'prompt_profile': self.prompt_profile,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'vlm_extra_body': plain(self.vlm_extra_body or {}),
            'llm_extra_body': plain(self.llm_extra_body or {}),
            'action_schema': self.action_schema_name,
            'action_schema_file': os.path.basename(str(self.action_schema_path)) if self.action_schema_path else None,
            'families': self.families,
            'april_tag': bool(self.april_tag_enabled),
            'gaze_detect': bool(self.gaze_detect_enabled),
            'gaze_backend': self.gaze_backend if self.gaze_detect_enabled else None,
            'gaze_model': self.gaze_model if self.gaze_detect_enabled else None,
            'gaze_loaded': self.gaze is not None,
            'face_detector_loaded': self.face_detector is not None,
            'gaze_head_scale': getattr(self.gaze, 'head_scale', None),
            # the features endpoint: what its geometry ran with, since nothing else records
            # the server's config (openmmla.utils.session_provenance reads this answer)
            'features': {'enabled': bool(self.features_enabled),
                         'pose_model': self.pose_model if self.features_enabled else None,
                         'pose_loaded': self.pose_estimator is not None,
                         'weights_path': getattr(self.pose_estimator, 'weights_path', None),
                         'gaze': bool(self.features_gaze) if self.features_enabled else None,
                         'pose_confidence': self.pose_confidence if self.features_enabled else None,
                         'keypoint_confidence': self.keypoint_confidence if self.features_enabled else None,
                         'inout_threshold': self.features_inout_threshold if self.features_enabled else None},
        }
        return {name: value for name, value in info.items() if value is not None}

    def process_features(self):
        """POST /vllm/features: for every image, the persons in it as skeletons with the AprilTag
        each wears, their head yaw and their gaze (point, in-frame probability, what it lands on),
        as geometry on the image (openmmla.services.vfa.features). Form fields: `images` (one or
        more), `angles` (JSON list, one name per image), `session_id`, `zones` (JSON: {name:
        polygon} for every image, or {angle: {name: polygon}}; a polygon in pixels, or in [0, 1]),
        `inout_threshold` (below it a gaze is out of frame), `keypoints` (false leaves the skeletons
        out of the answer, for a study that wants the derived features alone), `gaze` (false skips
        the gaze model for this request; the config's features.gaze otherwise)."""
        if not self.features_enabled:
            return jsonify({'error': 'the features endpoint is off (VLLMFrameAnalyzer.features.enabled)'}), 503
        if self.pose_estimator is None:
            return jsonify({'error': f'the pose model is not loaded ({self.pose_error})'}), 503
        try:
            session_id = request.values.get('session_id')
            angles = _json_value(request.values.get('angles'), [])
            if not isinstance(angles, list):
                angles = []
            zones = _json_value(request.values.get('zones'), {})
            if zones and not isinstance(zones, dict):
                return jsonify({'error': 'zones must be a JSON object: {name: polygon}, or {angle: {name: polygon}}'}), 400
            inout_threshold = _number(request.values.get('inout_threshold'), self.features_inout_threshold)
            keypoints = _as_bool(request.values.get('keypoints'), True)
            gaze = _as_bool(request.values.get('gaze'), self.features_gaze)
            image_files = request.files.getlist('images')
            if not image_files:
                return jsonify({'error': 'No images provided in request'}), 400

            frames = []
            for i, image_file in enumerate(image_files):
                angle = str(angles[i]) if i < len(angles) else f"perspective_{i + 1}"
                try:
                    frames.append(self._frame_features(image_file.read(), angle, _zones_for(zones, angle),
                                                       inout_threshold, keypoints, gaze))
                except ValueError as e:
                    # a frame that is not an image, or a zone that is not a polygon: the client's
                    # to fix, so no retry is asked for
                    return jsonify({'error': f"{image_file.filename or angle}: {e}"}), 400
            self.logger.info(f"Features for {session_id}: {len(frames)} frames, "
                             f"{sum(len(frame['persons']) for frame in frames)} persons")
            return jsonify({'frames': frames, 'pose_model': self.pose_estimator.model_name,
                            'gaze': bool(gaze and self.gaze is not None)}), 200
        except Exception as e:
            self.logger.error("Exception during feature extraction", exc_info=True)
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    def _frame_features(self, image_bytes: bytes, angle: str, zones: dict, inout_threshold: float,
                        keypoints: bool = True, gaze: bool = True) -> dict:
        """the features of one frame: its AprilTags (centres in pixels from the top-left corner),
        its persons from the pose model, its faces and gazes from the gaze model, put together."""
        image = load_image(image_bytes)
        height, width = image.shape[:2]
        tags = {}
        if self.detector is not None:
            for tag in self.detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):
                centre = getattr(tag, 'center', None)
                if centre is None:
                    centre = tag.corners.mean(axis=0)
                tags[int(tag.tag_id)] = (float(centre[0]), float(centre[1]))
        persons = self.pose_estimator.detect(image)
        faces, gaze_error = [], None
        if gaze and self.gaze is not None:
            try:
                gaze_results, _ = detect_gaze(
                    image_input=image_bytes, face_detector=self.face_detector, backend=self.gaze,
                    device=self.device if self.device else 'cpu',
                    normalize_bbox=False, normalize_target=False, render=False, show=False,
                    inout_thresh=inout_threshold, render_heatmap=False, save=False, raise_errors=True)
                faces = [{'bbox': result['face_bbox'], 'gaze_point': result['gaze_target'], 'inout': result['inout_score']}
                         for result in gaze_results]
            except Exception as e:
                # the frame keeps its skeletons; the answer says the gazes are missing for a reason
                gaze_error = f"{type(e).__name__}: {e}"
                self.logger.error(f"Gaze detection failed on a {angle} frame: {gaze_error}", exc_info=True)
        frame = frame_features(persons, tags, faces, zones, width, height, angle,
                               min_confidence=self.keypoint_confidence, inout_threshold=inout_threshold,
                               keypoints=keypoints)
        if gaze_error:
            frame['gaze_error'] = gaze_error
        return frame

    def process_request(self):
        """Process multiple images from different angles with contextual awareness."""
        try:
            session_id = request.values.get('session_id')
            
            # Parse JSON parameters with fallback to defaults
            try:
                participant_descriptions = json.loads(request.values.get('participant_descriptions', '{}'))
            except (json.JSONDecodeError, TypeError):
                participant_descriptions = {}
                
            try:
                angles = json.loads(request.values.get('angles', '[]'))
            except (json.JSONDecodeError, TypeError):
                angles = []
                
            try:
                angle_descriptions = json.loads(request.values.get('angle_descriptions', '[]'))
            except (json.JSONDecodeError, TypeError):
                angle_descriptions = []

            image_files = request.files.getlist('images')
            if not image_files:
                return jsonify({'error': 'No images provided in request'}), 400
                
            self.logger.info(
                f"Starting multi-angle analysis for {session_id} with {len(image_files)} images from angles: {angles}")

            # Process each image (apriltag and gaze detection)
            processed_images = {}
            for i, image_file in enumerate(image_files):
                image_bytes = image_file.read()
                original_filename = image_file.filename  # get the original filename
                
                # Use provided angle or fallback to generic
                if i < len(angles):
                    angle = angles[i]
                else:
                    angle = f"perspective_{i + 1}"  # Fallback to generic angle
                
                # Use provided angle description or create generic one
                if i < len(angle_descriptions):
                    angle_description = angle_descriptions[i]
                else:
                    angle_description = f"Image from {angle} perspective"
                
                self.logger.info(f"Processing image from {angle} perspective: {angle_description}")
                
                # Process the image with AprilTag detection and gaze detection
                processed_image = self._process_single_image(
                    image_bytes, angle, angle_description, original_filename, session_id, 
                    april_tag=self.april_tag_enabled, gaze_detect=self.gaze_detect_enabled
                )
                processed_images[angle] = processed_image

            # Analyze all processed images together (end-to-end or two-step approach)
            if self.end_to_end:
                # End-to-end approach: VLM does both observation and classification for multiple images
                messages = self._create_end_to_end_messages(processed_images, participant_descriptions)
                vlm_response = self._process_with_vlm(messages)

                # Extract observations, classifications, and justifications
                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': vlm_response.get('classifications', {}),
                    'justifications': vlm_response.get('justifications', {})
                }
            else:
                # Two-step approach: VLM for observations, LLM for classification
                vlm_messages = self._create_vlm_messages(processed_images, participant_descriptions)
                vlm_response = self._process_with_vlm(vlm_messages)

                # Process with LLM for text classification
                llm_messages = self._create_llm_messages(vlm_response)
                llm_result = self._process_with_llm(llm_messages)

                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': llm_result.get('classifications', {}),
                    'justifications': llm_result.get('justifications', {})
                }

            self.logger.info(f"Finished multi-angle analyzing for {session_id}.")
            return jsonify(result), 200

        except Exception as e:
            self.logger.error("Exception during processing images", exc_info=True)
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    def _process_single_image(self, image_bytes: bytes, angle: str, angle_description: str, 
                             original_filename: str | None = None, session_id: str | None = None,
                             april_tag: bool = True, gaze_detect: bool = True) -> Dict[str, Any]:
        """Process a single image with AprilTag detection and gaze detection.
        
        Args:
            image_bytes: Raw image bytes
            angle: The angle label for this image
            angle_description: Description of what this angle shows
            original_filename: Original filename from the request (optional)
            session_id: Session ID for organizing temp files (optional)
            april_tag: Whether to perform AprilTag detection (default: True)
            gaze_detect: Whether to perform gaze detection (default: True)
            
        Returns:
            dict: Processed image information including base64 encoding and metadata
        """
        # Step 1: Conditionally detect AprilTags and render them on the image
        if april_tag and self.detector is not None:
            tag_pos, apriltag_image = detect_apriltags(
                image_bytes,
                self.detector,
                normalize=True,
                render=True,
                show=False,
                save=False
            )

            # If AprilTags were detected and rendered, use that image for further processing
            if tag_pos:
                image_bytes = pil_image_to_bytes(apriltag_image)

        # Step 2: Conditionally detect gaze on the processed image if gaze detection is available
        if gaze_detect and self.gaze is not None:
            device = self.device if self.device is not None else 'cpu'
            
            gaze_results, rendered_image = detect_gaze(
                image_input=image_bytes,
                face_detector=self.face_detector,
                backend=self.gaze,
                device=device,
                normalize_bbox=True,
                normalize_target=True,
                render=True,
                show=False,
                inout_thresh=0.5,
                render_heatmap=False,
                save=False
            )
            if gaze_results:
                image_bytes = pil_image_to_bytes(rendered_image)

        # Step 3: Save the final image (processed or original)
        # create session-specific temp directory and unique save path
        if session_id:
            temp_session_dir = os.path.join(self.project_dir, 'temp', session_id)
        else:
            temp_session_dir = os.path.join(self.project_dir, 'temp', 'default')
        
        # ensure the session directory exists
        os.makedirs(temp_session_dir, exist_ok=True)
        self.logger.debug(f"Using temp directory: {temp_session_dir}")
        
        # create unique filename using original filename and angle
        if original_filename:
            # use original filename (e.g., "1754398113.456.jpg") + angle for uniqueness
            base_name = os.path.splitext(original_filename)[0]  # remove extension
            unique_filename = f'{base_name}_{angle}.png'
        else:
            # fallback to angle only if no original filename
            unique_filename = f'{angle}.png'
        
        save_path = os.path.join(temp_session_dir, unique_filename)
        image = Image.open(BytesIO(image_bytes))
        image.save(save_path)
        self.logger.debug(f"Saved image to: {save_path}")

        # Create a result dictionary with the processed image and metadata
        image_b64 = encode_image_base64(image_bytes)

        return {
            "image_b64": image_b64,
            "angle": angle,
            "angle_description": angle_description
        }


    def _create_end_to_end_messages(self, processed_images, participant_descriptions={}):
        """Generate a context-aware prompt message for end-to-end approach with multiple images.
        
        Args:
            processed_images: Dictionary containing processed images from different angles
            participant_descriptions: Dictionary of participant descriptions (tag_id -> description)
            
        Returns:
            list: Messages for the VLM
        """
        # Create system prompt
        system_prompt = self.multi_angle_end_system_prompt_template.replace(
            "{{num_perspectives}}", str(len(processed_images))
        )

        # Format angle descriptions
        angle_descriptions = []
        for angle, image_data in processed_images.items():
            angle_descriptions.append(f"- **{angle}**: {image_data['angle_description']}")

        # Load template and replace variables
        template = self.multi_angle_end_user_prompt_template
        template = template.replace("{{num_perspectives}}", str(len(processed_images)))
        template = template.replace("{{angle_descriptions}}", "\n".join(angle_descriptions))
        template = template.replace("{{participant_descriptions}}", json.dumps(participant_descriptions, indent=2) if participant_descriptions else "")
        template = template.replace("{{action_definitions}}", self.action_definitions)
        template = template.replace("{{decision_process}}", self.decision_process)

        # Create message content with multiple images
        user_prompt = [{"type": "text", "text": template}]

        # Add all the images to the content - fixing the type error with proper casting
        for angle, image_data in processed_images.items():
            user_prompt.append(cast(Dict[str, str], {
                "type": "image_url",
                "image_url": {
                    "url": image_data["image_b64"],
                    "detail": self.image_detail
                }
            }))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def _create_vlm_messages(self, processed_images, participant_descriptions={}):
        """Generate a context-aware prompt message for VLM with multiple images.
        
        Args:
            processed_images: Dictionary containing processed images from different angles
            participant_descriptions: Dictionary of participant descriptions (tag_id -> description)
            
        Returns:
            list: Messages for the VLM
        """
        # Create system prompt
        system_prompt = self.multi_angle_vlm_system_prompt_template.replace(
            "{{num_perspectives}}", str(len(processed_images))
        )

        # Format angle descriptions
        angle_descriptions = []
        for angle, image_data in processed_images.items():
            angle_descriptions.append(f"- **{angle}**: {image_data['angle_description']}")

        # Load template and replace variables
        template = self.multi_angle_vlm_user_prompt_template
        template = template.replace("{{num_perspectives}}", str(len(processed_images)))
        template = template.replace("{{angle_descriptions}}", "\n".join(angle_descriptions))
        template = template.replace("{{participant_descriptions}}", json.dumps(participant_descriptions, indent=2) if participant_descriptions else "")

        # Create message content with multiple images
        user_prompt = [{"type": "text", "text": template}]

        # Add all the images to the content - fixing the type error with proper casting
        for angle, image_data in processed_images.items():
            user_prompt.append(cast(Dict[str, str], {
                "type": "image_url",
                "image_url": {
                    "url": image_data["image_b64"],
                    "detail": self.image_detail
                }
            }))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def _create_llm_messages(self, vlm_response):
        """Generate a context-aware prompt message for LLM, asking the LLM to categorize actions based strictly on the provided
        description with consideration for practical context.
        
        Args:
            vlm_response: Response from the VLM containing observations
            
        Returns:
            list: Messages for the LLM
        """
        system_prompt = self.multi_angle_llm_system_prompt_template

        image_description = json.dumps(vlm_response.get('observations', {}), indent=2)

        # Load template and replace variables
        template = self.multi_angle_llm_user_prompt_template
        template = template.replace("{{image_description}}", image_description)
        template = template.replace("{{action_definitions}}", self.action_definitions)
        template = template.replace("{{decision_process}}", self.decision_process)

        user_prompt = [{"type": "text", "text": template}]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return messages

    def _process_with_vlm(self, messages):
        """Process the image with VLM API.
        
        Args:
            messages: Messages for the VLM
            
        Returns:
            dict: Response from the VLM
        """
        # build parameters dynamically - only include temperature and top_p if configured
        params = {
            "model": self.vlm_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "extra_body": self.vlm_extra_body,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
            
        vlm_response = self.vlm_client.chat.completions.create(**params)
        self.logger.debug(f"VLM response: {vlm_response}")
        
        result = vlm_response.choices[0].message.content
        if result:
            result = extract_json_from_codeblock(result)
            try:
                result_obj = json.loads(result)
                return result_obj
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON response: {result}")
                return {"error": "Failed to parse response from model"}
        else:
            return {"error": "Empty response from model"}

    def _process_with_llm(self, messages):
        """Process classification with LLM API.
        
        Args:
            messages: Messages for the LLM
            
        Returns:
            dict: Classification results from the LLM
        """
        # build parameters dynamically - only include temperature and top_p if configured
        params = {
            "model": self.llm_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "extra_body": self.llm_extra_body,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
            
        llm_response = self.llm_client.chat.completions.create(**params)
        self.logger.debug(f"LLM response: {llm_response}")
        llm_result = llm_response.choices[0].message.content
        if llm_result:
            llm_result = extract_json_from_codeblock(llm_result)
            try:
                llm_result_obj = json.loads(llm_result)
                return llm_result_obj
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON response from LLM: {llm_result}")
                return {"error": "Failed to parse response from model"}
        else:
            return {"error": "Empty response from model"}


def extract_json_from_codeblock(text: str) -> str:
    """Extracts JSON content from a JSON code block."""
    if not text:
        return "{}"

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def pil_image_to_bytes(pil_image, format='PNG'):
    """Convert a PIL Image to bytes."""
    img_byte_array = BytesIO()
    pil_image.save(img_byte_array, format=format)
    return img_byte_array.getvalue()
