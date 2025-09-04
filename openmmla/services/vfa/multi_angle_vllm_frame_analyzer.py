import json
import os
import re
from io import BytesIO
from PIL import Image
from typing import Dict, Any, cast

import torch
from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.gaze import detect_gaze
from openmmla.utils.video.image import encode_image_base64
from zai import ZhipuAiClient

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

    def _setup_yaml(self):
        analyzer_config = self.config['VLLMFrameAnalyzer']  # type: ignore

        # Load processing options from config first (needed for conditional setup)
        self.april_tag_enabled = analyzer_config.get('april_tag', True)
        self.gaze_detect_enabled = analyzer_config.get('gaze_detect', True)
        
        # Only load families if AprilTag detection is enabled
        if self.april_tag_enabled:
            if 'families' not in analyzer_config:
                raise ValueError("AprilTag detection is enabled but 'families' parameter is missing from config")
            self.families = analyzer_config['families']
        else:
            self.families = None
            
        self.backend = analyzer_config['backend']
        self.top_p = float(analyzer_config['top_p'])
        self.temperature = float(analyzer_config['temperature'])
        self.end_to_end = analyzer_config.get('end_to_end', False)
        
        # Image detail setting for vision models (low/high/auto)
        self.image_detail = analyzer_config.get('image_detail', 'auto')

        # Get prompt templates directory
        self.prompt_templates_dir = analyzer_config.get('prompt_templates_dir', 'prompts')
        if not os.path.isabs(self.prompt_templates_dir):
            self.prompt_templates_dir = os.path.join(self.project_dir, self.prompt_templates_dir)
        if not os.path.exists(self.prompt_templates_dir):
            raise FileNotFoundError(f"Prompt templates directory not found: {self.prompt_templates_dir}")
        self.logger.info(f"Prompt templates directory: {self.prompt_templates_dir}")


        if self.backend in ['ollama', 'vllm', 'openai', 'qwen', 'gemini', 'deepseek', 'llamacpp', 'grok', 'zhipuai']:
            backend_config = analyzer_config[self.backend]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.api_key = backend_config['api_key']
        self.vlm_model = backend_config['vlm_model']
        self.llm_model = backend_config['llm_model']
        self.vlm_base_url = backend_config.get('vlm_base_url', None)
        self.llm_base_url = backend_config.get('llm_base_url', None)

        # Load action definitions from config
        self.action_definitions_dict = analyzer_config['action_definitions']
        self.action_definitions = '\n'.join(
            [f"'{key}': {value}" for key, value in self.action_definitions_dict.items()])
        
        # Load decision priority from config
        self.decision_priority = analyzer_config.get('decision_priority', '')

        # Load VLM extra body from config
        self.vlm_extra_body = backend_config.get('VLMExtraBody', {})
        self.llm_extra_body = backend_config.get('LLMExtraBody', {})

        self.logger.info(f"API Key: {self.api_key}")
        self.logger.info(f"VLM Model: {self.vlm_model}")
        self.logger.info(f"LLM Model: {self.llm_model}")
        self.logger.info(f"VLM Base URL: {self.vlm_base_url}")
        self.logger.info(f"LLM Base URL: {self.llm_base_url}")
        self.logger.info(f"End-to-End: {self.end_to_end}")
        self.logger.info(f"AprilTag Detection: {self.april_tag_enabled}")
        self.logger.info(f"Gaze Detection: {self.gaze_detect_enabled}")

    def _load_prompt_templates(self):
        """Load prompt templates from files in the prompt_templates_dir."""
        # Initialize with default values in case files are not found
        self.multi_angle_end_system_prompt_template = ""
        self.multi_angle_end_user_prompt_template = ""
        self.multi_angle_vlm_system_prompt_template = ""
        self.multi_angle_vlm_user_prompt_template = ""
        self.multi_angle_llm_system_prompt_template = ""
        self.multi_angle_llm_user_prompt_template = ""

        # Define template files to load 
        template_files = {
            'multi_angle_end_system_prompt.txt': 'multi_angle_end_system_prompt_template',
            'multi_angle_end_user_prompt.txt': 'multi_angle_end_user_prompt_template',
            'multi_angle_vlm_system_prompt.txt': 'multi_angle_vlm_system_prompt_template',
            'multi_angle_vlm_user_prompt.txt': 'multi_angle_vlm_user_prompt_template',
            'multi_angle_llm_system_prompt.txt': 'multi_angle_llm_system_prompt_template',
            'multi_angle_llm_user_prompt.txt': 'multi_angle_llm_user_prompt_template',
        }

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

    def _setup_objects(self):
        # Conditionally setup AprilTag detector
        if self.april_tag_enabled and self.families:
            self.detector = Detector(families=self.families, nthreads=4)
            self.logger.info(f"AprilTag detector initialized with families: {self.families}")
        else:
            self.detector = None
            self.logger.info("AprilTag detection disabled - detector not initialized")
        
        # Conditionally setup gaze detection objects
        if self.gaze_detect_enabled:
            self.face_detector = RetinaFace.detect_faces
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                self.gazelle_model, self.gazelle_transform = torch.hub.load(
                    'fkryan/gazelle',
                    'gazelle_dinov2_vitl14_inout',
                    trust_repo=True
                )
                self.gazelle_model.eval()
                self.gazelle_model.to(device)
                self.device = device
                self.logger.info(f"Gazelle model loaded on {device}")
            except Exception as e:
                self.logger.error(f"Error loading Gazelle model: {e}")
                self.gazelle_model = None
                self.gazelle_transform = None
                self.device = None
        else:
            self.face_detector = None
            self.gazelle_model = None
            self.gazelle_transform = None
            self.device = None
            self.logger.info("Gaze detection disabled - models not initialized")

        # Setup VLM/LLM clients (required for analysis)
        if self.backend == 'zhipuai':
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

            # Process each image
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

            # Analyze all processed images together
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
                self.logger.info(f"VLM response: {vlm_response}")

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
        if gaze_detect and self.gazelle_model and self.gazelle_transform:
            device = self.device if self.device is not None else 'cpu'
            
            gaze_results, rendered_image = detect_gaze(
                image_input=image_bytes,
                face_detector=self.face_detector,
                gazelle_model=self.gazelle_model,
                gazelle_transform=self.gazelle_transform,
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

    def _format_participant_descriptions(self, participant_descriptions):
        """Format participant descriptions into readable text for prompts.
        
        Args:
            participant_descriptions: Dictionary of participant descriptions (tag_id -> description)
            
        Returns:
            str: Formatted participant descriptions
        """
        if not participant_descriptions:
            return ""

        description_text = "### Known Participants Reference:\n"
        description_text += "The following people may appear in the images. Use this information to help identify them by AprilTag ID:\n\n"

        # participant descriptions are already filtered and passed as a flat structure (tag_id -> description)
        for tag_id, description in participant_descriptions.items():
            description_text += f"Person with Tag ID {tag_id}: {description}\n"

        return description_text + "\n"

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

        # Create participant descriptions
        formatted_participant_descriptions = self._format_participant_descriptions(participant_descriptions)

        # Load template and replace variables
        template = self.multi_angle_end_user_prompt_template
        template = template.replace("{{num_perspectives}}", str(len(processed_images)))
        template = template.replace("{{angle_descriptions}}", "\n".join(angle_descriptions))
        template = template.replace("{{participant_descriptions}}", formatted_participant_descriptions)
        template = template.replace("{{action_definitions}}", self.action_definitions)
        template = template.replace("{{decision_priority}}", self.decision_priority)

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

        # Create participant descriptions
        formatted_participant_descriptions = self._format_participant_descriptions(participant_descriptions)

        # Load template and replace variables
        template = self.multi_angle_vlm_user_prompt_template
        template = template.replace("{{num_perspectives}}", str(len(processed_images)))
        template = template.replace("{{angle_descriptions}}", "\n".join(angle_descriptions))
        template = template.replace("{{participant_descriptions}}", formatted_participant_descriptions)

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
        template = template.replace("{{decision_priority}}", self.decision_priority)

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
        response = self.vlm_client.chat.completions.create(
            model=self.vlm_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format={"type": "json_object"},
            extra_body=self.vlm_extra_body
        )

        result = response.choices[0].message.content
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
        llm_response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            response_format={"type": "json_object"},
            extra_body=self.llm_extra_body
        )

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
