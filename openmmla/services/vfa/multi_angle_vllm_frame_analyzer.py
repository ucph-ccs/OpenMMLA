import json
import re
from io import BytesIO
from typing import Dict, List, Tuple, Any, Union, cast

import torch
from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.gaze import detect_gaze
from openmmla.utils.video.image import encode_image_base64


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

    def _setup_yaml(self):
        analyzer_config = self.config['VLLMFrameAnalyzer']  # type: ignore

        self.families = analyzer_config['families']
        self.backend = analyzer_config['backend']
        self.top_p = float(analyzer_config['top_p'])
        self.temperature = float(analyzer_config['temperature'])
        self.end_to_end = analyzer_config.get('end_to_end', False)
        
        # Load participant descriptions from config
        self.participant_descriptions = analyzer_config.get('participant_descriptions', {})
        self.logger.info(f"Loaded {len(self.participant_descriptions)} participant descriptions")

        # Load angle configuration (we'll be flexible with arbitrary angles)
        self.angle_config = analyzer_config.get('angle_config', {})
        self.logger.info(f"Loaded angle configurations: {list(self.angle_config.keys()) if self.angle_config else 'None'}")

        if self.backend in ['ollama', 'vllm', 'openai', 'qwen', 'gemini', 'deepseek']:
            backend_config = analyzer_config[self.backend]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.api_key = backend_config['api_key']
        self.vlm_model = backend_config['vlm_model']
        self.llm_model = backend_config['llm_model']
        self.vlm_base_url = backend_config.get('vlm_base_url', None)
        self.llm_base_url = backend_config.get('llm_base_url', None)

        # Load action definitions from config
        self.defined_actions = analyzer_config['defined_actions']
        self.action_definitions = '\n'.join(
            [f"'{key}': {value}" for key, value in self.defined_actions.items()])

        # Load VLM extra body from config
        self.vlm_extra_body = backend_config.get('VLMExtraBody', {})
        self.llm_extra_body = backend_config.get('LLMExtraBody', {})

        self.logger.info(f"API Key: {self.api_key}")
        self.logger.info(f"VLM Model: {self.vlm_model}")
        self.logger.info(f"LLM Model: {self.llm_model}")
        self.logger.info(f"VLM Base URL: {self.vlm_base_url}")
        self.logger.info(f"LLM Base URL: {self.llm_base_url}")
        self.logger.info(f"End-to-End: {self.end_to_end}")

    def _setup_objects(self):
        self.detector = Detector(families=self.families, nthreads=4)
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

        self.vlm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.vlm_base_url,
        )
        self.llm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.llm_base_url,
        )

    def process_request(self):
        """Process multiple images from different angles with contextual awareness."""
        try:
            base_id = request.values.get('base_id')
            
            # 检查是否使用images参数（多图像）还是image参数（单图像）
            if 'images' in request.files:
                image_files = request.files.getlist('images')
                angle_labels = request.values.getlist('angles')
            elif 'image' in request.files:
                # 兼容单图像情况，将其转换为列表格式处理
                image_files = [request.files['image']]
                angle_labels = []
            else:
                return jsonify({'error': 'No image files provided'}), 400
            
            # Validate we have at least one image
            if not image_files:
                return jsonify({'error': 'No images found in request'}), 400
                
            # Generate generic perspective labels if none provided
            if not angle_labels:
                angle_labels = [f"perspective_{i+1}" for i in range(len(image_files))]
            elif len(angle_labels) != len(image_files):
                return jsonify({'error': 'Number of angles does not match number of images'}), 400

            self.logger.info(f"Starting multi-angle analysis for {base_id} with {len(image_files)} images from angles: {angle_labels}")

            # Process each image
            processed_images = {}
            for i, (image_file, angle) in enumerate(zip(image_files, angle_labels)):
                image_bytes = image_file.read()
                
                # Process the image with AprilTag detection and gaze detection
                self.logger.info(f"Processed image from {angle} perspective")
                processed_image = self._process_single_image(image_bytes, angle)
                processed_images[angle] = processed_image
            
            # Analyze all processed images together
            if self.end_to_end:
                # End-to-end approach: VLM does both observation and classification for multiple images
                system_message = self._create_system_prompt(len(processed_images))
                messages = self._create_end_to_end_messages(processed_images, system_message, base_id)
                vlm_response = self._process_with_vlm(messages)

                # Extract observations, classifications, and justifications
                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': vlm_response.get('classifications', {}),
                    'justifications': vlm_response.get('justifications', {})
                }
            else:
                # Two-step approach: VLM for observations, LLM for classification
                system_message = self._create_system_prompt(len(processed_images))
                vlm_messages = self._create_vlm_messages(processed_images, system_message, base_id)
                vlm_response = self._process_with_vlm(vlm_messages)

                # Process with LLM for text classification
                llm_messages = self._create_llm_messages(vlm_response)
                llm_result = self._process_with_llm(llm_messages)

                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': llm_result.get('classifications', {}),
                    'justifications': llm_result.get('justifications', {})
                }

            self.logger.info(f"Finished multi-angle analyzing for {base_id}.")
            return jsonify(result), 200

        except Exception as e:
            self.logger.error("Exception during processing images", exc_info=True)
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    def _process_single_image(self, image_bytes: bytes, angle: str) -> Dict[str, Any]:
        """Process a single image with AprilTag and gaze detection.
        
        Args:
            image_bytes: Raw image bytes
            angle: The angle label for this image
            
        Returns:
            dict: Processed image information including base64 encoding and metadata
        """
        # Step 1: Detect AprilTags and render them on the image
        tag_pos, apriltag_image = detect_apriltags(
            image_bytes,
            self.detector,
            normalize=True,
            render=True,
            show=False,
            save=False
        )

        # If AprilTags were detected and rendered, use that image for gaze detection
        if tag_pos:
            image_bytes = pil_image_to_bytes(apriltag_image)

        # Step 2: Detect gaze on the AprilTag rendered image if gaze detection is available
        if self.gazelle_model and self.gazelle_transform:
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
                show=True,
                inout_thresh=0.5,
                render_heatmap=False,
                save=False
            )
            if gaze_results:
                image_bytes = pil_image_to_bytes(rendered_image)

        # Create a result dictionary with the processed image and metadata
        image_b64 = encode_image_base64(image_bytes)
        
        # Get angle description if available, otherwise create a generic one
        angle_description = self.angle_config.get(angle)
        if not angle_description:
            angle_description = f"Image from {angle} perspective"
            
        return {
            "image_b64": image_b64,
            "angle": angle,
            "angle_description": angle_description
        }

    def _create_system_prompt(self, num_perspectives: int):
        """Create plain text system prompt without images.
        
        Args:
            num_perspectives: Number of different perspectives/angles to be analyzed
            
        Returns:
            str: System prompt content
        """
        system_content = (
            f"You are an expert image analyst for lab experiments with expertise in analyzing {num_perspectives} camera angles simultaneously. "
            f"Your task is to analyze {num_perspectives} synchronized images showing the same scene from different perspectives. "
            "Focus specifically on two key elements across these views: "
            "1) Gaze direction (shown by colored lines) and 2) Hand interactions with objects. "
            "Use the multiple angles to resolve ambiguities - if something is obscured in one view, check if it's visible from another angle. "
            "Base all observations primarily on what is directly visible across all images, while making reasonable inferences when necessary."
        )

        return system_content

    def _format_participant_descriptions(self, base_id=None):
        """Format participant descriptions from config into readable text for prompts.
        
        Args:
            base_id: The base_id of the current request to filter participant descriptions
            
        Returns:
            str: Formatted participant descriptions
        """
        if not self.participant_descriptions:
            return ""
        
        description_text = "### Known Participants Reference:\n"
        description_text += "The following people may appear in the images. Use this information to help identify them by AprilTag ID:\n\n"
        
        # Handle the nested structure where participant_descriptions are organized by base_id
        if base_id and base_id in self.participant_descriptions:
            # Get descriptions specific to this base_id
            base_descriptions = self.participant_descriptions[base_id]
            for tag_id, description in base_descriptions.items():
                description_text += f"Person with Tag ID {tag_id}: {description}\n"
        # Fall back to flat structure if no matching base_id or base_id not provided
        elif isinstance(self.participant_descriptions, dict) and all(not isinstance(v, dict) for v in self.participant_descriptions.values()):
            # Old flat structure (tag_id -> description)
            for tag_id, description in self.participant_descriptions.items():
                description_text += f"Person with Tag ID {tag_id}: {description}\n"
        
        return description_text + "\n"

    def _create_end_to_end_messages(self, processed_images, system_message, base_id=None):
        """Generate a context-aware prompt message for end-to-end approach with multiple images.
        
        Args:
            processed_images: Dictionary containing processed images from different angles
            system_message: System prompt content
            base_id: The base_id of the current request
            
        Returns:
            list: Messages for the VLM
        """
        # Start with basic elements explanation
        user_text = (
            "# Multi-Perspective Lab Image Analysis\n\n"
            "### Visual Elements in Images:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
            "### Multiple Camera Views:\n"
            f"You have been provided with {len(processed_images)} synchronized images of the same scene, each from a different perspective:\n"
        )
        
        # Add descriptions for each angle
        for angle, image_data in processed_images.items():
            user_text += f"- **{angle}**: {image_data['angle_description']}\n"
        
        user_text += f"\nUse all {len(processed_images)} camera perspectives to resolve ambiguities. If something is obscured in one view, check if it's visible from another angle.\n\n"
        
        # Add participant descriptions if available
        participant_descriptions = self._format_participant_descriptions(base_id)
        if participant_descriptions:
            user_text += participant_descriptions
            
        # Add interpretation rules
        user_text += (
            "### MULTI-PERSPECTIVE OBSERVATION GUIDELINES:\n"
            "1. GAZE FOCUS: Describe in detail what each person is looking at based on the colored gaze lines across all available perspectives.\n"
            "   - Compare gaze lines across different perspectives for more accurate targeting\n"
            "   - Include the specific target object/area where the gaze line points\n"
            "   - Include the 'in: X.XX' probability value when mentioned in the image\n"
            "   - DO NOT infer head position - rely SOLELY on the rendered gaze lines\n"
            "   - WHEN NO GAZE LINE IS VISIBLE IN ANY VIEW: You may make a cautious inference based on head orientation\n"
            "   - Always mark gaze inferences with: 'INFERENCE (low confidence): Based on head orientation, likely looking at [object]'\n"
            "2. HAND STATUS: Provide detailed description of hand positions and activities, combining information from all perspectives.\n"
            "   - If hands are visible in one view but not another, prioritize the view where they're visible\n"
            "   - Describe exact hand positions and what they are touching/interacting with\n"
            "   - If hands are visible but inactive (resting, clasped), clearly state this\n"
            "   - If hands are obscured in all views, clearly indicate this\n"
            "3. RESOLVING AMBIGUITIES:\n"
            "   - First check if information missing in one perspective is visible in another\n"
            "   - Only make inferences when something is obscured in ALL available perspectives\n"
            "   - When making inferences, be explicit about the visual cues you're using from multiple perspectives\n"
            "4. CLEAR DISTINCTION BETWEEN OBSERVATION AND INFERENCE:\n"
            "   - For direct observations, describe exactly what is visible in which camera view\n"
            "   - For any inferences (when something is partially obscured in all views), clearly mark these by starting with 'INFERENCE:'\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "First, for each person with a visible AprilTag ID number in ANY view, map their identity with the shown ID number (e.g., \"6\", \"10\").\n"
            "Second, for people without visible AprilTag IDs in ANY view, try to match their appearance and clothing with known participant descriptions, and use that ID number.\n"
            "Finally, for anyone without a visible AprilTag ID in ANY view and no matching description, assign sequential numbers starting from 100 (e.g., \"100\", \"101\").\n"
            "IMPORTANT: ALWAYS use only numeric IDs (e.g., \"6\", \"10\", \"100\") without words like \"Person\" or \"Tag\" when identifying people in your response.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: Describe exactly where the gaze line points and what object/area is at that endpoint. Include 'in' probability. Note which camera view provides the clearest information.\n"
            "- **Hands Status**: Describe visible hand positions and activities in detail. Note which camera view provides the clearest information about hands.\n"
            "- **Position**: Basic location in frame for identification only.\n"
            "- **Clothing**: Brief description for identification only, matching with known participant descriptions when possible.\n\n"

            "### 3. Activity Classification\n"
            f"```\n{self.action_definitions}\n```\n"
            "- Classify each person's action based on your observations of their gaze focus and hand status, using the definitions above\n"
            "- Use the combination of gaze focus and hand status as your primary criteria for classification\n"
            "- When making reasonable inferences about obscured elements, clearly indicate this in your justification\n"
            "- Provide detailed justification explaining how the observed gaze and hand status led to your classification\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"observations\": {\n"
            "    \"0\": {\n"
            "      \"gaze_focus\": \"In perspective_1: Gaze line (green) points at colleague's face. In perspective_2: Gaze line confirms attention directed at person with Tag ID 1. In probability 0.92 indicates high confidence attention is within frame.\",\n"
            "      \"hands_status\": \"In perspective_1: Both hands are on keyboard, actively typing. Left hand positioned over WASD keys, right hand near spacebar. perspective_2 partially obscures hands but confirms typing activity.\",\n"
            "      \"position\": \"Center-right of frame in all views\",\n"
            "      \"clothing\": \"Yellow sweater matching description of participant with Tag ID 0\"\n"
            "    },\n"
            "    \"1\": {\n"
            "      \"gaze_focus\": \"In perspective_1: Gaze line (orange) points at documents on desk. perspective_2 confirms this. In probability 0.78 indicates moderate confidence attention is within frame.\",\n"
            "      \"hands_status\": \"In perspective_2: Hands are obscured by desk edge. In perspective_1: Hands clearly visible manipulating papers, sorting through documents.\",\n"
            "      \"position\": \"Left side of frame in all views\",\n"
            "      \"clothing\": \"Dark blue sweater matching description of participant with Tag ID 1\"\n"
            "    }\n"
            "  },\n"
            "  \"classifications\": {\n"
            "    \"0\": \"Working-Software\",\n"
            "    \"1\": \"Working-Document\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"0\": \"Classified as 'Working-Software' because gaze is directed at colleague's face (in: 0.92) while hands are actively typing on keyboard as clearly visible in perspective_1.\",\n"
            "    \"1\": \"Classified as 'Working-Document' because gaze is directed at documents on desk (in: 0.78) and hands are manipulating papers as visible in perspective_1.\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "### CRITICAL ID ASSIGNMENT RULES:\n"
            "1. ALWAYS use numeric IDs as keys in the JSON response, never use descriptive identifiers like 'Person 1' or 'Tag_1'\n"
            "2. PRESERVE the same exact numeric IDs across all observations, classifications and justifications\n"
            "3. Ensure you identify the same person consistently across all camera perspectives"
        )

        # Create message content with multiple images
        content = [{"type": "text", "text": user_text}]
        
        # Add all the images to the content - fixing the type error with proper casting
        for angle, image_data in processed_images.items():
            content.append(cast(Dict[str, str], {
                "type": "image_url", 
                "image_url": {
                    "url": image_data["image_b64"], 
                    "detail": "high"
                }
            }))

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content}
        ]

        return messages

    def _create_vlm_messages(self, processed_images, system_message, base_id=None):
        """Generate a context-aware prompt message for VLM with multiple images.
        
        Args:
            processed_images: Dictionary containing processed images from different angles
            system_message: System prompt content
            base_id: The base_id of the current request
            
        Returns:
            list: Messages for the VLM
        """
        # Start with basic elements explanation
        user_text = (
            "# Multi-Perspective Lab Image Analysis\n\n"
            "### Visual Elements in Images:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
            "### Multiple Camera Views:\n"
            f"You have been provided with {len(processed_images)} synchronized images of the same scene, each from a different perspective:\n"
        )
        
        # Add descriptions for each angle
        for angle, image_data in processed_images.items():
            user_text += f"- **{angle}**: {image_data['angle_description']}\n"
        
        user_text += f"\nUse all {len(processed_images)} camera perspectives to resolve ambiguities. If something is obscured in one view, check if it's visible from another angle.\n\n"
        
        # Add participant descriptions if available
        participant_descriptions = self._format_participant_descriptions(base_id)
        if participant_descriptions:
            user_text += participant_descriptions
            
        # Add interpretation rules
        user_text += (
            "### MULTI-PERSPECTIVE OBSERVATION GUIDELINES:\n"
            "1. GAZE FOCUS: Describe in detail what each person is looking at based on the colored gaze lines across all available perspectives.\n"
            "   - Compare gaze lines across different perspectives for more accurate targeting\n"
            "   - Include the specific target object/area where the gaze line points\n"
            "   - Include the 'in: X.XX' probability value when mentioned in the image\n"
            "   - DO NOT infer head position - rely SOLELY on the rendered gaze lines\n"
            "   - WHEN NO GAZE LINE IS VISIBLE IN ANY VIEW: You may make a cautious inference based on head orientation\n"
            "   - Always mark gaze inferences with: 'INFERENCE (low confidence): Based on head orientation, likely looking at [object]'\n"
            "2. HAND STATUS: Provide detailed description of hand positions and activities, combining information from all perspectives.\n"
            "   - If hands are visible in one view but not another, prioritize the view where they're visible\n"
            "   - Describe exact hand positions and what they are touching/interacting with\n"
            "   - If hands are visible but inactive (resting, clasped), clearly state this\n"
            "   - If hands are obscured in all views, clearly indicate this\n"
            "3. RESOLVING AMBIGUITIES:\n"
            "   - First check if information missing in one perspective is visible in another\n"
            "   - Only make inferences when something is obscured in ALL available perspectives\n"
            "   - When making inferences, be explicit about the visual cues you're using from multiple perspectives\n"
            "4. CLEAR DISTINCTION BETWEEN OBSERVATION AND INFERENCE:\n"
            "   - For direct observations, describe exactly what is visible in which camera view\n"
            "   - For any inferences (when something is partially obscured in all views), clearly mark these by starting with 'INFERENCE:'\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "First, for each person with a visible AprilTag ID number in ANY view, map their identity with the shown ID number (e.g., \"6\", \"10\").\n"
            "Second, for people without visible AprilTag IDs in ANY view, try to match their appearance and clothing with known participant descriptions, and use that ID number.\n"
            "Finally, for anyone without a visible AprilTag ID in ANY view and no matching description, assign sequential numbers starting from 100 (e.g., \"100\", \"101\").\n"
            "IMPORTANT: ALWAYS use only numeric IDs (e.g., \"6\", \"10\", \"100\") without words like \"Person\" or \"Tag\" when identifying people in your response.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: Describe exactly where the gaze line points and what object/area is at that endpoint. Include 'in' probability. Note which camera view provides the clearest information.\n"
            "- **Hands Status**: Describe visible hand positions and activities in detail. Note which camera view provides the clearest information about hands.\n"
            "- **Position**: Basic location in frame for identification only.\n"
            "- **Clothing**: Brief description for identification only, matching with known participant descriptions when possible.\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"observations\": {\n"
            "    \"0\": {\n"
            "      \"gaze_focus\": \"In perspective_1: Gaze line (green) points at colleague's face. In perspective_2: Gaze line confirms attention directed at person with Tag ID 1. In probability 0.92 indicates high confidence attention is within frame.\",\n"
            "      \"hands_status\": \"In perspective_1: Both hands are on keyboard, actively typing. Left hand positioned over WASD keys, right hand near spacebar. perspective_2 partially obscures hands but confirms typing activity.\",\n"
            "      \"position\": \"Center-right of frame in all views\",\n"
            "      \"clothing\": \"Yellow sweater matching description of participant with Tag ID 0\"\n"
            "    },\n"
            "    \"1\": {\n"
            "      \"gaze_focus\": \"In perspective_1: Gaze line (orange) points at documents on desk. perspective_2 confirms this. In probability 0.78 indicates moderate confidence attention is within frame.\",\n"
            "      \"hands_status\": \"In perspective_2: Hands are obscured by desk edge. In perspective_1: Hands clearly visible manipulating papers, sorting through documents.\",\n"
            "      \"position\": \"Left side of frame in all views\",\n"
            "      \"clothing\": \"Dark blue sweater matching description of participant with Tag ID 1\"\n"
            "    }\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "### CRITICAL ID ASSIGNMENT RULES:\n"
            "1. ALWAYS use numeric IDs as keys in the JSON response, never use descriptive identifiers like 'Person 1' or 'Tag_1'\n"
            "2. Ensure you identify the same person consistently across all camera perspectives"
        )

        # Create message content with multiple images
        content = [{"type": "text", "text": user_text}]
        
        # Add all the images to the content - fixing the type error with proper casting
        for angle, image_data in processed_images.items():
            content.append(cast(Dict[str, str], {
                "type": "image_url", 
                "image_url": {
                    "url": image_data["image_b64"], 
                    "detail": "high"
                }
            }))

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content}
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
        system_content = (
            "You are an expert action classifier focused on gaze focus and hand interactions. "
            "Your task is to analyze detailed observations about people's gaze and hand activities "
            "from multiple camera perspectives and classify their actions according to specific definitions."
        )

        image_description = vlm_response.get('observations', {})
        user_text = (
            "# Multi-Perspective Action Classification\n\n"
            "### 1. Observation Data (from multiple camera views):\n"
            f"```\n{image_description}\n```\n\n"

            "### 2. Action Categories:\n"
            f"```\n{self.action_definitions}\n```\n\n"

            "### 3. Classification Approach:\n"
            "- Focus primarily on two key elements for classification:\n"
            "  1. Gaze Focus - what the person is looking at (target of gaze line)\n"
            "  2. Hands Status - what the hands are doing/touching (or inferred to be doing if partially obscured)\n"
            "- Consider information from ALL available camera perspectives to make the most accurate assessment\n"
            "- When one perspective provides clear information that another doesn't, prioritize the clearer view\n"
            "- Classify each person based on the ACTION DEFINITIONS provided above\n"
            "- Consider both direct observations and clearly marked inferences in the observations\n"
            "- The combination of where someone is looking and what their hands are doing should determine classification\n"
            "- Provide detailed justification for each classification, explicitly referencing the observed gaze and hand status from specific camera views\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"classifications\": {\n"
            "    \"0\": \"Working-Software\",\n"
            "    \"1\": \"Working-Document\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"0\": \"Classified as 'Working-Software' because gaze is directed at colleague's face (in: 0.92) while hands are actively typing on keyboard as clearly visible in perspective_1.\",\n"
            "    \"1\": \"Classified as 'Working-Document' because gaze is directed at documents on desk (in: 0.78) and hands are manipulating papers as visible in perspective_1.\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "### CRITICAL ID ASSIGNMENT RULES:\n"
            "1. ALWAYS use numeric IDs as keys in the JSON response, never use descriptive identifiers like 'Person 1' or 'Tag_1'\n"
            "2. PRESERVE the same exact numeric IDs that were used in the observation data\n"
            "3. All IDs must be consistent across classifications and justifications"
        )

        llm_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text}
        ]

        return llm_messages

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