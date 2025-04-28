import json
import re
from io import BytesIO

import torch
from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.gaze import detect_gaze
from openmmla.utils.video.image import encode_image_base64


class ContextAwareVLLMFrameAnalyzer(Server):
    """Context-aware VLLM frame analyzer analyzes the image with multimodal language model (VLMs and LLMs)
    and predicts individuals' action recognition results mapped with AprilTag IDs. Includes contextual
    understanding of gaze and activity, and participant descriptions."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the Context-aware VLM frame analyzer.

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
        """Process the image with full rendering pipeline and contextual awareness."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        try:
            base_id = request.values.get('base_id')
            image_file = request.files['image']
            image_bytes = image_file.read()

            self.logger.info(f"Starting analysis for {base_id}...")

            # Step 1: Detect AprilTags and render them on the image
            tag_positions, apriltag_image = detect_apriltags(
                image_bytes,
                self.detector,
                normalize=True,
                render=True,
                show=False,
                save=False
            )

            if apriltag_image:
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
                if rendered_image:
                    image_bytes = pil_image_to_bytes(rendered_image)

            image_b64 = encode_image_base64(image_bytes)

            if self.end_to_end:
                # End-to-end approach: VLM does both observation and classification
                system_message = self._create_system_prompt()
                messages = self._create_end_to_end_messages(image_b64, system_message, base_id)
                vlm_response = self._process_with_vlm(messages)

                # Extract observations, classifications, and justifications
                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': vlm_response.get('classifications', {}),
                    'justifications': vlm_response.get('justifications', {})
                }
            else:
                # Two-step approach: VLM for observations, LLM for classification
                system_message = self._create_system_prompt()
                vlm_messages = self._create_vlm_messages(image_b64, system_message, base_id)
                vlm_response = self._process_with_vlm(vlm_messages)

                # Process with LLM for text classification
                llm_messages = self._create_llm_messages(vlm_response)
                llm_result = self._process_with_llm(llm_messages)

                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': llm_result.get('classifications', {}),
                    'justifications': llm_result.get('justifications', {})
                }

            self.logger.info(f"Finished analyzing for {base_id}.")
            return jsonify(result), 200

        except Exception as e:
            self.logger.error("Exception during processing image", exc_info=True)
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    def _create_system_prompt(self):
        """Create plain text system prompt without images."""
        if self.end_to_end:
            system_content = (
                "You are an expert image analyst for lab experiments. Focus specifically on two key elements: "
                "1) Gaze direction (shown by colored lines) and 2) Hand interactions with objects. "
                "You should also use contextual information to make reasonable inferences about participant activities. "
                "Base all observations primarily on what is directly visible in the image, but consider practical context."
            )
        else:
            system_content = (
                "You are an expert in analyzing images with focus on two specific elements: "
                "1) Gaze direction (shown by colored lines) and 2) Hand interactions with objects. "
                "Describe primarily what is directly visible, but use contextual cues to make reasonable inferences about "
                "activities that may be partially obscured."
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
        description_text += "The following people may appear in the image. Use this information to help identify them by AprilTag ID:\n\n"

        # Handle the nested structure where participant_descriptions are organized by base_id
        if base_id and base_id in self.participant_descriptions:
            # Get descriptions specific to this base_id
            base_descriptions = self.participant_descriptions[base_id]
            for tag_id, description in base_descriptions.items():
                description_text += f"Person with Tag ID {tag_id}: {description}\n"
        # Fall back to flat structure if no matching base_id or base_id not provided
        elif isinstance(self.participant_descriptions, dict) and all(
                not isinstance(v, dict) for v in self.participant_descriptions.values()):
            # Old flat structure (tag_id -> description)
            for tag_id, description in self.participant_descriptions.items():
                description_text += f"Person with Tag ID {tag_id}: {description}\n"

        return description_text + "\n"

    def _create_end_to_end_messages(self, img_base64, system_message, base_id=None):
        """Generate a context-aware prompt message for end-to-end approach with the rendered image.
        
        Args:
            img_base64: Base64 encoded image string
            system_message: System prompt content
            base_id: The base_id of the current request
            
        Returns:
            list: Messages for the VLM
        """
        # Start with basic elements explanation
        user_text = (
            "# Lab Image Analysis\n\n"
            "### Visual Elements in Image:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        )

        # Add participant descriptions if available
        participant_descriptions = self._format_participant_descriptions(base_id)
        if participant_descriptions:
            user_text += participant_descriptions

        # Add interpretation rules with updated flexibility for gaze focus
        user_text += (
            "### OBSERVATION GUIDELINES:\n"
            "1. GAZE FOCUS: Describe in detail what each person is looking at based on the colored gaze lines.\n"
            "   - Include the specific target object/area where the gaze line points\n"
            "   - Include the 'in: X.XX' probability value when mentioned in the image\n"
            "   - DO NOT infer head position - rely SOLELY on the rendered gaze lines\n"
            "   - WHEN NO GAZE LINE IS VISIBLE: You may make a cautious inference based on head orientation\n"
            "   - Always mark gaze inferences with: 'INFERENCE (low confidence): Based on head orientation, likely looking at [object]'\n"
            "2. HAND STATUS: Provide detailed description of hand positions and activities.\n"
            "   - Describe exact hand positions and what they are touching/interacting with\n"
            "   - If hands are visible but inactive (resting, clasped), clearly state this\n"
            "   - If hands are obscured or partially visible, clearly indicate this\n"
            "3. CAUTION WHEN HANDS ARE OBSCURED:\n"
            "   - When hands are obscured, DO NOT automatically assume they are interacting with nearby objects\n"
            "   - Only infer hand interactions if there are STRONG visual cues (e.g., clear arm position and angle directly connecting to an object)\n"
            "   - If uncertain about hand activity behind obstruction, state: 'Hands are obscured by [object]. Unable to determine exact hand activity.'\n"
            "   - Avoid assumptions about precise hand positions when not visible - distinguish between what you can see and what you cannot\n"
            "4. CLEAR DISTINCTION BETWEEN OBSERVATION AND INFERENCE:\n"
            "   - For direct observations, describe exactly what is visible\n"
            "   - For any inferences (when something is partially obscured), clearly mark these by starting with 'INFERENCE:'\n"
            "   - When making inferences, indicate your confidence level (e.g., 'INFERENCE (low confidence): hands may be near keyboard but not clearly engaged')\n"
            "   - Example of appropriate inference: 'INFERENCE (high confidence): Based on visible wrist angle and arm position pointing directly at keyboard, hands are likely typing'\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "First, for each person with a visible AprilTag ID number, map their identity with the shown ID number (e.g., \"6\", \"10\").\n"
            "Second, for people without visible AprilTag IDs, try to match their appearance and clothing with known participant descriptions, and use that ID number.\n"
            "Finally, for anyone without a visible AprilTag ID and no matching description, assign sequential numbers starting from 100 (e.g., \"100\", \"101\").\n"
            "IMPORTANT: ALWAYS use only numeric IDs (e.g., \"6\", \"10\", \"100\") without words like \"Person\" or \"Tag\" when identifying people in your response.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: Describe exactly where the gaze line points and what object/area is at that endpoint. Include 'in' probability.\n"
            "- **Hands Status**: Describe visible hand positions and activities in detail. If hands are obscured, state that clearly.\n"
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
            "      \"gaze_focus\": \"Gaze line (green) points directly at laptop screen. In probability 0.92 indicates high confidence attention is within frame.\",\n"
            "      \"hands_status\": \"Both hands are on keyboard, actively typing. Left hand positioned over WASD keys, right hand near spacebar.\",\n"
            "      \"position\": \"Center-right of frame\",\n"
            "      \"clothing\": \"Yellow sweater matching description of participant with Tag ID 0\"\n"
            "    },\n"
            "    \"1\": {\n"
            "      \"gaze_focus\": \"Gaze line (orange) points at documents on desk. In probability 0.78 indicates moderate confidence attention is within frame.\",\n"
            "      \"hands_status\": \"Hands are partially obscured by desk edge. INFERENCE: Based on visible wrist position and arm angle, appears to be holding or manipulating papers.\",\n"
            "      \"position\": \"Left side of frame\",\n"
            "      \"clothing\": \"Dark blue sweater matching description of participant with Tag ID 1\"\n"
            "    },\n"
            "    \"2\": {\n"
            "      \"gaze_focus\": \"No gaze line is visible for this person. INFERENCE (low confidence): Based on head orientation, likely looking at the electronic components on the desk.\",\n"
            "      \"hands_status\": \"Both hands are visible on the desk, manipulating electronic components. Left hand is holding a wire, right hand appears to be adjusting a small device.\",\n"
            "      \"position\": \"Center of frame\",\n"
            "      \"clothing\": \"Red t-shirt, no matching description in known participants\"\n"
            "    }\n"
            "  },\n"
            "  \"classifications\": {\n"
            "    \"0\": \"Working-Software\",\n"
            "    \"1\": \"Working-Document\",\n"
            "    \"2\": \"Working-Hardware\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"0\": \"Classified as 'Working-Software' because gaze is directed at laptop screen (in: 0.92) AND hands are actively interacting with keyboard.\",\n"
            "    \"1\": \"Classified as 'Working-Document' because gaze is directed at documents on desk (in: 0.78) and inferred hand position suggests manipulation of papers.\",\n"
            "    \"2\": \"Classified as 'Working-Hardware' because inferred gaze is toward electronic components (based on head orientation) AND hands are visibly manipulating these hardware components.\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "### CRITICAL ID ASSIGNMENT RULES:\n"
            "1. ALWAYS use numeric IDs as keys in the JSON response, never use descriptive identifiers like 'Person 1' or 'Tag_1'\n"
            "2. PRESERVE the same exact numeric IDs that were used in the observation data\n"
            "3. All IDs must be consistent across classifications and justifications"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img_base64, "detail": "high"}}
            ]}
        ]

        return messages

    def _create_vlm_messages(self, img_base64, system_message, base_id=None):
        """Generate a context-aware prompt message for VLM with the rendered image.
        
        Args:
            img_base64: Base64 encoded image string
            system_message: System prompt content
            base_id: The base_id of the current request
            
        Returns:
            list: Messages for the VLM
        """
        # Start with basic elements explanation
        user_text = (
            "# Lab Image Analysis\n\n"
            "### Visual Elements in Image:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        )

        # Add participant descriptions if available
        participant_descriptions = self._format_participant_descriptions(base_id)
        if participant_descriptions:
            user_text += participant_descriptions

        # Add interpretation rules with updated flexibility for gaze focus
        user_text += (
            "### OBSERVATION GUIDELINES:\n"
            "1. GAZE FOCUS: Describe in detail what each person is looking at based on the colored gaze lines.\n"
            "   - Include the specific target object/area where the gaze line points\n"
            "   - Include the 'in: X.XX' probability value when mentioned in the image\n"
            "   - DO NOT infer head position - rely SOLELY on the rendered gaze lines\n"
            "   - WHEN NO GAZE LINE IS VISIBLE: You may make a cautious inference based on head orientation\n"
            "   - Always mark gaze inferences with: 'INFERENCE (low confidence): Based on head orientation, likely looking at [object]'\n"
            "2. HAND STATUS: Provide detailed description of hand positions and activities.\n"
            "   - Describe exact hand positions and what they are touching/interacting with\n"
            "   - If hands are visible but inactive (resting, clasped), clearly state this\n"
            "   - If hands are obscured or partially visible, clearly indicate this\n"
            "3. CAUTION WHEN HANDS ARE OBSCURED:\n"
            "   - When hands are obscured, DO NOT automatically assume they are interacting with nearby objects\n"
            "   - Only infer hand interactions if there are STRONG visual cues (e.g., clear arm position and angle directly connecting to an object)\n"
            "   - If uncertain about hand activity behind obstruction, state: 'Hands are obscured by [object]. Unable to determine exact hand activity.'\n"
            "   - Avoid assumptions about precise hand positions when not visible - distinguish between what you can see and what you cannot\n"
            "4. CLEAR DISTINCTION BETWEEN OBSERVATION AND INFERENCE:\n"
            "   - For direct observations, describe exactly what is visible\n"
            "   - For any inferences (when something is partially obscured), clearly mark these by starting with 'INFERENCE:'\n"
            "   - When making inferences, indicate your confidence level (e.g., 'INFERENCE (low confidence): hands may be near keyboard but not clearly engaged')\n"
            "   - Example of appropriate inference: 'INFERENCE (high confidence): Based on visible wrist angle and arm position pointing directly at keyboard, hands are likely typing'\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "First, for each person with a visible AprilTag ID number, map their identity with the shown ID number (e.g., \"6\", \"10\").\n"
            "Second, for people without visible AprilTag IDs, try to match their appearance and clothing with known participant descriptions, and use that ID number.\n"
            "Finally, for anyone without a visible AprilTag ID and no matching description, assign sequential numbers starting from 100 (e.g., \"100\", \"101\").\n"
            "IMPORTANT: ALWAYS use only numeric IDs (e.g., \"6\", \"10\", \"100\") without words like \"Person\" or \"Tag\" when identifying people in your response.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: Describe exactly where the gaze line points and what object/area is at that endpoint. Include 'in' probability.\n"
            "- **Hands Status**: Describe visible hand positions and activities in detail. If hands are obscured, state that clearly.\n"
            "- **Position**: Basic location in frame for identification only.\n"
            "- **Clothing**: Brief description for identification only, matching with known participant descriptions when possible.\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"observations\": {\n"
            "    \"0\": {\n"
            "      \"gaze_focus\": \"Gaze line (green) points directly at laptop screen. In probability 0.92 indicates high confidence attention is within frame.\",\n"
            "      \"hands_status\": \"Both hands are on keyboard, actively typing. Left hand positioned over WASD keys, right hand near spacebar.\",\n"
            "      \"position\": \"Center-right of frame\",\n"
            "      \"clothing\": \"Yellow sweater matching description of participant with Tag ID 0\"\n"
            "    },\n"
            "    \"1\": {\n"
            "      \"gaze_focus\": \"Gaze line (orange) points at documents on desk. In probability 0.78 indicates moderate confidence attention is within frame.\",\n"
            "      \"hands_status\": \"Hands are partially obscured by desk edge. INFERENCE: Based on visible wrist position and arm angle, appears to be holding or manipulating papers.\",\n"
            "      \"position\": \"Left side of frame\",\n"
            "      \"clothing\": \"Dark blue sweater matching description of participant with Tag ID 1\"\n"
            "    },\n"
            "    \"2\": {\n"
            "      \"gaze_focus\": \"No gaze line is visible for this person. INFERENCE (low confidence): Based on head orientation, likely looking at the electronic components on the desk.\",\n"
            "      \"hands_status\": \"Both hands are visible on the desk, manipulating electronic components. Left hand is holding a wire, right hand appears to be adjusting a small device.\",\n"
            "      \"position\": \"Center of frame\",\n"
            "      \"clothing\": \"Red t-shirt, no matching description in known participants\"\n"
            "    }\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "### CRITICAL ID ASSIGNMENT RULES:\n"
            "1. ALWAYS use numeric IDs as keys in the JSON response, never use descriptive identifiers like 'Person 1' or 'Tag_1'\n"
            "2. PRESERVE the same exact numeric IDs that were used in the observation data\n"
            "3. All IDs must be consistent across classifications and justifications"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img_base64, "detail": "high"}}
            ]}
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
            "and classify their actions according to specific definitions."
        )

        image_description = vlm_response.get('observations', {})
        user_text = (
            "# Action Classification\n\n"
            "### 1. Observation Data:\n"
            f"```\n{image_description}\n```\n\n"

            "### 2. Action Categories:\n"
            f"```\n{self.action_definitions}\n```\n\n"

            "### 3. Classification Approach:\n"
            "- Focus primarily on two key elements for classification:\n"
            "  1. Gaze Focus - what the person is looking at (target of gaze line)\n"
            "  2. Hands Status - what the hands are doing/touching (or inferred to be doing if partially obscured)\n"
            "- Classify each person based on the ACTION DEFINITIONS provided above\n"
            "- Consider both direct observations and clearly marked inferences in the observations\n"
            "- The combination of where someone is looking and what their hands are doing should determine classification\n"
            "- Provide detailed justification for each classification, explicitly referencing the observed gaze and hand status\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"classifications\": {\n"
            "    \"0\": \"Working-Software\",\n"
            "    \"1\": \"Working-Document\",\n"
            "    \"2\": \"Working-Hardware\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"0\": \"Classified as 'Working-Software' because gaze is directed at laptop screen (in: 0.92) AND hands are actively interacting with keyboard.\",\n"
            "    \"1\": \"Classified as 'Working-Document' because gaze is directed at documents on desk (in: 0.78) and inferred hand position suggests manipulation of papers.\",\n"
            "    \"2\": \"Classified as 'Working-Hardware' because inferred gaze is toward electronic components (based on head orientation) AND hands are visibly manipulating these hardware components.\"\n"
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
