import json
import os
import re
from io import BytesIO

import torch
from flask import request, jsonify
from openai import OpenAI
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.utils.video.gaze import detect_gaze
from openmmla.utils.video.image import encode_image_base64


class FewShotVLLMFrameAnalyzer(Server):
    """VLLM frame analyzer with few-shot learning capabilities. Analyzes images with multimodal language model
    (VLMs and LLMs) to identify AprilTags and predict individuals' action recognition results."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the Few-Shot VLM frame analyzer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()
        self._load_apriltag_examples()

    def _setup_yaml(self):
        analyzer_config = self.config['VLLMFrameAnalyzer']  # type: ignore

        self.backend = analyzer_config['backend']
        self.top_p = float(analyzer_config['top_p'])
        self.temperature = float(analyzer_config['temperature'])
        self.end_to_end = analyzer_config.get('end_to_end', False)
        self.apriltag_examples_dir = analyzer_config.get('apriltag_examples_dir', 'data/apriltag_examples')

        if not os.path.isabs(self.apriltag_examples_dir):
            self.apriltag_examples_dir = os.path.join(self.project_dir, self.apriltag_examples_dir)

        if self.backend in ['ollama', 'vllm', 'openai', 'qwen', 'gemini', 'deepseek', 'llamacpp', 'grok']:
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
        self.logger.info(f"AprilTag Examples Dir: {self.apriltag_examples_dir}")
        self.logger.info(f"End-to-End: {self.end_to_end}")

    def _setup_objects(self):
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

    def _load_apriltag_examples(self):
        """Load AprilTag examples from the specified directory."""
        self.apriltag_examples = []

        if not os.path.exists(self.apriltag_examples_dir):
            self.logger.warning(f"AprilTag examples directory not found: {self.apriltag_examples_dir}")
            return

        try:
            # Look for image files and extract tag IDs from filenames
            for filename in os.listdir(self.apriltag_examples_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Expected filename format: tag_<id>.<ext>
                    match = re.match(r'tag_(\d+)\.\w+', filename.lower())
                    if match:
                        tag_id = match.group(1)
                        filepath = os.path.join(self.apriltag_examples_dir, filename)

                        try:
                            with open(filepath, 'rb') as f:
                                image_bytes = f.read()
                                img_base64 = encode_image_base64(image_bytes)

                            self.apriltag_examples.append({
                                'tag_id': tag_id,
                                'image': img_base64
                            })
                            self.logger.info(f"Loaded AprilTag example for tag ID {tag_id}")
                        except Exception as e:
                            self.logger.error(f"Error loading AprilTag example image {filename}: {e}")

            self.logger.info(f"Loaded {len(self.apriltag_examples)} AprilTag examples")
        except Exception as e:
            self.logger.error(f"Error loading AprilTag examples: {e}")

    def process_request(self):
        """Process the image with few-shot learning approach."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        try:
            base_id = request.values.get('base_id')
            image_file = request.files['image']
            image_bytes = image_file.read()

            self.logger.info(f"Starting analysis for {base_id}...")

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
                vlm_messages = self._create_end_to_end_messages(image_b64, system_message)
                vlm_response = self._process_with_vlm(vlm_messages)

                result = {
                    'observations': vlm_response.get('observations', {}),
                    'classifications': vlm_response.get('classifications', {}),
                    'justifications': vlm_response.get('justifications', {})
                }
            else:
                # Two-step approach: VLM for observations, LLM for classification
                system_message = self._create_system_prompt()
                vlm_messages = self._create_vlm_messages(image_b64, system_message)
                vlm_response = self._process_with_vlm(vlm_messages)
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
                "You are an expert multimodal assistant for lab experiments. Your tasks are to:"
                "\n1. Analyze images focusing on gaze direction (colored lines) and hand interactions with objects."
                "\n2. Identify AprilTags based on provided visual examples."
                "\n3. Classify actions based on gaze, hand interactions, and predefined categories."
                "\nBase all observations and classifications ONLY on what is directly visible in the image."
            )
        else:
            system_content = (
                "You are an expert image analyst focusing on gaze direction and hand interactions. "
                "Identify AprilTags based on provided visual examples. "
                "Describe ONLY what is directly visible with absolute certainty."
            )

        return system_content

    def _create_end_to_end_messages(self, img_base64, system_message):
        """Create end-to-end messages with AprilTag examples before task description and emphasis on gaze endpoints."""

        user_text = (
            "# Lab Image Analysis\n\n"
            "### Visual Elements in Image:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show EXACT gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        )

        # Create user message content
        user_content = [{"type": "text", "text": user_text}]

        # Add AprilTag examples and images (if available)
        if hasattr(self, 'apriltag_examples') and self.apriltag_examples:
            apriltag_intro = "### AprilTag Reference Examples:\nBelow are examples of AprilTags with their corresponding IDs. Study these patterns carefully to identify similar tags in the main image:\n\n"
            user_content.append({"type": "text", "text": apriltag_intro})

            for example in self.apriltag_examples:
                example_text = f"Tag ID {example['tag_id']}: Study this pattern carefully."
                user_content.append({"type": "text", "text": example_text})
                user_content.append({"type": "image_url", "image_url": {"url": example['image'], "detail": "high"}})

        # Add task description
        task_instructions = (
            "\n### CRITICAL TASK:\n"
            "1. First, identify all AprilTags in the image by comparing them with the example AprilTags provided above.\n"
            "2. Match each AprilTag ID to the person wearing it.\n"
            "3. For each identified person, analyze their gaze and hand interactions.\n\n"

            "### CRITICAL INTERPRETATION RULES:\n"
            "1. GAZE ENDPOINT FOCUS: Pay EXTREME attention to the EXACT ENDPOINT of colored gaze lines - specifically what object or area the line TERMINATES ON. This endpoint is the precise target of the person's attention.\n"
            "2. If a person has NO colored line extending from their face, or their 'in' probability is low (below 0.5), "
            "this means they are NOT looking at anything within the frame. This person should be classified as 'Distracted'.\n"
            "3. The presence of a colored line indicates the person is looking at something within the frame - focus on the ENDPOINT of the line to determine the exact object they're looking at.\n"
            "4. Higher 'in' probability values (closer to 1.0) indicate higher confidence that the person is attending to "
            "something within the frame.\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "For each person with a visible AprilTag, compare the AprilTag with the examples shown previously to determine the person's correct ID number.\n"
            "For anyone without a visible AprilTag, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: MOST CRITICAL - Describe the EXACT OBJECT or AREA where the colored gaze line ENDS (the target). Name the specific object the endpoint of the line touches. If NO gaze line is visible or 'in' probability is low, explicitly state that the person is not looking at anything within the frame.\n"
            "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
            "- **Position**: Basic location in frame for verification only.\n"
            "- **Clothing**: Brief description for identification only.\n\n"

            "### 3. Critical Rules:\n"
            "- For gaze focus: The ENDPOINT of the colored line shows EXACTLY what object they're looking at - trust this visual indicator above all else.\n"
            "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
            "- If gaze line or hands aren't clearly visible, explicitly state this.\n"
            "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
            "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"

            "### 4. Classify each person's action\n"
            f"```\n{self.action_definitions}\n```\n"
            "- Base classifications primarily on 1) gaze endpoint (what specific object the gaze line terminates on) and 2) hand interactions\n"
            "- People WITHOUT gaze lines or with low 'in' values MUST be classified as 'Distracted' since they are not attending to anything in the frame\n"
            "- If gaze line isn't visible, default to 'Unclear' unless other evidence is definitive\n"
            "- Provide justification focused specifically on gaze endpoint and hand-object interactions\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"observations\": {\n"
            "    \"12\": {\n"
            "      \"gaze_focus\": \"Gaze line endpoint touches directly on the computer screen. In probability 0.92 confirms strong attention to screen.\",\n"
            "      \"hands_status\": \"Right hand on mouse with fingertips touching left and right buttons; left hand on keyboard with fingers on ASDF keys. Both hands actively touching and manipulating these input devices\",\n"
            "      \"position\": \"Center-right\",\n"
            "      \"clothing\": \"Dark blue shirt\"\n"
            "    },\n"
            "    \"43\": {\n"
            "      \"gaze_focus\": \"No gaze line visible. In probability 0.21 indicates person is not looking at anything within the frame.\",\n"
            "      \"hands_status\": \"Both hands hanging at sides, not in contact with any objects. Fingers relaxed and slightly curled, palms facing inward toward thighs\",\n"
            "      \"position\": \"Left side\",\n"
            "      \"clothing\": \"Light t-shirt, dark pants\"\n"
            "    }\n"
            "  },\n"
            "  \"classifications\": {\n"
            "    \"12\": \"Working-Software\",\n"
            "    \"43\": \"Distracted\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"12\": \"Classified as 'Working-Software' because gaze line endpoint touches directly on computer screen (in: 0.92) and hands are actively interacting with input devices (mouse and keyboard)\",\n"
            "    \"43\": \"Classified as 'Distracted' because the low in probability (0.21) and absence of gaze line indicate the person is not attending to anything visible in the frame\"\n"
            "  }\n"
            "}\n"
            "```"
        )
        user_content.append({"type": "text", "text": task_instructions})

        # Add separator and main analysis task text
        user_content.append({"type": "text", "text": "\n### MAIN IMAGE TO ANALYZE:"})

        # Add main analysis image
        user_content.append({"type": "image_url", "image_url": {"url": img_base64, "detail": "high"}})

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        return messages

    def _create_vlm_messages(self, img_base64, system_message):
        """Create VLM messages with AprilTag examples before task description and emphasis on gaze endpoints."""

        # Basic introduction
        user_text = "# Lab Image Analysis\n\n"

        # Visual elements explanation
        user_text += (
            "### Visual Elements in Image:\n"
            "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
            "- Colored boxes around faces: These indicate detected faces.\n"
            "- Colored lines from faces: These show EXACT gaze direction (where someone is looking).\n"
            "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        )

        # Create user message content
        user_content = [{"type": "text", "text": user_text}]

        # Add AprilTag examples and images (if available)
        if hasattr(self, 'apriltag_examples') and self.apriltag_examples:
            apriltag_intro = "### AprilTag Reference Examples:\nBelow are examples of AprilTags with their corresponding IDs. Study these patterns carefully to identify similar tags in the main image:\n\n"
            user_content.append({"type": "text", "text": apriltag_intro})

            for example in self.apriltag_examples:
                example_text = f"Tag ID {example['tag_id']}: Study this pattern carefully."
                user_content.append({"type": "text", "text": example_text})
                user_content.append({"type": "image_url", "image_url": {"url": example['image'], "detail": "high"}})

        # Add task description
        task_instructions = (
            "\n### CRITICAL TASK:\n"
            "1. First, identify all AprilTags in the image by comparing them with the example AprilTags provided above.\n"
            "2. Match each AprilTag ID to the person wearing it.\n"
            "3. For each identified person, analyze their gaze and hand interactions.\n\n"

            "### CRITICAL INTERPRETATION RULES:\n"
            "1. GAZE ENDPOINT FOCUS: Pay EXTREME attention to the EXACT ENDPOINT of colored gaze lines - specifically what object or area the line TERMINATES ON. This endpoint is the precise target of the person's attention.\n"
            "2. If a person has NO colored line extending from their face, or their 'in' probability is low (below 0.5), "
            "this means they are NOT looking at anything within the frame. This person is not attending to anything visible in the scene.\n"
            "3. The presence of a colored line indicates the person is looking at something within the frame - focus on the ENDPOINT of the line to determine the exact object they're looking at.\n"
            "4. Higher 'in' probability values (closer to 1.0) indicate higher confidence that the person is attending to "
            "something within the frame.\n\n"

            "### 1. Describe each person by their AprilTag ID number\n"
            "For each person with a visible AprilTag, compare the AprilTag with the examples shown previously to determine the person's correct ID number.\n"
            "For anyone without a visible AprilTag, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"

            "### 2. For each person, describe ONLY these elements:\n"
            "- **Gaze Focus**: MOST CRITICAL - Describe the EXACT OBJECT or AREA where the colored gaze line ENDS (the target). Name the specific object the endpoint of the line touches. If NO gaze line is visible or 'in' probability is low, explicitly state that the person is not looking at anything within the frame.\n"
            "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
            "- **Position**: Basic location in frame for verification only.\n"
            "- **Clothing**: Brief description for identification only.\n\n"

            "### 3. Critical Rules:\n"
            "- For gaze focus: The ENDPOINT of the colored line shows EXACTLY what object they're looking at - trust this visual indicator above all else.\n"
            "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
            "- If gaze line or hands aren't clearly visible, explicitly state this.\n"
            "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
            "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"observations\": {\n"
            "    \"12\": {\n"
            "      \"gaze_focus\": \"Gaze line endpoint touches directly on the computer screen. In probability 0.92 confirms strong attention to screen.\",\n"
            "      \"hands_status\": \"Right hand on mouse with fingertips touching left and right buttons; left hand on keyboard with fingers on ASDF keys. Both hands actively touching and manipulating these input devices\",\n"
            "      \"position\": \"Center-right\",\n"
            "      \"clothing\": \"Dark blue shirt\"\n"
            "    },\n"
            "    \"43\": {\n"
            "      \"gaze_focus\": \"No gaze line visible. In probability 0.21 indicates person is not looking at anything within the frame.\",\n"
            "      \"hands_status\": \"Both hands hanging at sides, not in contact with any objects. Fingers relaxed and slightly curled, palms facing inward toward thighs\",\n"
            "      \"position\": \"Left side\",\n"
            "      \"clothing\": \"Light t-shirt, dark pants\"\n"
            "    }\n"
            "  }\n"
            "}\n"
            "```"
        )
        user_content.append({"type": "text", "text": task_instructions})

        # Add separator and main analysis task text
        user_content.append({"type": "text", "text": "\n### MAIN IMAGE TO ANALYZE:"})

        # Add main analysis image
        user_content.append({"type": "image_url", "image_url": {"url": img_base64, "detail": "high"}})

        # Create complete message
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        return messages

    def _create_llm_messages(self, vlm_response):
        """Create messages for LLM classification based on VLM observations.
        
        Args:
            vlm_response: Response from the VLM containing observations
            
        Returns:
            list: Messages for the LLM
        """
        system_content = (
            "You are an expert action classifier focused specifically on gaze focus and hand interactions. "
            "Pay special attention to where gaze lines point (or if they're absent), and what objects hands are touching. "
            "Only use information explicitly stated in the observations."
        )

        image_description = vlm_response.get('observations', {})
        user_text = (
            "# Action Classification\n\n"
            "### 1. Observation Data:\n"
            f"```\n{image_description}\n```\n\n"

            "### 2. Action Categories:\n"
            f"```\n{self.action_definitions}\n```\n\n"

            "### 3. Classification Rules:\n"
            "- Give GREATER WEIGHT to these two critical factors, in order of importance:\n"
            "  1. Gaze focus - where the gaze line points (or if it's absent)\n"
            "  2. Hand interactions - what objects hands are touching\n"
            "- ABSOLUTELY CRITICAL: If a person has NO gaze line or their 'in' probability is low (below 0.5), classify them as 'Distracted' "
            "  as they are not looking at anything within the frame\n"
            "- If gaze isn't described or is unclear, default to 'Unclear' unless other evidence is definitive\n"
            "- Position and clothing should not factor into classification\n\n"

            "### Response Format\n"
            "```json\n"
            "{\n"
            "  \"classifications\": {\n"
            "    \"12\": \"Working-Software\",\n"
            "    \"43\": \"Distracted\"\n"
            "  },\n"
            "  \"justifications\": {\n"
            "    \"12\": \"Classified as 'Working-Software' because gaze line shows direct attention to computer screen (in: 0.92) and hands are described as actively interacting with keyboard and mouse\",\n"
            "    \"43\": \"Classified as 'Distracted' because the low in probability (0.21) and absence of gaze line indicate the person is not attending to anything visible in the frame\"\n"
            "  }\n"
            "}\n"
            "```"
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
