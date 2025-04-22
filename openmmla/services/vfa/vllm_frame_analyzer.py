import json
import re
import torch
from io import BytesIO
from PIL import Image

from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector
from retinaface import RetinaFace

from openmmla.services.server import Server
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.gaze import detect_gaze
from openmmla.utils.video.image import encode_image_base64, load_image


class VLLMFrameAnalyzer(Server):
    """VLLM frame analyzer analyzes the image with multimodal language model (VLMs and LLMs) and predicts individuals'
    action recognition results mapped with AprilTag IDs."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the VLM frame analyzer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()

    def _setup_yaml(self):
        analyzer_config = self.config['VLLMFrameAnalyzer'] # type: ignore

        self.families = analyzer_config['families'] 
        self.backend = analyzer_config['backend']
        self.top_p = float(analyzer_config['top_p'])
        self.temperature = float(analyzer_config['temperature'])
        self.end_to_end = analyzer_config.get('end_to_end', False)

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
        self.action_definitions = '\n'.join(
            [f"'{key}': {value}" for key, value in analyzer_config['defined_actions'].items()])

        # Load VLM extra body from config
        self.vlm_extra_body = backend_config.get('VLMExtraBody', {})
        self.llm_extra_body = backend_config.get('LLMExtraBody', {})

        print(f"API Key: {self.api_key}")
        print(f"VLM Model: {self.vlm_model}")
        print(f"LLM Model: {self.llm_model}")
        print(f"VLM Base URL: {self.vlm_base_url}")
        print(f"LLM Base URL: {self.llm_base_url}")

    def _setup_objects(self):
        self.detector = Detector(families=self.families, nthreads=4)
        
        # Set up face detector for gaze detection
        self.face_detector = RetinaFace.detect_faces
        
        # Set up Gazelle model for gaze detection
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
            print(f"Gazelle model loaded on {device}")
        except Exception as e:
            print(f"Error loading Gazelle model: {e}")
            self.gazelle_model = None
            self.gazelle_transform = None
            self.device = None
        
        # Setup OpenAI client
        self.vlm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.vlm_base_url,
        )
        self.llm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.llm_base_url,
        )

    def process_request(self):
        """Process the image with full rendering pipeline."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        try:
            base_id = request.values.get('base_id')
            image_file = request.files['image']
            image_data = image_file.read()

            self.logger.info(f"starting analyzing for {base_id}...")

            # Step 1: Detect AprilTags and render them on the image
            tag_positions, apriltag_image = detect_apriltags(
                image_data, 
                self.detector, 
                normalize=True,
                render=True,
                show=False,
                save=False
            )
            
            if not apriltag_image:
                self.logger.warning("AprilTag detection did not produce a rendered image")
                # Use the original image as fallback
                apriltag_image = load_image(image_data)
                if not isinstance(apriltag_image, Image.Image):
                    apriltag_image = Image.fromarray(apriltag_image)
            
            # Step 2: Detect gaze on the AprilTag rendered image if gaze detection is available
            if self.gazelle_model and self.gazelle_transform:
                # Convert image to bytes for gaze detection
                apriltag_bytes = pil_image_to_bytes(apriltag_image)
                
                # Run gaze detection
                gaze_results, rendered_image = detect_gaze(
                    image_input=apriltag_bytes,
                    face_detector=self.face_detector,
                    gazelle_model=self.gazelle_model,
                    gazelle_transform=self.gazelle_transform,
                    device=self.device,
                    normalize_bbox=True,
                    normalize_target=True,
                    render=True,
                    show=False,
                    inout_thresh=0.5,
                    render_heatmap=False,
                    save=False
                )
                
                if rendered_image:
                    final_image = rendered_image
                else:
                    final_image = apriltag_image
            else:
                # If gaze detection is unavailable, just use the AprilTag image
                final_image = apriltag_image
                
            # Step 3: Process with VLM using the fully rendered image
            if self.end_to_end:
                messages = generate_end_to_end_prompt_msg(self.action_definitions, final_image)
                
                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    extra_body=self.vlm_extra_body
                )   

                result = response.choices[0].message.content
                if result:
                    result = extract_json_from_codeblock(result)
                    result_obj = json.loads(result)
                else:
                    result_obj = {"error": "Empty response from model"}

                self.logger.info(f"finished analyzing for {base_id}.")
                return jsonify({
                    'observations': result_obj.get('observations', {}),
                    'classifications': result_obj.get('classifications', {}),
                    'justifications': result_obj.get('justifications', {})
                }), 200
            else:
                # Two-step approach (VLM for observation, LLM for classification)
                vlm_messages = generate_vlm_prompt_msg(final_image)
                
                vlm_response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=vlm_messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    extra_body=self.vlm_extra_body
                )
                
                vlm_result = vlm_response.choices[0].message.content
                if vlm_result:
                    vlm_result = extract_json_from_codeblock(vlm_result)
                    vlm_result_obj = json.loads(vlm_result)
                else:
                    vlm_result_obj = {"observations": {}}

                # Process with LLM for text classification
                llm_messages = generate_llm_prompt_msg(vlm_result, self.action_definitions)
                
                llm_response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=llm_messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    extra_body=self.llm_extra_body
                )       
                
                llm_result = llm_response.choices[0].message.content
                if llm_result:
                    llm_result = extract_json_from_codeblock(llm_result)
                    llm_result_obj = json.loads(llm_result)
                else:
                    llm_result_obj = {"classifications": {}, "justifications": {}}

                self.logger.info(f"finished analyzing for {base_id}.")
                return jsonify({
                    'observations': vlm_result_obj.get('observations', {}),
                    'classifications': llm_result_obj.get('classifications', {}),
                    'justifications': llm_result_obj.get('justifications', {})
                }), 200

        except Exception as e:
            self.logger.error("Exception during processing image", exc_info=True)
            return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


def generate_end_to_end_prompt_msg(action_definitions, rendered_image):
    """Generate a prompt message for end-to-end approach with the rendered image.

    Args:
        action_definitions: The definitions of the actions to categorize
        rendered_image: The PIL image with AprilTags and gaze visualization
    Returns:
        Formatted prompt message
    """
    system_content = (
        "You are an expert image analyst for lab experiments. Focus specifically on two key elements: "
        "1) Gaze direction (shown by colored lines) and 2) Hand interactions with objects. "
        "Base all observations ONLY on what is directly visible in the image."
    )

    user_text = (
        "# Lab Image Analysis\n\n"
        "### Visual Elements in Image:\n"
        "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
        "- Colored boxes around faces: These indicate detected faces.\n"
        "- Colored lines from faces: These show EXACT gaze direction (where someone is looking).\n"
        "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        
        "### CRITICAL INTERPRETATION RULES:\n"
        "1. ALWAYS trust the colored gaze lines as the PRIMARY indicator of what someone is looking at. These lines are direct measurements, not inferences.\n"
        "2. If a person has NO colored line extending from their face, or their 'in' probability is low (below 0.5), "
        "this means they are NOT looking at anything within the frame. This person should be classified as 'Distracted' "
        "or similar category as they are not attending to anything visible in the scene.\n"
        "3. The presence of a colored line indicates the person is looking at something within the frame - pay attention "
        "to where this line points to determine what they're focused on.\n"
        "4. Higher 'in' probability values (closer to 1.0) indicate higher confidence that the person is attending to "
        "something within the frame.\n\n"
        
        "### 1. Describe each person by their AprilTag ID number\n"
        "For each person with a visible AprilTag ID number, provide comprehensive observations.\n"
        "For anyone without a visible AprilTag ID number, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"

        "### 2. For each person, describe ONLY these elements:\n"
        "- **Gaze Focus**: MOST CRITICAL - Describe EXACTLY where the colored gaze line points to (if present). This is the primary indicator of attention. If NO gaze line is visible or 'in' probability is low, explicitly state that the person is not looking at anything within the frame.\n"
        "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
        "- **Position**: Basic location in frame for verification only.\n"
        "- **Clothing**: Brief description for identification only.\n\n"

        "### 3. Critical Rules:\n"
        "- For gaze focus: The colored line shows EXACTLY what they're looking at - trust this visual indicator above all else.\n"
        "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
        "- If gaze line or hands aren't clearly visible, explicitly state this.\n"
        "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
        "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"

        "### 4. Classify each person's action\n"
        f"```\n{action_definitions}\n```\n"
        "- Base classifications primarily on 1) gaze focus (where the gaze line points) and 2) hand interactions\n"
        "- People WITHOUT gaze lines or with low 'in' values MUST be classified as 'Distracted' since they are not attending to anything in the frame\n"
        "- If gaze line isn't visible, default to 'Unclear' unless other evidence is definitive\n"
        "- Provide justification focused specifically on gaze focus and hand-object interactions\n\n"

        "### Response Format\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"gaze_focus\": \"Gaze line points directly at computer screen. In probability 0.92 confirms strong attention to screen.\",\n"
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
        "    \"12\": \"Classified as 'Working-Software' because gaze line shows direct attention to computer screen (in: 0.92) and hands are actively interacting with input devices (mouse and keyboard)\",\n"
        "    \"43\": \"Classified as 'Distracted' because the low in probability (0.21) and absence of gaze line indicate the person is not attending to anything visible in the frame\"\n"
        "  }\n"
        "}\n"
        "```"
    )

    # Convert PIL image to base64
    image_bytes = pil_image_to_bytes(rendered_image)
    image_b64 = encode_image_base64(image_bytes)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}}
        ]}
    ]

    return messages


def generate_vlm_prompt_msg(rendered_image):
    """Generate a prompt message for VLM with the rendered image.

    Args:
        rendered_image: The PIL image with AprilTags and gaze visualization
    Returns:
        Formatted prompt message
    """
    system_content = (
        "You are an expert in analyzing images with focus on two specific elements: "
        "1) Gaze direction (shown by colored lines) and 2) Hand interactions with objects. "
        "Describe ONLY what is directly visible with absolute certainty."
    )

    user_text = (
        "# Lab Image Analysis\n\n"
        "### Visual Elements in Image:\n"
        "- Black squares with white numbers: These are AprilTags. Each person has a unique ID tag.\n"
        "- Colored boxes around faces: These indicate detected faces.\n"
        "- Colored lines from faces: These show EXACT gaze direction (where someone is looking).\n"
        "- 'in: X.XX' values: These indicate the confidence that the person is looking at something inside the frame.\n\n"
        
        "### CRITICAL INTERPRETATION RULES:\n"
        "1. ALWAYS trust the colored gaze lines as the PRIMARY indicator of what someone is looking at. These lines are direct measurements, not inferences.\n"
        "2. If a person has NO colored line extending from their face, or their 'in' probability is low (below 0.5), "
        "this means they are NOT looking at anything within the frame. This person is not attending to anything visible in the scene.\n"
        "3. The presence of a colored line indicates the person is looking at something within the frame - pay attention "
        "to where this line points to determine what they're focused on.\n"
        "4. Higher 'in' probability values (closer to 1.0) indicate higher confidence that the person is attending to "
        "something within the frame.\n\n"
        
        "### 1. Describe each person by their AprilTag ID number\n"
        "For each person with a visible AprilTag ID number, provide comprehensive observations.\n"
        "For anyone without a visible AprilTag ID number, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"

        "### 2. For each person, describe ONLY these elements:\n"
        "- **Gaze Focus**: MOST CRITICAL - Describe EXACTLY where the colored gaze line points to (if present). This is the primary indicator of attention. If NO gaze line is visible or 'in' probability is low, explicitly state that the person is not looking at anything within the frame.\n"
        "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
        "- **Position**: Basic location in frame for verification only.\n"
        "- **Clothing**: Brief description for identification only.\n\n"

        "### 3. Critical Rules:\n"
        "- For gaze focus: The colored line shows EXACTLY what they're looking at - trust this visual indicator above all else.\n"
        "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
        "- If gaze line or hands aren't clearly visible, explicitly state this.\n"
        "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
        "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"

        "### Response Format\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"gaze_focus\": \"Gaze line points directly at computer screen. In probability 0.92 confirms strong attention to screen.\",\n"
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

    # Convert PIL image to base64
    image_bytes = pil_image_to_bytes(rendered_image)
    image_b64 = encode_image_base64(image_bytes)
    
    vlm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}}
        ]}
    ]

    return vlm_messages


def generate_llm_prompt_msg(image_description, action_definitions):
    """Generate a prompt message for LLM, asking the LLM to categorize actions based strictly on the provided
    description."""
    system_content = (
        "You are an expert action classifier focused specifically on gaze focus and hand interactions. "
        "Pay special attention to where gaze lines point (or if they're absent), and what objects hands are touching. "
        "Only use information explicitly stated in the observations."
    )

    user_text = (
        "# Action Classification\n\n"
        "### 1. Observation Data:\n"
        f"```\n{image_description}\n```\n\n"

        "### 2. Action Categories:\n"
        f"```\n{action_definitions}\n```\n\n"

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


def extract_json_from_codeblock(text: str) -> str:
    """Extracts JSON content from a JSON code block."""
    if not text:
        return "{}"
        
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


# Helper function to convert PIL Image to bytes
def pil_image_to_bytes(pil_image, format='PNG'):
    """Convert a PIL Image to bytes."""
    img_byte_array = BytesIO()
    pil_image.save(img_byte_array, format=format)
    return img_byte_array.getvalue()
