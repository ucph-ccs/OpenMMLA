import json

from flask import request, jsonify
from openai import OpenAI
from pupil_apriltags import Detector

from openmmla.services.server import Server
from openmmla.utils.video.apriltag import detect_apriltags
from openmmla.utils.video.image import encode_image_base64


class VideoFrameAnalyzer(Server):
    """Video frame analyzer receives images, processes them with VLM and LLM, and returns the results."""

    def __init__(self, project_dir, config_path):
        super().__init__(project_dir=project_dir, config_path=config_path)

        self.families = self.config['AprilTag']['families']
        self.backend = self.config['VideoFrameAnalyzer']['backend']
        self.top_p = float(self.config['VideoFrameAnalyzer']['top_p'])
        self.temperature = float(self.config['VideoFrameAnalyzer']['temperature'])

        if self.backend in ['ollama', 'vllm', 'openai', 'qwen', 'gemini', 'deepseek']:
            backend_config = self.config['VideoFrameAnalyzer'][self.backend]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.api_key = backend_config['api_key']
        self.vlm_model = backend_config['vlm_model']
        self.llm_model = backend_config['llm_model']
        self.vlm_base_url = backend_config.get('vlm_base_url', None)
        self.llm_base_url = backend_config.get('llm_base_url', None)

        print(f"API Key: {self.api_key}")
        print(f"VLM Model: {self.vlm_model}")
        print(f"LLM Model: {self.llm_model}")
        print(f"VLM Base URL: {self.vlm_base_url}")
        print(f"LLM Base URL: {self.llm_base_url}")

        self.detector = Detector(families=self.families, nthreads=4)

        self.vlm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.vlm_base_url,
        )

        self.llm_client = OpenAI(
            api_key=self.api_key,
            base_url=self.llm_base_url,
        )

        # Load action definitions from config
        self.action_definitions = '\n'.join(
            [f"'{key}': {value}" for key, value in self.config['VideoFrameAnalyzer']['defined_actions'].items()])

        # Load VLM extra body from config
        self.vlm_extra_body = backend_config.get('VLMExtraBody', {})
        self.llm_extra_body = backend_config.get('LLMExtraBody', {})

    def process_request(self, end_to_end=False):
        """Process the image with option for end-to-end or two-step approach.
        
        Args:
            end_to_end (bool): If True, use single API call for direct classification.
                              If False, use separate calls for description and classification.
        """
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        try:
            image_file = request.files['image']
            image_data = image_file.read()

            # Detect AprilTags
            id_positions = detect_apriltags(image_data, self.detector, show=False)

            if end_to_end:
                # Direct classification with single API call
                messages, response_format = generate_end_to_end_prompt_msg(id_positions, image_data,
                                                                           self.action_definitions)

                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format=response_format,
                    extra_body=self.vlm_extra_body
                )

                result = response.choices[0].message.content
                result_obj = json.loads(result)

                return jsonify({
                    'observations': result_obj.get('observations', {}),
                    'classifications': result_obj.get('classifications', {}),
                    'justifications': result_obj.get('justifications', {})
                }), 200

            else:
                # Process with VLM for image captioning
                vlm_messages, response_format = generate_vlm_prompt_msg(id_positions, image_data)
                vlm_response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=vlm_messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format=response_format,
                    extra_body=self.vlm_extra_body
                )
                vlm_result = vlm_response.choices[0].message.content
                vlm_result_obj = json.loads(vlm_result)

                # Process with LLM for text classification
                llm_messages, llm_response_format = generate_llm_prompt_msg(vlm_result, self.action_definitions)
                llm_response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=llm_messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    response_format=llm_response_format,
                    extra_body=self.llm_extra_body
                )
                llm_result = llm_response.choices[0].message.content
                llm_result_obj = json.loads(llm_result)

                return jsonify({
                    'observations': vlm_result_obj.get('observations', {}),
                    'classifications': llm_result_obj.get('classifications', {}),
                    'justifications': llm_result_obj.get('justifications', {})
                }), 200

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500


def generate_end_to_end_prompt_msg(id_positions, image_data, action_definitions):
    """Generate a prompt message for end-to-end approach.

    Args:
        id_positions: The positions of the detected tags
        image_data: The image data
        action_definitions: The definitions of the actions to categorize
    Returns:
        Tuple of (formatted prompt message, response format)
    """

    system_content = (
        "You are an expert image analyst for lab experiments. Focus specifically on two key elements: "
        "1) Precise hand interactions with objects, and 2) Exact face/head direction. "
        "Base all observations ONLY on what is directly visible in the image."
    )

    user_text = (
        "# Lab Image Analysis\n\n"
        "### 1. First identify everyone\n"
        "Match each person with these AprilTag IDs (coordinates are from bottom-left (0,0) to top-right (1,1)):\n"
        f"```\n{id_positions}\n```\n"
        "Label anyone without a tag as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"
        
        "### 2. For each person, describe ONLY these elements:\n"
        "- **Face/Head Direction**: CRITICAL - Describe the EXACT physical orientation of the head and what specific object/area it's facing toward. Use precise terms like: front view, profile left/right, 3/4 view left/right, or facing away. Always indicate what the face is pointed toward (e.g., \"Profile view facing frame-right, directly toward whiteboard\").\n"
        "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
        "- **Position**: Basic location in frame (coordinates + brief description) for verification only.\n"
        "- **Clothing**: Brief description for identification only.\n\n"
        
        "### 3. Critical Rules:\n"
        "- For face direction: ONLY describe physical orientation of the head itself, not eye gaze. Always specify what object/person/area the face is physically pointed toward.\n"
        "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
        "- If face direction or hands aren't clearly visible, explicitly state this and explain why (e.g., \"Face not visible, obscured by monitor\" or \"Left hand not visible, obscured by desk\").\n"
        "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
        "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"
        
        "### 4. Classify each person's action\n"
        f"```\n{action_definitions}\n```\n"
        "- Base classifications primarily on 1) face/head direction and 2) hand interactions\n"
        "- If either face direction or hands aren't clearly visible, default to 'Unclear' unless other evidence is definitive\n"
        "- Provide justification focused specifically on face direction and hand-object interactions\n\n"
        
        "### Response Format\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"face_direction\": \"Three-quarter view facing frame-left, oriented directly toward computer monitor 40cm in front of person\",\n"
        "      \"hands_status\": \"Right hand on mouse with fingertips touching left and right buttons; left hand on keyboard with fingers on ASDF keys. Both hands actively touching and manipulating these input devices\",\n"
        "      \"position\": \"Center-right (0.7, 0.5)\",\n"
        "      \"clothing\": \"Dark blue shirt\"\n"
        "    },\n"
        "    \"43\": {\n"
        "      \"face_direction\": \"Direction not clearly visible due to distance and lighting\",\n"
        "      \"hands_status\": \"Both hands hanging at sides, not in contact with any objects. Fingers relaxed and slightly curled, palms facing inward toward thighs\",\n"
        "      \"position\": \"Left side (0.15, 0.5)\",\n"
        "      \"clothing\": \"Light t-shirt, dark pants\"\n"
        "    }\n"
        "  },\n"
        "  \"classifications\": {\n"
        "    \"12\": \"Working-Software\",\n"
        "    \"43\": \"Unclear\"\n"
        "  },\n"
        "  \"justifications\": {\n"
        "    \"12\": \"Classified as 'Working-Software' because face is oriented directly toward computer monitor and hands are actively interacting with input devices (mouse and keyboard)\",\n"
        "    \"43\": \"Classified as 'Unclear' because face direction is not visible and hands are not interacting with any objects\"\n"
        "  }\n"
        "}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [{"type": "text", "text": user_text},
                                     {"type": "image_url",
                                      "image_url": {"url": encode_image_base64(image_data), "detail": "high"}}]}
    ]

    response_format = {
        "type": "json_object"
    }

    return messages, response_format


def generate_vlm_prompt_msg(id_positions=None, image_input=None):
    """Generate a prompt message for VLM, asking the VLM to describe the individuals in the image."""
    system_content = (
        "You are an expert in analyzing images with focus on two specific elements: "
        "1) Precise face/head direction and 2) Exact hand interactions with objects. "
        "Describe ONLY what is directly visible with absolute certainty."
    )

    user_text = (
        "# Lab Image Analysis\n\n"
        "### 1. First identify everyone\n"
        "Match each person with these AprilTag IDs (coordinates are from bottom-left (0,0) to top-right (1,1)):\n"
        f"```\n{id_positions}\n```\n"
        "Label anyone without a tag as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n\n"
        
        "### 2. For each person, describe ONLY these elements:\n"
        "- **Face/Head Direction**: CRITICAL - Describe the EXACT physical orientation of the head and what specific object/area it's facing toward. Use precise terms like: front view, profile left/right, 3/4 view left/right, or facing away. Always indicate what the face is pointed toward (e.g., \"Profile view facing frame-right, directly toward whiteboard\").\n"
        "- **Hands Status**: CRITICAL - Describe EXACTLY what the hands are doing. Are they touching/holding objects? What specific part of the object? Describe precise contact points (e.g., \"Right hand resting on mouse with index finger on left button; left hand on keyboard home row\").\n"
        "- **Position**: Basic location in frame (coordinates + brief description) for verification only.\n"
        "- **Clothing**: Brief description for identification only.\n\n"
        
        "### 3. Critical Rules:\n"
        "- For face direction: ONLY describe physical orientation of the head itself, not eye gaze. Always specify what object/person/area the face is physically pointed toward.\n"
        "- For hands: Be EXTREMELY SPECIFIC about whether hands are touching/interacting with objects. State exactly which part of the hand contacts which part of the object.\n"
        "- If face direction or hands aren't clearly visible, explicitly state this and explain why (e.g., \"Face not visible, obscured by monitor\" or \"Left hand not visible, obscured by desk\").\n"
        "- Never assume a hand is touching something unless you can clearly see the point of contact.\n"
        "- Use camera perspective for spatial references (left/right/top/bottom).\n\n"
        
        "### Response Format\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"face_direction\": \"Three-quarter view facing frame-left, oriented directly toward computer monitor 40cm in front of person\",\n"
        "      \"hands_status\": \"Right hand on mouse with fingertips touching left and right buttons; left hand on keyboard with fingers on ASDF keys. Both hands actively touching and manipulating these input devices\",\n"
        "      \"position\": \"Center-right (0.7, 0.5)\",\n"
        "      \"clothing\": \"Dark blue shirt\"\n"
        "    },\n"
        "    \"43\": {\n"
        "      \"face_direction\": \"Direction not clearly visible due to distance and lighting\",\n"
        "      \"hands_status\": \"Both hands hanging at sides, not in contact with any objects. Fingers relaxed and slightly curled, palms facing inward toward thighs\",\n"
        "      \"position\": \"Left side (0.15, 0.5)\",\n"
        "      \"clothing\": \"Light t-shirt, dark pants\"\n"
        "    }\n"
        "  }\n"
        "}\n"
        "```"
    )

    if image_input:
        image_b64 = encode_image_base64(image_input)
        vlm_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [{"type": "text", "text": user_text},
                                         {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}}]}
        ]
    else:
        vlm_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"(<image>./</image>)\n{user_text}"}
        ]

    response_format = {
        "type": "json_object"
    }

    return vlm_messages, response_format


def generate_llm_prompt_msg(image_description, action_definitions):
    """Generate a prompt message for LLM, asking the LLM to categorize actions based strictly on the provided
    description."""
    system_content = (
        "You are an expert action classifier focused specifically on face direction and hand interactions. "
        "Pay special attention to where faces are oriented and what objects hands are touching. "
        "Only use information explicitly stated in the observations."
    )

    user_text = (
        "# Action Classification\n\n"
        "### 1. Observation Data:\n"
        f"```\n{image_description}\n```\n\n"
        
        "### 2. Action Categories:\n"
        f"```\n{action_definitions}\n```\n\n"
        
        "### 3. Classification Rules:\n"
        "- Give EQUAL WEIGHT to these two critical factors:\n"
        "  1. Face/head direction - where the face is physically pointed\n"
        "  2. Hand interactions - what objects hands are touching\n"
        "- If face isn't described as physically oriented toward an object/screen, don't assume attention to it\n"
        "- If hands aren't described as physically touching/manipulating objects, don't assume interaction\n"
        "- If either face direction or hands are described as \"not visible\" or \"unclear,\" default to 'Unclear' unless other evidence is definitive\n"
        "- Position and clothing should not factor into classification\n\n"
        
        "### Response Format\n"
        "```json\n"
        "{\n"
        "  \"classifications\": {\n"
        "    \"12\": \"Working-Software\",\n"
        "    \"43\": \"Unclear\"\n"
        "  },\n"
        "  \"justifications\": {\n"
        "    \"12\": \"Classified as 'Working-Software' because face is oriented directly toward computer monitor and hands are described as actively interacting with keyboard and mouse\",\n"
        "    \"43\": \"Classified as 'Unclear' because description states face direction is not visible and hands are not interacting with any objects\"\n"
        "  }\n"
        "}\n"
        "```"
    )

    llm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_text}
    ]

    llm_response_format = {
        "type": "json_object"
    }

    return llm_messages, llm_response_format
