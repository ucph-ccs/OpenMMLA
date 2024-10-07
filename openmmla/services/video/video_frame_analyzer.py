import re

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

        if self.backend == 'ollama':
            backend_config = self.config['VideoFrameAnalyzer']['ollama']
        elif self.backend == 'vllm':
            backend_config = self.config['VideoFrameAnalyzer']['vllm']
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.api_key = backend_config['api_key']
        self.vlm_base_url = backend_config['vlm_base_url']
        self.llm_base_url = backend_config['llm_base_url']
        self.vlm_model = backend_config['vlm_model']
        self.llm_model = backend_config['llm_model']

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

    def process_request(self):
        """Process the image."""
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        try:
            image_file = request.files['image']
            image_data = image_file.read()

            # Detect AprilTags
            id_positions = detect_apriltags(image_data, self.detector, show=False)

            # Process with VLM
            vlm_messages = generate_vlm_prompt_msg(id_positions, image_data)
            vlm_response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=vlm_messages,
                top_p=self.top_p,
                temperature=self.temperature,
                extra_body=self.vlm_extra_body
            )
            image_description = vlm_response.choices[0].message.content

            # Process with LLM
            llm_messages = generate_llm_prompt_msg(image_description, self.action_definitions)
            llm_response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=llm_messages,
                top_p=self.top_p,
                temperature=self.temperature
            )
            categorization_result = llm_response.choices[0].message.content
            parsed_result = parse_text_to_dict(categorization_result)

            return jsonify({
                'image_description': image_description,
                'categorization_result': parsed_result
            }), 200

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500


def generate_vlm_prompt_msg(id_position_dict=None, image_input=None):
    """Generate a prompt message for VLM, asking the VLM to describe the individuals in the image.

    Args:
        id_position_dict: Dictionary containing the position of each person in the image
        image_input: Either a string (file path) or bytes (image data)

    Returns:
        Formatted prompt message
    """
    if not id_position_dict:
        user_text = (
            "Analyze this laboratory image:\n\n"
            "For each person you can identify in the image:\n"
            "1. Assign a unique ID (e.g., Person 1, Person 2, etc.)\n"
            "2. Describe only what you can see:\n"
            "   a) Approximate position in the image\n"
            "   b) Posture & orientation\n"
            "   c) Gaze direction\n"
            "   d) Hands & objects (if visible)\n"
            "   e) Clear interactions\n"
            "\nFormat:\n"
            "Person [X]:\n"
            "- Position: [brief location]\n"
            "- Visible details: [key observations]\n"
            "\nBe specific about what you can see and what you can't. "
            "If hands or any other part is not visible, state this explicitly. "
            "Do not make assumptions about unseen elements. "
            "Analyze each person you can identify in the image, one by one."
        )
    else:
        user_text = (
            f"Analyze this laboratory image with {len(id_position_dict)} individuals:\n"
        )

        for person_id, position in id_position_dict.items():
            user_text += f"- ID {person_id}: pos {position}\n"

        user_text += (
            "\nFor each person:\n"
            "1. Confirm ID based on position.\n"
            "2. Describe only what you can see:\n"
            "   a) Posture & orientation\n"
            "   b) Gaze direction\n"
            "   c) Hands & objects (if visible)\n"
            "   d) Clear interactions\n"
            "\nFormat:\n"
            "ID [X]:\n"
            "- Position: [brief location]\n"
            "- Visible details: [key observations]\n"
            "\nBe specific about what you can see and what you can't. "
            "If hands or any other part is not visible, state this explicitly. "
            "Do not make assumptions about unseen elements."
        )

    user_text += " Take a deep breath and do the analysis step by step."

    system_content = (
        "You're a capable video ethnographer analyzing a lab experiment."
    )

    if image_input:  # OpenAI API format message
        image_b64 = encode_image_base64(image_input)
        vlm_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [{"type": "text", "text": user_text},
                                         {"type": "image_url", "image_url": {"url": image_b64}}]}
        ]
    else:
        vlm_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"(<image>./</image>)\n{user_text}"}
        ]

    return vlm_messages


def generate_llm_prompt_msg(image_description, action_definitions):
    """Generate a prompt message for LLM, asking the LLM to categorize actions.

    Args:
        image_description: Description of the image generated by VLM
        action_definitions: Definitions of the actions to categorize

    Returns:
        Formatted prompt message
    """
    user_text = (f"You will now assist with analysing an image description text. Your task is to first categorize each"
                 f"individual's action from the given image description into one of the defined action classes from"
                 f"the action definitions. Then, format the categorization results into a target format.\n\n"
                 f"Image Description:\n{image_description}\n\n"
                 f"Action Definitions:\n{action_definitions}\n\n"
                 f"Carefully read the image description and action definitions and proceed with the following instructions:\n"
                 f"1. Categorize their action into one of the defined classes.\n"
                 f"2. Format your categorization results into a dictionary string with <ID> : <Action> key value pairs,"
                 f" for example, {{'1' : 'Action_1', '7' : 'Action_2'}}.\n"
                 f"Take a deep breath and do the analysis step by step.")

    system_content = (
        "You are now a taxonomist and sociologist who is good at categorizing human behaviors based on the given "
        "context of image description and actions definition. ")

    llm_messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_text}
    ]

    return llm_messages


def parse_text_to_dict(ai_message: str):
    """Parse a string containing key-value pairs and return a dictionary.

    Args:
        ai_message: The input string to parse.
    Returns:
        A dictionary with keys as ids and values as action descriptions.
    """
    pattern = r"[\"'`]?(\d+)[\"'`]?\s*:\s*[\"'`]?([\w\s]+)[\"'`]?"
    matches = re.finditer(pattern, ai_message)
    return {match.group(1).strip(): match.group(2).strip() for match in matches}
