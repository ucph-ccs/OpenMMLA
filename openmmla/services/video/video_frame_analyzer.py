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
        elif self.backend == 'openai':
            backend_config = self.config['VideoFrameAnalyzer']['openai']
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.api_key = backend_config['api_key']
        self.vlm_model = backend_config['vlm_model']
        self.llm_model = backend_config['llm_model']
        self.vlm_base_url = backend_config.get('vlm_base_url', None)
        self.llm_base_url = backend_config.get('llm_base_url', None)

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
                messages = generate_end_to_end_prompt_msg(id_positions, image_data, self.action_definitions)

                response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    extra_body=self.vlm_extra_body
                )

                result = response.choices[0].message.content
                print(result)
                parsed_result = parse_text_to_dict(result)

                return jsonify({
                    'categorization_result': parsed_result
                }), 200

            else:
                # Process with VLM for image captioning
                vlm_messages = generate_vlm_prompt_msg(id_positions, image_data)
                vlm_response = self.vlm_client.chat.completions.create(
                    model=self.vlm_model,
                    messages=vlm_messages,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    extra_body=self.vlm_extra_body
                )
                image_description = vlm_response.choices[0].message.content

                # Process with LLM for text classification
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


def generate_end_to_end_prompt_msg(id_positions, image_data, action_definitions, ):
    """Generate a prompt message for end-to-end approach.

    Args:
        id_positions: The positions of the detected tags
        image_data: The image data
        action_definitions: The definitions of the actions to categorize
    Returns:
        Formatted prompt message
    """

    system_content = (
        "You are a video ethnographer analyzing a lab experiment. Your task is to classify each person's actions into "
        "defined categories based on their positions and behaviors in the image."
    )
    user_text = (
        "# Analyze Laboratory Image: Step-by-Step Process\n\n"
        "### 1. **Grounding Process**\n"
        "Identify each individual in the image based on their position:\n\n"
        "- **Coordinate System Definition**:\n"
        "  - The bottom-left corner of the image is **(0, 0)**.\n"
        "  - The top-right corner of the image is **(1, 1)**.\n"
        "  - `'Left'` and `'Right'` are determined by comparing the **x-coordinates** of the positions:\n"
        "    - Lower x means `'Left'`.\n"
        "    - Higher x means `'Right'`.\n"
        "  - `'Top'` and `'Bottom'` are determined by comparing the **y-coordinates** of the positions:\n"
        "    - Lower y means `'closer to Bottom'`.\n"
        "    - Higher y means `'closer to Top'`.\n\n"
        "- Validate the detected individuals listed below by matching their positions.\n"
        "- For any additional individuals not listed, assign them a unique ID (e.g., `<id>` where `id` is a new "
        "number) and note their approximate position.\n\n"
        "**Detected Individuals and Their Positions**:\n"
        f"```\n{id_positions}\n```\n\n"
        "---\n\n"
        "### 2. **Captioning Process**\n"
        "For each individual (both detected and undetected), describe only what you can see:\n\n"
        "- **Position**: Relative location in the image (e.g., Left, Center, Right, Top, Bottom).\n"
        "- **Posture & Orientation**: Describe their posture (e.g., seated, standing) and where they are facing.\n"
        "- **Gaze Direction**: Indicate where they appear to be looking (e.g., at another person, at a device).\n"
        "- **Hands & Objects**: Describe the state of their hands and any objects they are interacting with (e.g., "
        "holding a arduino board/raspberry pi/iPad/Phone, touching keyboard/mouse/screen/keyboard).\n"
        "- **Interactions**: Clearly describe any visible interactions with other people or objects.\n\n"
        "---\n\n"
        "### 3. **Action Classification**\n"
        "Based on the observations from the captioning process, classify each person's action into one of these "
        "categories:\n"
        f"```\n{action_definitions}\n```\n\n"
        "**Classification Guidelines**\n"
        "- Match each person's described behavior to the most appropriate action category\n"
        "- Consider all visible details including posture, gaze, and interactions\n"
        "- Base classifications only on what is explicitly described\n\n"
        "---\n\n"
        "### Response Format\n"
        "Return your response in **two stages**:\n\n"
        "#### a) **Grounding and Captioning**\n"
        "Provide the position and description for each person:\n"
        "```\n"
        "ID [id]:\n"
        "- Position: [brief location in the image]\n"
        "- Visible Details: [specific observations, avoid assumptions]\n"
        "```\n\n"
        "#### b) **Action Classification**\n"
        "Map each person's ID to one of the action categories:\n"
        "```\n"
        "{7: 'Communicating', 8: 'Working-Software', 9: 'Distracted'}\n"
        "```\n\n"
        "---\n\n"
        "**Instructions**:\n"
        "Take a deep breath and follow the process step by step, ensuring grounding and description are accurate "
        "before action classification."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": [{"type": "text", "text": user_text},
                                     {"type": "image_url", "image_url": {"url": encode_image_base64(image_data)}}]}
    ]


def generate_vlm_prompt_msg(id_positions=None, image_input=None):
    """Generate a prompt message for VLM, asking the VLM to describe the individuals in the image."""
    system_content = (
        "You are a video ethnographer analyzing a lab experiment. Your task is to identify and describe each person's "
        "position and behaviors in the image."
    )
    user_text = (
        "# Analyze Laboratory Image: Step-by-Step Process\n\n"
        "### 1. **Grounding Process**\n"
        "Identify each individual in the image based on their position:\n\n"
        "- **Coordinate System Definition**:\n"
        "  - The bottom-left corner of the image is **(0, 0)**.\n"
        "  - The top-right corner of the image is **(1, 1)**.\n"
        "  - `'Left'` and `'Right'` are determined by comparing the **x-coordinates** of the positions:\n"
        "    - Lower x means `'Left'`.\n"
        "    - Higher x means `'Right'`.\n"
        "  - `'Top'` and `'Bottom'` are determined by comparing the **y-coordinates** of the positions:\n"
        "    - Lower y means `'closer to Bottom'`.\n"
        "    - Higher y means `'closer to Top'`.\n\n"
        "- Validate the detected individuals listed below by matching their positions.\n"
        "- For any additional individuals not listed, assign them a unique ID (e.g., `<id>` where <id> is a new "
        "number) and note their approximate position.\n\n"
        "**Detected Individuals and Their Positions**:\n"
        f"```\n{id_positions}\n```\n\n"
        "---\n\n"
        "### 2. **Captioning Process**\n"
        "For each individual (both detected and undetected), describe only what you can see:\n\n"
        "- **Position**: Relative location in the image (e.g., Left, Center, Right, Top, Bottom).\n"
        "- **Posture & Orientation**: Describe their posture (e.g., seated, standing) and where they are facing.\n"
        "- **Gaze Direction**: Indicate where they appear to be looking (e.g., at another person, at a device).\n"
        "- **Hands & Objects**: Describe the state of their hands and any objects they are interacting with (e.g., "
        "holding a arduino board/raspberry pi/iPad/Phone, touching keyboard/mouse/screen/keyboard).\n"
        "- **Interactions**: Clearly describe any visible interactions with other people or objects.\n\n"
        "### Response Format\n"
        "Provide the position and description for each person:\n"
        "```\n"
        "ID [id]:\n"
        "- Position: [brief location in the image]\n"
        "- Visible Details: [specific observations, avoid assumptions]\n"
        "```\n\n"
        "**Instructions**:\n"
        "Take a deep breath and follow the process step by step, ensuring grounding and description are accurate."
    )

    if image_input:
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
    """Generate a prompt message for LLM, asking the LLM to categorize actions."""
    system_content = (
        "You are a video ethnographer specializing in action classification. Your task is to categorize each person's "
        "actions into defined categories based on their behaviors described in the image."
    )

    user_text = (
        "# Action Classification Process\n\n"
        "### 1. Review the Image Description\n"
        "Below is a detailed description of individuals in a laboratory setting:\n\n"
        f"```\n{image_description}\n```\n\n"
        "### 2. Action Categories\n"
        "Classify each person's action into one of these defined categories:\n"
        f"```\n{action_definitions}\n```\n\n"
        "### 3. Classification Guidelines\n"
        "- Match each person's described behavior to the most appropriate action category\n"
        "- Consider all visible details including posture, gaze, and interactions\n"
        "- Base classifications only on what is explicitly described\n\n"
        "### Response Format\n"
        "Return your classification as a dictionary mapping IDs to actions:\n"
        "```\n"
        "{7: 'Communicating', 8: 'Working-Software', 9: 'Distracted'}\n"
        "```\n\n"
        "**Instructions**:\n"
        "Take a deep breath and classify each person's actions carefully based on the provided description."
    )

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
    pattern = r"[\"'`]?(\d+)[\"'`]?\s*:\s*[\"'`]?([\w\s-]+)[\"'`]?"
    matches = re.finditer(pattern, ai_message)
    return {match.group(1).strip(): match.group(2).strip() for match in matches}
