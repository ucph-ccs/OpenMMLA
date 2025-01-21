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
        "- For any additional individuals not listed, assign them a unique ID (e.g., `Person [X]` where X is a new "
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
        "holding a arduino board/raspberry pi/iPad/Phone, touching keyboard/mouse/screen/keyboard.\n"
        "- **Interactions**: Clearly describe any visible interactions with other people or objects.\n\n"
        "---\n\n"
        "### 3. **Action Classification**\n"
        "Based on the observations from the captioning process, classify each person's action into one of these "
        "categories:\n"
        f"```\n{action_definitions}\n```\n\n"
        "---\n\n"
        "### Response Format\n"
        "Return your response in **two stages**:\n\n"
        "#### a) **Grounding and Captioning**\n"
        "Provide the position and description for each person:\n"
        "```\n"
        "ID [X]:\n"
        "- Position: [brief location in the image]\n"
        "- Visible Details: [specific observations, avoid assumptions]\n"
        "```\n\n"
        "#### b) **Action Classification**\n"
        "Map each person's ID to one of the action categories:\n"
        "```\n"
        "{7: 'Communicating', 8: 'Working-Software', 'Person 9': 'Distracted'}\n"
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
    """Generate a prompt message for VLM, asking the VLM to describe the individuals in the image.

    Args:
        id_positions: Dictionary containing the position of each person in the image
        image_input: Either a string (file path) or bytes (image data)

    Returns:
        Formatted prompt message
    """
    system_content = (
        "You're a capable video ethnographer analyzing a lab experiment."
    )
    user_text = (
        f"Analyze this laboratory image with the detected individuals' IDs and their center "
        f"positions.\n"
        f"The coordinate system is defined as follows:\n"
        f" - The bottom-left corner of the image is (0, 0).\n"
        f" - The top-right corner of the image is (1, 1).\n"
        f" - 'Left' and 'Right' are determined by comparing the x-coordinates of the positions "
        f"(lower x means 'left', higher x means 'right').\n"
        f" - 'Top' and 'Bottom' are determined by comparing the y-coordinates of the positions "
        f"(lower y means 'closer to bottom', higher y means 'closer to top').\n\n"
        f"Here are the individuals detected in the image and their positions:\n"
        f"{id_positions}\n\n"
        f"Analyze each person as follows:\n"
        f"1. **Detected Persons**: For individuals listed in the provided dictionary, validate their position and "
        f"describe them using the steps below.\n"
        f"2. **Undetected Persons**: Identify any additional individuals not listed in the dictionary and analyze "
        f"them in the same way.\n\n"
        f"For each person, provide:\n"
        f"a) **Posture & Orientation**: Describe their posture (e.g., seated, standing) and where they are facing.\n"
        f"b) **Gaze Direction**: Indicate where they appear to be looking (e.g., at another person, at a device).\n"
        f"c) **Hands & Objects**: Describe the state of their hands and any objects they are interacting with ("
        f"e.g., holding a arduino board/raspberry pi/iPad/Phone, touching keyboard/mouse/screen/keyboard).\n"
        f"d) **Interactions**: Clearly describe any visible interactions with other people or objects.\n\n"
        f"Format your response for each person as follows:\n"
        f"ID [X]:\n"
        f"- Position: [relative location in the image, left/right, top/bottom]\n"
        f"- Visible Details: [specific observations, avoid assumptions]\n\n"
        f"Example:\n"
        f"ID 1:\n"
        f"- Position: [Left, Top]\n"
        f"- Visible Details: Seated, facing right. Hands holding a laptop, gaze directed at the screen.\n\n"
        f"Be specific about what you can and cannot see. For example:\n"
        f"- If hands, gaze, or any part of the body is not visible, state this explicitly.\n"
        f"- Do not make assumptions about unseen parts of the person or their behavior.\n\n"
        f"Analyze each person carefully, step by step, ensuring that both detected and undetected individuals are "
        f"included in your analysis. If you identify additional persons not listed in the dictionary, assign them a "
        f"unique ID (e.g., 'Person [X]' where X is a new number) and describe them using the same format."
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
    system_content = (
        "You are now a taxonomist and sociologist who is good at categorizing human behaviors based on the given "
        "context of image description and actions definition. ")

    user_text = (f"You will now assist with analysing an image description text. Your task is to first categorize each"
                 f"individual's action from the given image description into one of the defined action classes from"
                 f"the action definitions. Then, format the categorization results into a target format.\n\n"
                 f"Image Description:\n{image_description}\n\n"
                 f"Action Definitions:\n{action_definitions}\n\n"
                 f"Carefully read the image description and action definitions and proceed with the following "
                 f"instructions:\n"
                 f"1. Categorize their actions into one of the defined classes.\n"
                 f"2. Format your categorization results into a dictionary string with <ID> : <Action> key value pairs,"
                 f" for example, {{'1' : 'Action_1', '7' : 'Action_2'}}.\n"
                 f"Take a deep breath and do the analysis step by step.")

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
