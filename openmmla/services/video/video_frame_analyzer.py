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
        "You are a precise video ethnographer analyzing a lab experiment. Your task is to identify, describe, and classify each person's "
        "actions based SOLELY on what is directly visible in the image. Never make assumptions about what might be happening outside "
        "the frame or what people might be thinking. If something is not clearly visible, explicitly state this uncertainty."
    )

    user_text = (
        "# Analyze Laboratory Image: FACTS ONLY\n\n"
        "### 1. **Grounding Process**\n"
        "Identify each individual in the image using a two-step process:\n\n"
        "**STEP 1: First match the EXACT IDs listed below**\n"
        "The following individuals have already been detected with AprilTags. YOU MUST use these EXACT IDs:\n"
        f"```\n{id_positions}\n```\n\n"
        "**STEP 2: Only after matching all the above IDs**\n"
        "If you see additional people without AprilTags, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n"
        "IMPORTANT: Only use \"Untagged Person\" labels for individuals who DO NOT match any of the IDs listed above.\n\n"
        "**Coordinate System Definition**:\n"
        "  - The bottom-left corner of the image is **(0, 0)**.\n"
        "  - The top-right corner of the image is **(1, 1)**.\n"
        "  - `'Left'` and `'Right'` are determined by comparing the **x-coordinates** of the positions:\n"
        "    - Lower x means `'Left'`.\n"
        "    - Higher x means `'Right'`.\n"
        "  - `'Top'` and `'Bottom'` are determined by comparing the **y-coordinates** of the positions:\n"
        "    - Lower y means `'closer to Bottom'`.\n"
        "    - Higher y means `'closer to Top'`.\n\n"
        "---\n\n"
        "### 2. **Objective Observation Only**\n"
        "For each individual, describe ONLY what you can DIRECTLY observe in EXTENSIVE DETAIL:\n\n"
        "- **Position**: Precise location in the image (e.g., Left side, Center-right, Upper-left quadrant).\n"
        "- **Posture**: Detailed body position (e.g., seated with back straight, standing with weight on left leg, leaning forward).\n"
        "- **Gaze Direction**: ONLY if clearly visible in the frame. If their face is not visible or gaze direction is ambiguous, state \"gaze direction not clearly visible\".\n"
        "- **Hands & Objects**: ONLY describe hand positions that are CLEARLY VISIBLE. Do NOT assume hand positions that aren't fully visible. Use descriptions like:\n"
        "  - \"Hands resting on table, not holding any objects\"\n"
        "  - \"Hands clasped/interlaced together in lap\"\n"
        "  - \"Arms crossed across chest, hands not interacting with any objects\"\n"
        "  - \"Right hand visible resting on armrest, left hand not visible\"\n"
        "  - \"Both hands partially visible but exact positions and any objects being held cannot be determined\"\n"
        "  - If you cannot clearly see if hands are holding objects, state this explicitly\n"
        "- **Interactions**: ONLY interactions that are completely contained within the frame. Be specific about the nature of any interactions.\n"
        "- **Clothing/Appearance**: Note relevant details about clothing or distinctive appearance features if visible.\n\n"
        "### 3. **Strict Rules for Descriptions**\n"
        "- PROVIDE EXHAUSTIVE DETAIL in your descriptions - avoid brevity\n"
        "- Do NOT infer what someone might be looking at outside the frame\n"
        "- Do NOT assume conversations are happening if you cannot see clear evidence\n"
        "- Do NOT interpret facial expressions unless they are clearly visible\n"
        "- Do NOT make assumptions about intent or thoughts\n"
        "- Do NOT guess at activities that might be happening outside the frame\n"
        "- NEVER assume a hand is holding an object unless you can clearly see both the hand and the object\n"
        "- Do NOT try to guess what specific type of object someone is holding if it's not clearly visible\n"
        "- Use phrases like \"appears to be\" for any description that isn't 100% certain\n"
        "- When in doubt, note uncertainty rather than guessing\n\n"
        "---\n\n"
        "### 4. **Action Classification**\n"
        "Based ONLY on your objective observations, classify each person's action into one of these categories:\n"
        f"```\n{action_definitions}\n```\n\n"
        "**Classification Rules**\n"
        "- ONLY use information explicitly stated in your observations\n"
        "- If you noted something is \"not visible\" or \"unclear,\" do NOT use it for classification\n"
        "- If the observed behavior doesn't clearly fit a category, classify as 'Unclear'\n"
        "- Do NOT make assumptions about what might be happening\n"
        "- Require clear evidence in your observations to assign a non-'Unclear' category\n"
        "- When in doubt, classify as 'Unclear' rather than making assumptions\n\n"
        "---\n\n"
        "### Response Format\n"
        "Return your response as a JSON object with the following structure (the IDs below are just EXAMPLES, use the actual IDs from the image):\n\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"position\": \"Center-right of the frame, approximately at coordinates (0.7, 0.5)\",\n"
        "      \"posture\": \"Seated at desk with back slightly hunched forward, upper body fully visible with shoulders relaxed\",\n"
        "      \"gaze\": \"Eyes clearly directed downward at visible laptop screen positioned on desk\",\n"
        "      \"hands_objects\": \"Right hand positioned on visible keyboard, fingers spread over keys; left hand resting on desk surface, not holding any objects\",\n"
        "      \"interactions\": \"Actively typing on keyboard with right hand, no interaction with other people visible in frame\",\n"
        "      \"clothing\": \"Wearing a dark blue long-sleeve shirt with collar visible\"\n"
        "    },\n"
        "    \"43\": {\n"
        "      \"position\": \"Left side of frame near wall\",\n"
        "      \"posture\": \"Standing with weight distributed evenly, facing toward center of room\",\n"
        "      \"gaze\": \"Gaze direction not clearly visible due to distance from camera and lighting conditions\",\n"
        "      \"hands_objects\": \"Both hands visible at sides, not holding any objects. Fingers appear to be relaxed\",\n"
        "      \"interactions\": \"No clear interactions visible within the frame\",\n"
        "      \"clothing\": \"Wearing what appears to be a light-colored t-shirt and dark pants\"\n"
        "    },\n"
        "    \"Untagged Person 1\": {\n"
        "      \"position\": \"Bottom-right corner of frame, partially visible\",\n"
        "      \"posture\": \"Seated at what appears to be a desk or table, only upper torso visible in frame\",\n"
        "      \"gaze\": \"Appears to be looking at visible computer monitor directly in front of them, eyes clearly focused on screen\",\n"
        "      \"hands_objects\": \"Hands visible and resting in lap, interlaced with fingers clasped together. No objects being held\",\n"
        "      \"interactions\": \"No interaction with other individuals visible\",\n"
        "      \"clothing\": \"Wearing a red and white patterned shirt, possibly plaid or checkered\"\n"
        "    }\n"
        "  },\n"
        "  \"classifications\": {\n"
        "    \"12\": \"Working-Software\",\n"
        "    \"43\": \"Unclear\",\n"
        "    \"Untagged Person 1\": \"Observing\"\n"
        "  },\n"
        "  \"justifications\": {\n"
        "    \"12\": \"Classified as 'Working-Software' because observation clearly shows they are typing on a keyboard while looking at computer screen, indicating direct software interaction\",\n"
        "    \"43\": \"Classified as 'Unclear' because gaze direction is not clearly visible and no specific activity is observable\",\n"
        "    \"Untagged Person 1\": \"Classified as 'Observing' because they are clearly looking at a computer monitor without evidence of typing or software manipulation, suggesting they are viewing content\"\n"
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
        "You are a precise video analyst tasked with ONLY describing what is DIRECTLY VISIBLE in the image in EXTENSIVE DETAIL. "
        "Never make assumptions about what might be happening outside the frame or what people might be thinking. "
        "If something is not clearly visible (like a person's gaze direction or what they're holding), explicitly "
        "state that it is not clearly visible rather than guessing. Be thorough and specific in all descriptions. "
        "Never hallucinate or make up details about hand positions or objects being held if they are not clearly visible."
    )

    user_text = (
        "# Laboratory Image Analysis: STRICTLY OBSERVABLE FACTS ONLY\n\n"
        "### 1. **Grounding Process**\n"
        "Identify each individual in the image using a two-step process:\n\n"
        "**STEP 1: First match the EXACT IDs listed below**\n"
        "The following individuals have already been detected with AprilTags. YOU MUST use these EXACT IDs:\n"
        f"```\n{id_positions}\n```\n\n"
        "**STEP 2: Only after matching all the above IDs**\n"
        "If you see additional people without AprilTags, label them as \"Untagged Person 1\", \"Untagged Person 2\", etc.\n"
        "IMPORTANT: Only use \"Untagged Person\" labels for individuals who DO NOT match any of the IDs listed above.\n\n"
        "**Coordinate System Definition**:\n"
        "  - The bottom-left corner of the image is **(0, 0)**.\n"
        "  - The top-right corner of the image is **(1, 1)**.\n"
        "  - `'Left'` and `'Right'` are determined by comparing the **x-coordinates** of the positions:\n"
        "    - Lower x means `'Left'`.\n"
        "    - Higher x means `'Right'`.\n"
        "  - `'Top'` and `'Bottom'` are determined by comparing the **y-coordinates** of the positions:\n"
        "    - Lower y means `'closer to Bottom'`.\n"
        "    - Higher y means `'closer to Top'`.\n\n"
        "---\n\n"
        "### 2. **Objective Observation Only**\n"
        "For each individual, describe ONLY what you can DIRECTLY observe in EXTENSIVE DETAIL:\n\n"
        "- **Position**: Precise location in the image (e.g., Left side, Center-right, Upper-left quadrant).\n"
        "- **Posture**: Detailed body position (e.g., seated with back straight, standing with weight on left leg, leaning forward).\n"
        "- **Gaze Direction**: ONLY if clearly visible in the frame. If their face is not visible or gaze direction is ambiguous, state \"gaze direction not clearly visible\".\n"
        "- **Hands & Objects**: ONLY describe hand positions that are CLEARLY VISIBLE. Common accurate descriptions include:\n"
        "  - \"Hands resting on table, not holding any objects\"\n"
        "  - \"Hands clasped/interlaced together in lap\"\n"
        "  - \"Arms crossed across chest, hands not interacting with any objects\"\n"
        "  - \"Right hand visible resting on armrest, left hand not visible\"\n"
        "  - \"Both hands partially visible but exact positions and any objects being held cannot be determined\"\n"
        "  - \"Hands positioned on keyboard, actively typing\"\n"
        "  - \"Hands at rest, palms down on table surface\"\n"
        "  - DO NOT assume hands are holding objects unless clearly visible in the image\n"
        "- **Interactions**: ONLY interactions that are completely contained within the frame. Be specific about the nature of any interactions.\n"
        "- **Clothing/Appearance**: Note relevant details about clothing or distinctive appearance features if visible.\n\n"
        "### 3. **Strict Rules for Descriptions**\n"
        "- PROVIDE EXHAUSTIVE DETAIL in your descriptions - avoid brevity\n"
        "- Do NOT infer what someone might be looking at outside the frame\n"
        "- Do NOT assume conversations are happening if you cannot see clear evidence\n"
        "- Do NOT interpret facial expressions unless they are clearly visible\n"
        "- Do NOT make assumptions about intent or thoughts\n"
        "- Do NOT guess at activities that might be happening outside the frame\n"
        "- NEVER assume a hand is holding an object unless you can clearly see both the hand and the object\n"
        "- Do NOT try to guess what specific type of object someone is holding if it's not clearly visible\n"
        "- Use phrases like \"appears to be\" for any description that isn't 100% certain\n"
        "- When in doubt, note uncertainty rather than guessing\n\n"
        "### Response Format\n"
        "Return your observations as a JSON object with the following structure (the IDs below are just EXAMPLES, use the actual IDs from the detected list and \"Untagged Person X\" for others):\n\n"
        "```json\n"
        "{\n"
        "  \"observations\": {\n"
        "    \"12\": {\n"
        "      \"position\": \"Center-right of the frame, approximately at coordinates (0.7, 0.5)\",\n"
        "      \"posture\": \"Seated at desk with back slightly hunched forward, upper body fully visible with shoulders relaxed\",\n"
        "      \"gaze\": \"Eyes clearly directed downward at visible laptop screen positioned on desk\",\n"
        "      \"hands_objects\": \"Both hands visible on keyboard, fingers positioned over keys. No objects being held\",\n"
        "      \"interactions\": \"Actively typing on keyboard, no interaction with other people visible in frame\",\n"
        "      \"clothing\": \"Wearing a dark blue long-sleeve shirt with collar visible\"\n"
        "    },\n"
        "    \"43\": {\n"
        "      \"position\": \"Left side of frame near wall\",\n"
        "      \"posture\": \"Standing with weight distributed evenly, facing toward center of room\",\n"
        "      \"gaze\": \"Gaze direction not clearly visible due to distance from camera and lighting conditions\",\n"
        "      \"hands_objects\": \"Hands positioned at sides, not holding any visible objects. Palms appear to be facing inward\",\n"
        "      \"interactions\": \"No clear interactions visible within the frame\",\n"
        "      \"clothing\": \"Wearing what appears to be a light-colored t-shirt and dark pants\"\n"
        "    },\n"
        "    \"Untagged Person 1\": {\n"
        "      \"position\": \"Bottom-right corner of frame, partially visible\",\n"
        "      \"posture\": \"Seated at what appears to be a desk or table, only upper torso visible in frame\",\n"
        "      \"gaze\": \"Appears to be looking at visible computer monitor directly in front of them, eyes clearly focused on screen\",\n"
        "      \"hands_objects\": \"Hands visible and resting in lap, interlaced with fingers clasped together. No objects being held\",\n"
        "      \"interactions\": \"No interaction with other individuals visible\",\n"
        "      \"clothing\": \"Wearing a red and white patterned shirt, possibly plaid or checkered\"\n"
        "    }\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "If something is not visible or unclear, explicitly state this fact within the appropriate field. NEVER use short or minimal descriptions - provide comprehensive details for every field. DO NOT hallucinate hand positions or objects being manipulated."
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
    """Generate a prompt message for LLM, asking the LLM to categorize actions based strictly on the provided description."""
    system_content = (
        "You are a precise action classifier who only bases classifications on explicitly stated observations. "
        "Never infer information beyond what is directly stated in the description. "
        "When faced with ambiguity or insufficient information, classify as 'Unclear' rather than making assumptions."
    )

    user_text = (
        "# Action Classification: STRICTLY BASED ON PROVIDED DESCRIPTION\n\n"
        "### 1. **Review the Observation Data**\n"
        "Below is a factual description of individuals in a laboratory setting:\n\n"
        f"```\n{image_description}\n```\n\n"
        "### 2. **Action Categories**\n"
        "Classify each person's action into one of these defined categories based ONLY on what is explicitly described:\n"
        f"```\n{action_definitions}\n```\n\n"
        "### 3. **Classification Rules**\n"
        "- ONLY use information explicitly stated in the description\n"
        "- If the description says something is \"not visible\" or \"unclear,\" do NOT use it for classification\n"
        "- If the described behavior doesn't clearly fit a category, classify as 'Unclear'\n"
        "- Do NOT make assumptions about what might be happening\n"
        "- Require clear evidence in the description to assign a non-'Unclear' category\n"
        "- Any statement indicating uncertainty (e.g., \"appears to be\") should be treated with caution\n\n"
        "### Response Format\n"
        "Return a JSON object with ID-to-action mappings and justifications based strictly on the provided description (the IDs below are just EXAMPLES, use the actual IDs from the observations):\n"
        "```json\n"
        "{\n"
        "  \"classifications\": {\n"
        "    \"12\": \"Working-Software\",\n"
        "    \"43\": \"Unclear\",\n"
        "    \"Untagged Person 1\": \"Observing\"\n"
        "  },\n"
        "  \"justifications\": {\n"
        "    \"12\": \"Classified as 'Working-Software' because description explicitly states they are typing on keyboard while looking at computer screen\",\n"
        "    \"43\": \"Classified as 'Unclear' because description states their gaze direction is not clearly visible and hands are partially visible\",\n"
        "    \"Untagged Person 1\": \"Classified as 'Observing' because description states they are looking at a visible screen without interacting with it\"\n"
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
