import json
import os
from typing import Any

from openmmla.utils.requests import send_request_with_retry


def request_multi_angle_frame_analyze(image_paths: list[str], angles: list[str], angle_descriptions: list[str],
                                      session_id: str, url: str,
                                      participant_descriptions: dict | None = None,
                                      timeout: int = 300) -> dict[str, Any] | None:
    """Request multi-angle frame analysis from the server.
    
    Args:
        image_paths: List of paths to images from different angles
        angles: List of angle names corresponding to each image
        angle_descriptions: List of angle descriptions corresponding to each image
        session_id: Session ID for identification
        url: URL of the frame analyzer service
        participant_descriptions: Dictionary of participant descriptions for this session (optional)
        timeout: Request timeout in seconds
        
    Returns:
        dict: Analysis results, or None if request fails
    """

    def process_response(response):
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode response JSON: {e}")

    # Validate input parameters
    if len(image_paths) != len(angles) or len(angles) != len(angle_descriptions):
        raise ValueError("image_paths, angles, and angle_descriptions must have the same length")
    
    # Check if all image files exist
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

    data = {}
    data['session_id'] = session_id

    # Add participant descriptions if provided
    if participant_descriptions:
        data['participant_descriptions'] = json.dumps(participant_descriptions)
        
    # Add angles and angle descriptions as JSON
    data['angles'] = json.dumps(angles)
    data['angle_descriptions'] = json.dumps(angle_descriptions)

    files = []

    # Create multipart form data with images
    for image_path in image_paths:
        with open(image_path, 'rb') as image_file:
            file_content = image_file.read()
            files.append(
                ('images', (os.path.basename(image_path), file_content, 'image/jpeg'))
            )

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)
