import json
import os
from typing import Any

from openmmla.utils.requests import send_request_with_retry


def request_multi_angle_frame_analyze(image_paths: list[str], session_id: str, url: str,
                                      angles: list[str] | None = None) -> dict[str, Any] | None:
    """Request multi-angle frame analysis from the server.
    
    Args:
        image_paths: List of paths to images from different angles
        session_id: Session ID for identification
        url: URL of the frame analyzer service
        angles: List of angle labels corresponding to each image (optional)
        
    Returns:
        dict: Analysis results, or None if request fails
    """

    def process_response(response):
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode response JSON: {e}")

    # Check if all image files exist
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")

    data = {}
    data['session_id'] = session_id

    files = []

    # Create multipart form data with images and angles
    for path in image_paths:
        with open(path, 'rb') as image_file:
            file_content = image_file.read()
            files.append(
                ('images', (os.path.basename(path), file_content, 'image/jpeg'))
            )

    # Add angles if provided
    if angles:
        if len(angles) == len(image_paths):
            for angle in angles:
                data.setdefault('angles', []).append(angle)

    return send_request_with_retry(url, files, data, timeout=40, process_response=process_response)
