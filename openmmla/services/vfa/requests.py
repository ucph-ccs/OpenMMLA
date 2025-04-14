import json
import os

from openmmla.utils.requests import send_request_with_retry


def request_frame_analyze(image_path: str, base_id: str, url: str) -> dict:
    def process_response(response):
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to decode response JSON: {e}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        data = {'base_id': base_id}

        return send_request_with_retry(url, files, data, timeout=20, process_response=process_response)
