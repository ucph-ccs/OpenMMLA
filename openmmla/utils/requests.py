import json
import socket
import time
from typing import Union, List, Tuple
from urllib.parse import urlparse, urlunparse

import numpy as np
import requests
import soundfile as sf

from openmmla.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_url(url: str) -> str:
    """Resolve hostname in URL to IP address.
    
    Args:
        url: URL with hostname (e.g., 'http://uber-server.local:8080/infer')
        
    Returns:
        URL with resolved IP (e.g., 'http://192.168.1.100:8080/infer')
    """
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return url

        ip = socket.gethostbyname(parsed.hostname)
        resolved = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        return urlunparse(resolved)
    except socket.gastrror as e:
        logger.warning(f"Could not resolve hostname in {url}: {e}")
        return url


def send_request_with_retry(url, files, data, max_retries=4, timeout=10, process_response=None):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200 and process_response:
                return process_response(response)
            else:
                logger.warning(
                    f"Error in sending/requesting {url} with status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout occurred: {e}, retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}, retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(2)
    logger.warning("Max retries exceeded.")
    return None


def request_speech_enhancement(audio_path: str, base_id: str, url: str) -> str:
    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate)}

    return send_request_with_retry(url, files, data, process_response=process_response)


def request_audio_inference(audio_path: str, base_id: str, url: str) -> np.ndarray:
    def process_response(response):
        response_json = response.json()
        embeddings_list = json.loads(response_json["embeddings"])
        embeddings = np.array(embeddings_list)
        return embeddings

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate)}

    return send_request_with_retry(url, files, data, process_response=process_response)


def request_voice_activity_detection(audio_path: str, base_id: str, inplace: int, url: str) -> Union[str, None]:
    def process_response(response):
        if response.headers['Content-Type'] == 'audio/wav':
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            return audio_path
        else:
            result = response.json()
            return audio_path if result.get('result') != "None" else None

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate), 'inplace': str(inplace)}

    return send_request_with_retry(url, files, data, process_response=process_response)


def request_speech_separation(audio_file_path: str, base_id: str, url: str) -> List[str]:
    def process_response(response):
        response_json = response.json()
        processed_bytes_streams = response_json.get("processed_bytes_streams")
        return processed_bytes_streams

    audio_data, sample_rate = sf.read(audio_file_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id}

    return send_request_with_retry(url, files, data, process_response=process_response)


def request_speech_transcription(
        frames: Union[bytes, List[bytes], Tuple[bytes]],
        base_id: str,
        sp: bool,
        url: str
) -> str:
    def process_response(response):
        response_json = response.json()
        transcription_text = response_json.get('text')
        return transcription_text

    fr = '8000' if sp else '16000'
    files = {'audio': ('audio.wav', frames, 'audio/wav')}
    data = {'base_id': base_id, 'fr': fr}

    return send_request_with_retry(url, files, data, timeout=15, process_response=process_response)


def request_audio_resampling(audio_path: str, base_id: str, target_fr: int, url: str) -> str:
    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate), 'target_fr': str(target_fr)}

    return send_request_with_retry(url, files, data, process_response=process_response)
