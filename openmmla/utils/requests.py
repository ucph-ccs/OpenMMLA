import json
import time
from typing import Union, List, Tuple

import numpy as np
import requests
import soundfile as sf

from openmmla.utils.logger import get_logger

logger = get_logger(__name__)


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


def request_speech_enhancement(audio_path: str, base_id: str, audio_server_host: str) -> str:
    url = f'http://{audio_server_host}:8080/enhance'

    # url = f'http://{audio_server_host}:5003/enhance'

    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate)}

    return send_request_with_retry(url, files, data, process_response=process_response)


def request_audio_inference(audio_path: str, base_id: str, audio_server_host: str) -> np.ndarray:
    url = f'http://{audio_server_host}:8080/infer'

    # url = f'http://{audio_server_host}:5002/infer'

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


def request_voice_activity_detection(audio_path: str, base_id: str, inplace: int, audio_server_host: str) -> Union[
    str, None]:
    url = f'http://{audio_server_host}:8080/vad'

    # url = f'http://{audio_server_host}:5004/vad'

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


def request_speech_separation(audio_file_path: str, base_id: str, audio_server_host: str) -> List[str]:
    url = f'http://{audio_server_host}:8080/separate'

    # url = f'http://{audio_server_host}:5001/separate'

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
        audio_server_host: str
) -> str:
    url = f'http://{audio_server_host}:8080/transcribe'

    # url = f'http://{audio_server_host}:5000/transcribe'

    def process_response(response):
        response_json = response.json()
        transcription_text = response_json.get('text')
        return transcription_text

    fr = '8000' if sp else '16000'
    files = {'audio': ('audio.wav', frames, 'audio/wav')}
    data = {'base_id': base_id, 'fr': fr}

    return send_request_with_retry(url, files, data, timeout=15, process_response=process_response)


def request_resampling(audio_path: str, base_id: str, target_fr: int, audio_server_host) -> str:
    url = f'http://{audio_server_host}:8080/resample'

    # url = f'http://{audio_server_host}:5005/resample'

    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    audio_data, sample_rate = sf.read(audio_path, dtype='int16')
    audio_bytes = audio_data.tobytes()

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate), 'target_fr': str(target_fr)}

    return send_request_with_retry(url, files, data, process_response=process_response)
