import json
from typing import Union, List, Tuple

import numpy as np
import soundfile as sf

from openmmla.utils.requests import send_request_with_retry


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
