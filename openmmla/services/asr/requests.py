import json
import wave

import numpy as np

from openmmla.utils.requests import send_request_with_retry


def request_speech_enhancement(audio_path: str, base_id: str, url: str, timeout: int = 10) -> str | None:
    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate)}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)


def request_audio_inference(audio_path: str, base_id: str, url: str, timeout: int = 10) -> np.ndarray | None:
    def process_response(response):
        response_json = response.json()
        embeddings_list = json.loads(response_json["embeddings"])
        embeddings = np.array(embeddings_list)
        return embeddings

    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate)}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)


def request_voice_activity_detection(audio_path: str, base_id: str, inplace: int, url: str, timeout: int = 10) -> str | None:
    def process_response(response):
        if response.headers['Content-Type'] == 'audio/wav':
            with open(audio_path, 'wb') as f:
                f.write(response.content)
            return audio_path
        else:
            result = response.json()
            return audio_path if result.get('result') != "None" else None

    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate), 'inplace': str(inplace)}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)


def request_speech_separation(audio_path: str, base_id: str, url: str, timeout: int = 10) -> list[str] | None:
    def process_response(response):
        response_json = response.json()
        processed_bytes_streams = response_json.get("processed_bytes_streams")
        return processed_bytes_streams

    with wave.open(audio_path, 'rb') as wav_file:
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)


def request_speech_transcription(
        frames: bytes | list[bytes] | tuple[bytes],
        frame_rate: int,
        base_id: str,
        url: str,
        timeout: int = 15
) -> dict | None:
    def process_response(response):
        response_dict = response.json() # {text: str, words: list[dict]}, words is optional
        return response_dict

    files = {'audio': ('audio.wav', frames, 'audio/wav')}
    data = {'base_id': base_id, 'fr': frame_rate}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)


def request_audio_resampling(audio_path: str, base_id: str, target_fr: int, url: str, timeout: int = 10) -> str | None:
    def process_response(response):
        with open(audio_path, 'wb') as out_file:
            out_file.write(response.content)
        return audio_path

    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    files = {'audio': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'base_id': base_id, 'fr': str(sample_rate), 'target_fr': str(target_fr)}

    return send_request_with_retry(url, files, data, timeout=timeout, process_response=process_response)
