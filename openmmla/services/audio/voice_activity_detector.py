import gc
import io
import os
from typing import Union

import torch
from flask import request, jsonify, send_file

from openmmla.utils.audio.processing import write_frames_to_wav
from openmmla.utils.logger import get_logger


class VoiceActivityDetector:
    """ Voice activity detector detects the speaker's voice activity. It receives audio signal from base station and
    sends back the detected voice signal."""

    def __init__(self, project_dir, use_cuda=True, use_onnx=True):
        """Initialize the voice activity detector.

        Args:
            project_dir: the project directory.
            use_cuda: whether to use CUDA or not.
            use_onnx: whether to use ONNX or not.
        """
        self.server_logger_dir = os.path.join(project_dir, 'logger')
        self.server_file_folder = os.path.join(project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_file_folder, exist_ok=True)

        self.logger = get_logger('silero', os.path.join(self.server_logger_dir, 'vad_server.log'))

        self.use_cuda = use_cuda
        self.use_onnx = use_onnx
        self.cuda_enable = use_cuda and torch.cuda.is_available()

        # Initialize VAD
        torch.set_num_threads(1)
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=use_onnx)
        except Exception:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../silero-vad'),
                model='silero_vad', source='local', force_reload=True, onnx=use_onnx)

        (self.get_speech_timestamps, self.save_audio, self.read_audio,
         self.VADIterator, self.collect_chunks) = utils

    def apply_vad(self, input_path: str, sampling_rate: int, inplace: int) -> Union[str, None]:
        wav = self.read_audio(input_path, sampling_rate=sampling_rate)
        speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, sampling_rate=sampling_rate)
        if not speech_timestamps:
            return None
        if inplace:
            self.save_audio(input_path, self.collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
        if self.cuda_enable:
            torch.cuda.empty_cache()
        return input_path

    def vad(self):
        """Perform voice activity detection.

        Returns:
            A tuple containing the response and status code.
        """
        try:
            base_id = request.values.get('base_id')
            fr = int(request.values.get('fr', 16000))
            inplace = int(request.values.get('inplace', 0))
            audio_file = request.files['audio']
            audio_file_path = os.path.join(self.server_file_folder, f'vad_audio_{base_id}.wav')
            write_frames_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

            self.logger.info(f"starting VAD for {base_id}...")
            result = self.apply_vad(audio_file_path, fr, inplace)
            self.logger.info(f"finished VAD for {base_id}.")

            if inplace and result:
                with open(result, 'rb') as f:
                    audio_data = f.read()
                return send_file(io.BytesIO(audio_data), mimetype="audio/wav"), 200
            else:
                # For cases where inplace is False or speech timestamps are not detected
                return jsonify({"result": result or "None"}), 200

        except Exception as e:
            self.logger.error(f"Error processing VAD for {base_id}: {str(e)}")
            return jsonify({"error": str(e)}), 500
        finally:
            torch.cuda.empty_cache()
            gc.collect()
