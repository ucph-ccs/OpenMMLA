import gc
import io

import torch
from flask import request, jsonify, send_file
from silero_vad import load_silero_vad, read_audio, save_audio, get_speech_timestamps, collect_chunks

from openmmla.services.server import Server
from openmmla.utils.audio.io import write_bytes_to_wav


class VoiceActivityDetector(Server):
    """ Voice activity detector detects the speaker's voice activity. It receives audio signal from base station and
    sends back the detected voice signal."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the voice activity detector.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()

    def _setup_yaml(self):
        self.cuda = self.config['VoiceActivityDetector'].get('cuda', True)
        self.onnx = self.config['VoiceActivityDetector'].get('onnx', False)
        self.cuda = self.cuda and torch.cuda.is_available()

    def _setup_objects(self):
        self.vad_model = load_silero_vad(onnx=self.onnx)

    def process_request(self):
        """Perform voice activity detection.

        Returns:
            A tuple containing the response and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                fr = int(request.values.get('fr', 16000))
                inplace = int(request.values.get('inplace', 0))
                audio_file = request.files['audio']
                audio_file_path = self._get_temp_file_path('vad_audio', base_id, 'wav')
                write_bytes_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                self.logger.info(f"starting VAD for {base_id}...")
                result = self._apply_vad(audio_file_path, fr, inplace)
                self.logger.info(f"finished VAD for {base_id}.")

                if inplace and result:
                    with open(result, 'rb') as f:
                        audio_data = f.read()
                    return send_file(io.BytesIO(audio_data), mimetype="audio/wav"), 200
                else:
                    # For cases where inplace is False or speech timestamps are not detected
                    return jsonify({"result": result or "None"}), 200
            except Exception as e:
                self.logger.error(f"Exception during voice activity detection", exc_info=True)
                return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400

    def _apply_vad(self, input_path: str, sampling_rate: int, inplace: int) -> str | None:
        wav = read_audio(input_path, sampling_rate=sampling_rate)
        speech_timestamps = get_speech_timestamps(wav, self.vad_model, sampling_rate=sampling_rate)
        if not speech_timestamps:
            return None
        if inplace:
            save_audio(input_path, collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
        if self.cuda:
            torch.cuda.empty_cache()
        return input_path
