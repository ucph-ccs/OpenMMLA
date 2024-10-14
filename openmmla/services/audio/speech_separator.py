import base64
import gc

import torch
from flask import request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from openmmla.services.server import Server
from openmmla.utils.audio.processing import write_frames_to_wav


class SpeechSeparator(Server):
    """Speech separator separates the audio signals from a mixed signal. It receives audio signal from base station,
    and sends back two separated signals. It can help to differentiate a speaker's voice from an overlapped speaking
    environment."""

    def __init__(self, project_dir, config_path, use_cuda=True):
        """Initialize the speech separator.

        Args:
            project_dir: the project directory.
            use_cuda: whether to use CUDA or not.
        """
        super().__init__(project_dir=project_dir, config_path=config_path, use_cuda=use_cuda)
        self.cuda_enable = use_cuda and torch.cuda.is_available()
        self.device = 'gpu' if self.cuda_enable else 'cpu'

        sp_model = self.config['SpeechSeparator']['model']
        sp_model_local = self.config['SpeechSeparator']['model_local']

        # Initialize speech separation model
        try:
            self.separation_model = pipeline(Tasks.speech_separation, device=self.device, model=sp_model)
        except ValueError:
            self.separation_model = pipeline(Tasks.speech_separation, device=self.device, model=sp_model_local)

    def process_request(self):
        """Resample the audio.

        Returns:
            A tuple containing the response (separated bytes streams) and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                audio_file = request.files['audio']
                audio_file_path = self._get_temp_file_path('separate_audio', base_id, 'wav')
                write_frames_to_wav(audio_file_path, audio_file.read(), 1, 2, 8000)

                result = self.separation_model(audio_file_path)
                processed_bytes_streams = []

                for signal in result['output_pcm_list']:
                    encoded_bytes_stream = base64.b64encode(signal).decode('utf-8')
                    processed_bytes_streams.append(encoded_bytes_stream)

                # Return the processed audio data
                return jsonify({"processed_bytes_streams": processed_bytes_streams}), 200
            except Exception as e:
                self.logger.error(f"during speech separation, {e} happens.")
                return jsonify({"error": str(e)}), 500
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400
