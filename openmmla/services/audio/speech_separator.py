import base64
import configparser
import gc
import os

import torch
from flask import request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from openmmla.utils.audio.processing import write_frames_to_wav
from openmmla.utils.logger import get_logger


class SpeechSeparator:
    """Speech separator separates the audio signals from a mixed signal. It receives audio signal from base station,
    and sends back two separated signals. It can help to differentiate a speaker's voice from an overlapped speaking
    environment."""

    def __init__(self, project_dir, config_path, use_cuda=True):
        """Initialize the speech separator.

        Args:
            project_dir: the project directory.
            use_cuda: whether to use CUDA or not.
        """
        # Check if the project directory exists
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project directory not found at {project_dir}")

        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(project_dir, config_path)

        # Check if the configuration file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)
        sp_model = config['SpeechSeparator']['model']
        sp_model_local = config['SpeechSeparator']['model_local']

        self.server_logger_dir = os.path.join(project_dir, 'logger')
        self.server_file_folder = os.path.join(project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_file_folder, exist_ok=True)

        self.logger = get_logger('modelscope', os.path.join(self.server_logger_dir, 'separate_server.log'))
        self.cuda_enable = use_cuda and torch.cuda.is_available()
        self.device = 'gpu' if self.cuda_enable else 'cpu'

        # Initialize speech separation model
        try:
            self.separation_model = pipeline(Tasks.speech_separation, device=self.device, model=sp_model)
        except ValueError:
            self.separation_model = pipeline(Tasks.speech_separation, device=self.device, model=sp_model_local)

    def separate_speech(self):
        """Separate the speech from the audio.

        Returns:
            A tuple containing the JSON response and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                audio_file = request.files['audio']
                audio_file_path = os.path.join(self.server_file_folder, f'separate_audio_{base_id}.wav')
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
