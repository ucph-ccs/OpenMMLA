import gc
import json
import os

import nemo.collections.asr as nemo_asr
import torch
import yaml
from flask import request, jsonify

from openmmla.utils.audio.processing import write_frames_to_wav
from openmmla.utils.logger import get_logger


class AudioInferer:
    """Audio inferer generates the embeddings from audio signal in latent space. It receives audio signal from
    node base, and sends back the embeddings."""

    def __init__(self, project_dir, config_path, use_cuda=True):
        """Initialize the audio inferer.

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

        with open(self.config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.sr_model = config['AudioInferer']['model']

        self.server_logger_dir = os.path.join(project_dir, 'logger')
        self.server_file_folder = os.path.join(project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_file_folder, exist_ok=True)

        self.logger = get_logger('titanet', os.path.join(self.server_logger_dir, 'infer_server.log'))
        self.cuda_enable = use_cuda and torch.cuda.is_available()

        # Initialize titanet model
        if self.cuda_enable:
            self.infer_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.sr_model)
        else:
            self.infer_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.sr_model,
                                                                                       map_location='cpu')
        self.infer_model.eval()

    def infer(self):
        """Perform inference on the audio.

        Returns:
            A tuple containing the JSON response and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                fr = int(request.values.get('fr'))
                audio_file = request.files['audio']
                audio_file_path = os.path.join(self.server_file_folder, f'infer_audio_{base_id}.wav')
                write_frames_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                self.logger.info(f"starting inference for {base_id}...")
                feature = self.infer_model.get_embedding(audio_file_path)
                embeddings = json.dumps(feature.cpu().numpy().tolist())
                self.logger.info(f"finished inference for {base_id}.")

                return jsonify({"embeddings": embeddings}), 200
            except Exception as e:
                self.logger.error(f"during inference, {e} happens.")
                return jsonify({"error": str(e)}), 500
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400
