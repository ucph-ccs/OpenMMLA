import gc
import json

import nemo.collections.asr as nemo_asr
import torch
from flask import request, jsonify

from openmmla.services.server import Server
from openmmla.utils.audio.processing import write_frames_to_wav


class AudioInferer(Server):
    """Audio inferer generates the embeddings from audio signal in latent space. It receives audio signal from
    node base, and sends back the embeddings."""

    def __init__(self, project_dir, config_path, use_cuda=True):
        """Initialize the audio inferer.

        Args:
            project_dir: the project directory.
            use_cuda: whether to use CUDA or not.
        """
        super().__init__(project_dir=project_dir, config_path=config_path, use_cuda=use_cuda)
        self.cuda_enable = use_cuda and torch.cuda.is_available()

        self.sr_model = self.config['AudioInferer']['model']

        # Initialize titanet model
        if self.cuda_enable:
            self.infer_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.sr_model)
        else:
            self.infer_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.sr_model,
                                                                                       map_location='cpu')
        self.infer_model.eval()

    def process_request(self):
        """Perform inference on the audio.

        Returns:
            A tuple containing the JSON response (audio embeddings) and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                fr = int(request.values.get('fr'))
                audio_file = request.files['audio']
                audio_file_path = self._get_temp_file_path('infer_audio', base_id, 'wav')
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
