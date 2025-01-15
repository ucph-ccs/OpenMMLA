import copy
import gc
import json
import os

import librosa
import numpy as np
from flask import request, jsonify

from openmmla.services.server import Server
from openmmla.utils.audio.io import write_frames_to_wav

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    import torch
except ImportError:
    torch = None

try:
    from nemo.core.classes import IterableDataset
except ImportError:
    IterableDataset = None

try:
    from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
except ImportError:
    NeuralType, AudioSignal, LengthsType = None, None, None

try:
    from torch.utils.data import DataLoader
except ImportError:
    DataLoader = None

if NeuralType:
    class AudioDataLayer(IterableDataset):
        @property
        def output_types(self):
            return {
                'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
                'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            }

        def __init__(self, sample_rate):
            super().__init__()
            self._sample_rate = sample_rate
            self.output = True

        def __iter__(self):
            return self

        def __next__(self):
            if not self.output:
                raise StopIteration
            self.output = False
            return torch.as_tensor(self.signal, dtype=torch.float32), \
                torch.as_tensor(self.signal_shape, dtype=torch.int64)

        def set_signal(self, signal):
            self.signal = signal.astype(np.float32) / 32768.
            self.signal_shape = self.signal.size
            self.output = True

        def __len__(self):
            return 1


class AudioInferer(Server):
    """Audio inferer generates the embeddings from audio signal in latent space. It receives audio signal from
    node base, and sends back the embeddings."""

    def __init__(self, project_dir: str, config_path: str, use_cuda: bool = True, use_onnx: bool = False):
        """Initialize the audio inferer.

        Args:
            project_dir (str): the project directory.
            config_path (str): path to the configuration file.
            use_cuda (bool): whether to use CUDA or not.
            use_onnx (bool): whether to use ONNX or not.
        """
        super().__init__(project_dir=project_dir, config_path=config_path, use_cuda=use_cuda, use_onnx=use_onnx)
        self.cuda_enable = use_cuda and torch is not None and torch.cuda.is_available()

        self.model_name = self.config['AudioInferer']['model']

        if self.use_onnx:
            self.onnx_model_name = self.config['AudioInferer']['model_onnx']

        self._load_model()

    def _load_model(self):
        if self.use_onnx:
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
            cfg = copy.deepcopy(self.model._cfg)  # Preserve a copy of the full config
            self.model.preprocessor = self.model.from_config_dict(cfg.preprocessor)
            self.model.eval()
            self.model = self.model.to(self.model.device)
            self.data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
            self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)
            if not os.path.exists(self.onnx_model_name):
                self.model.export(self.onnx_model_name)
            if self.cuda_enable:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_name,
                                                                providers=['CUDAExecutionProvider'])
            else:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_name,
                                                                providers=['CPUExecutionProvider'])
            self.logger.info("ONNX is enabled.")
        else:
            if self.cuda_enable:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
            else:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name,
                                                                                     map_location='cpu')
            self.model.eval()
            self.logger.info("ONNX is disabled.")

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
                feature = self._infer(audio_file_path)
                embeddings = json.dumps(feature.tolist())
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

    def _infer(self, audio_path):
        if self.use_onnx:
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            feature, _ = self._infer_signal_onnx(audio)
        else:
            feature = self.model.get_embedding(audio_path).cpu().numpy()
        if self.cuda_enable:
            torch.cuda.empty_cache()
        return feature

    def _infer_signal_onnx(self, signal):
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(self.model.device), audio_signal_len.to(self.model.device)
        processed_signal, processed_signal_len = self.model.preprocessor(
            input_signal=audio_signal, length=audio_signal_len,
        )
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(processed_signal),
                      self.ort_session.get_inputs()[1].name: self.to_numpy(processed_signal_len)}
        logits, emb = self.ort_session.run(None, ort_inputs)
        return emb, logits

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
