import copy
import gc
import json
import os

import librosa
import numpy as np
import torch
from flask import request, jsonify

from openmmla.services.server import Server
from openmmla.utils.audio.io import write_bytes_to_wav

try:
    import nemo.collections.asr as nemo_asr
    from nemo.core.classes import IterableDataset
    from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
    from torch.utils.data import DataLoader
    _NEMO_AVAILABLE = True
except ImportError:
    _NEMO_AVAILABLE = False

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    import wespeaker as _wespeaker
    _WESPEAKER_AVAILABLE = True
    _WESPEAKER_IMPORT_ERROR = None
except ImportError as exc:
    _WESPEAKER_AVAILABLE = False
    _WESPEAKER_IMPORT_ERROR = exc


if _NEMO_AVAILABLE:
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

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the audio inferer.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()

    def _setup_yaml(self):
        config = self.config['AudioInferer']
        self.cuda = config.get('cuda', True)
        self.cuda = self.cuda and torch.cuda.is_available()
        self.backend = config.get('backend', 'nemo')

        if self.backend == 'nemo':
            nemo_config = config.get('nemo', {})
            self.onnx = nemo_config.get('onnx', config.get('onnx', False))
            self.model_name = nemo_config.get('model', config.get('model', ''))
            self.onnx_model_name = nemo_config.get('model_onnx', config.get('model_onnx', ''))
        elif self.backend == 'wespeaker':
            ws_config = config.get('wespeaker', {})
            self.model_name = ws_config.get('model', config.get('model', 'w2vbert2_mfa'))
        else:
            raise ValueError(f"Unsupported AudioInferer backend: '{self.backend}'. Use 'nemo' or 'wespeaker'.")

        self.logger.info(f"AudioInferer backend: {self.backend}, model: {self.model_name}, cuda: {self.cuda}")

    def _setup_objects(self):
        if self.backend == 'nemo':
            self._setup_nemo()
        elif self.backend == 'wespeaker':
            self._setup_wespeaker()

    def _setup_nemo(self):
        if not _NEMO_AVAILABLE:
            raise ImportError("NeMo is not installed. Install it with 'pip install nemo_toolkit[asr]'")

        if self.onnx:
            if not os.path.exists(self.onnx_model_name):
                self.onnx_model_name = os.path.join(self.project_dir, f'{self.model_name}.onnx')
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
                cfg = copy.deepcopy(self.model._cfg)
                self.model.preprocessor = self.model.from_config_dict(cfg.preprocessor)
                self.model.eval()
                self.model = self.model.to(self.model.device)
                self.data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
                self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)
                self.model.export(self.onnx_model_name)

            if self.cuda:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_name,
                                                                providers=['CUDAExecutionProvider'])
            else:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_name,
                                                                providers=['CPUExecutionProvider'])
            self.logger.info("NeMo backend initialized with ONNX runtime")
        else:
            if self.cuda:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
            else:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name,
                                                                                     map_location='cpu')
            self.model.eval()
            self.logger.info("NeMo backend initialized")

    def _setup_wespeaker(self):
        if not _WESPEAKER_AVAILABLE:
            raise ImportError(
                "WeSpeaker backend is unavailable. Install the asr-server-wespeaker "
                f"extra dependencies. Original import error: {_WESPEAKER_IMPORT_ERROR}"
            )

        device = "cuda:0" if self.cuda else "cpu"
        self.logger.info(f"Loading WeSpeaker model: {self.model_name}...")
        self.ws_model = _wespeaker.load_model(self.model_name)
        self.ws_model.set_device(device)
        self.logger.info(f"WeSpeaker backend initialized (model={self.model_name}, device={device})")

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
                write_bytes_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                self.logger.info(f"starting inference for {base_id}...")
                feature = self._infer(audio_file_path)
                embeddings = json.dumps(feature.tolist())
                self.logger.info(f"finished inference for {base_id}.")

                return jsonify({"embeddings": embeddings}), 200
            except Exception as e:
                self.logger.error(f"Exception during audio inference", exc_info=True)
                return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400

    def _infer(self, audio_path):
        if self.backend == 'wespeaker':
            feature = self._infer_wespeaker(audio_path)
        elif self.onnx:
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            feature, _ = self._infer_signal_onnx(audio)
        else:
            feature = self.model.get_embedding(audio_path).cpu().numpy()
        if self.cuda:
            torch.cuda.empty_cache()
        return feature

    def _infer_wespeaker(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)
        device = "cuda:0" if self.cuda else "cpu"
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        embedding = self.ws_model.extract_embedding_from_pcm(audio_tensor, 16000)
        if embedding is None:
            raise RuntimeError("WeSpeaker failed to extract embedding")
        if embedding.is_cuda:
            embedding = embedding.cpu()
        return embedding.detach().numpy().squeeze()

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
