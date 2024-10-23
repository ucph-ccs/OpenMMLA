import copy
import os
import socket

import librosa
import numpy as np
import yaml

from openmmla.utils.audio.processing import segment_wav
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import request_audio_inference

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


class AudioRecognizer:
    """The audio recognizer compares audio segments against the known speakers' embeddings in the speaker library and
    gives label. During recognition, it will update the speaker profile when new recognized speaker's audio has high
    similarity by incorporating the new features into the old one.
    """
    logger = get_logger('audio-recognizer')

    def __init__(self, config_path, audio_db, local=False, model_path=None, onnx_model_path=None, use_onnx=False,
                 use_cuda=True):
        config = yaml.safe_load(open(config_path, 'r'))
        self.audio_db = audio_db
        self.local = local

        if self.local and model_path:
            self.model_path = model_path
            self.onnx_model_path = onnx_model_path
            self.cuda_enable = use_cuda and torch is not None and torch.cuda.is_available()
            self.use_onnx = use_onnx
            self.model = None
            self.data_layer = None
            self.data_loader = None
            self.ort_session = None
            self._load_model()
        else:
            self.audio_server_host = socket.gethostbyname(config['Server']['audio_server_host'])

        self.registered_speaker_names = []
        self.registered_speaker_features = None
        self.speaker_feature_count = {}
        self._load_audio_db(self.audio_db)
        self.logger.info(f"Successfully initialize audio recognizer, with local set to {local}.")

    def register(self, path, user_name):
        user_dir = os.path.join(self.audio_db, user_name)
        segment_wav(input_file=path, output_dir=user_dir)
        audio_files = os.listdir(user_dir)
        features = []
        for audio_file in audio_files:
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(user_dir, audio_file)
                feature = self._infer(audio_path)[0]
                features.append(feature)

        features = np.array(features)
        features = features / (np.linalg.norm(features, ord=2, axis=-1, keepdims=True))
        reference_emb = np.sum(features, axis=0) / len(features)

        self.registered_speaker_names.append(user_name)
        self.speaker_feature_count[user_name] = len(features)
        if self.registered_speaker_features is None:
            self.registered_speaker_features = reference_emb[np.newaxis, :]
        else:
            self.registered_speaker_features = np.vstack([self.registered_speaker_features, reference_emb])

    def recognize(self, path, update_threshold=0.6):
        try:
            feature = self._infer(path)[0]
        except TypeError as e:  # In case the feature is None
            self.logger.warning(f'{e} happens when inferring, discard this segment')
            return '', -1
        feature = feature / np.linalg.norm(feature, ord=2)
        scores = np.dot(self.registered_speaker_features, feature)
        max_similarity_name = self.registered_speaker_names[np.argmax(scores)]
        max_similarity = np.max(scores)
        if max_similarity > update_threshold:
            self._update_features(max_similarity_name, feature)
        return max_similarity_name, max_similarity

    def recognize_among_candidates(self, path, candidates, origin_label, keep_threshold=0.1):
        candidates = [candidate for candidate in candidates if candidate not in ['unknown', 'silent']]
        if not candidates:
            return origin_label, 0  # Silent -> Unknown or Unknown -> Silent
        if not set(candidates).issubset(set(self.registered_speaker_names)):
            raise ValueError("Some candidates are not registered in the database.")
        try:
            feature = self._infer(path)[0]
        except TypeError as e:
            self.logger.warning(f'{e} happens when inferring, discard this segment')
            return '', -1
        feature = feature / np.linalg.norm(feature, ord=2)
        # Filter features of only the candidates
        candidate_indices = [self.registered_speaker_names.index(candidate) for candidate in candidates]
        candidate_features = self.registered_speaker_features[candidate_indices]
        # Calculate similarity scores with the candidates
        scores = np.dot(candidate_features, feature)
        max_index = np.argmax(scores)
        max_similarity = scores[max_index]
        if origin_label in ['unknown', 'silent']:  # Keep unknown and silent if similarity smaller than keep_threshold
            max_similarity_name = candidates[max_index] if max_similarity > keep_threshold else origin_label
        else:
            max_similarity_name = candidates[max_index]
        return max_similarity_name, max_similarity

    def reset_db(self, audio_db):
        self.audio_db = audio_db
        self._load_audio_db(self.audio_db)
        self.logger.info("Successfully reload audio database.")

    def _load_model(self):
        if self.use_onnx:
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_path)
            # Preserve a copy of the full config
            cfg = copy.deepcopy(self.model._cfg)
            self.model.preprocessor = self.model.from_config_dict(cfg.preprocessor)
            self.model.eval()
            self.model = self.model.to(self.model.device)
            self.data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
            self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=self.data_layer.collate_fn)
            if not os.path.exists(self.onnx_model_path):
                self.model.export(self.onnx_model_path)
            if self.cuda_enable:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path,
                                                                providers=['CUDAExecutionProvider'])
            else:
                self.ort_session = onnxruntime.InferenceSession(self.onnx_model_path,
                                                                providers=['CPUExecutionProvider'])
            self.logger.info("ONNX is enabled.")
        else:
            if self.cuda_enable:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_path)
            else:
                self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_path,
                                                                                     map_location='cpu')
            self.model.eval()
            self.logger.info("ONNX is disabled.")

    def _load_audio_db(self, audio_db_path):
        # Clean the speaker profiles
        self.registered_speaker_names = []
        self.registered_speaker_features = None
        if not os.path.exists(audio_db_path):
            os.mkdir(audio_db_path)
            return

        for person_dir in os.listdir(audio_db_path):
            if person_dir.endswith('.DS_Store'):
                continue

            person_path = os.path.join(audio_db_path, person_dir)
            person_features = []
            for audio in os.listdir(person_path):
                if audio.endswith('.DS_Store'):
                    continue
                audio_path = os.path.join(person_path, audio)
                try:
                    feature = self._infer(audio_path)[0]
                except TypeError as e:  # In case the feature is None
                    self.logger.warning(f'{e} happens when inferring, return')
                    return
                person_features.append(feature)

            person_features = np.array(person_features)
            person_features = person_features / (np.linalg.norm(person_features, ord=2, axis=-1, keepdims=True))
            reference_emb = np.sum(person_features, axis=0) / len(person_features)

            self.registered_speaker_names.append(person_dir)
            self.speaker_feature_count[person_dir] = len(person_features)
            if self.registered_speaker_features is None:
                self.registered_speaker_features = reference_emb[np.newaxis, :]
            else:
                self.registered_speaker_features = np.vstack([self.registered_speaker_features, reference_emb])
            self.logger.info("Loaded %s audio." % person_dir)

    def _infer(self, audio_path):
        if self.local:
            if self.use_onnx:
                audio, sample_rate = librosa.load(audio_path, sr=16000)
                feature, _ = self._infer_signal_onnx(audio)
            else:
                feature = self.model.get_embedding(audio_path).cpu().numpy()
            if self.cuda_enable:
                torch.cuda.empty_cache()
            return feature
        else:
            return request_audio_inference(audio_path, self.audio_db.split('/')[-1], self.audio_server_host)

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

    def _update_features(self, speaker_name, new_feature):
        speaker_index = self.registered_speaker_names.index(speaker_name)
        new_feature_normalized = new_feature / np.linalg.norm(new_feature, ord=2)
        num_existing_features = self.speaker_feature_count[speaker_name]
        total_features = num_existing_features + 1
        current_avg_feature = self.registered_speaker_features[speaker_index] * num_existing_features
        new_avg_feature = (current_avg_feature + new_feature_normalized) / total_features
        self.registered_speaker_features[speaker_index] = new_avg_feature
        self.speaker_feature_count[speaker_name] = total_features

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
