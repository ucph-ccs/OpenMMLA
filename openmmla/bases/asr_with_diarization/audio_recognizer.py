import os

import numpy as np
import yaml

from openmmla.services.audio import AudioInferer
from openmmla.utils.audio.files import segment_wav
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url


class AudioRecognizer:
    """The audio recognizer compares audio segments against the known speakers' embeddings in the speaker library and
    gives label. During recognition, it will update the speaker profile when new recognized speaker's audio has high
    similarity by incorporating the new features into the old one.
    """
    logger = get_logger('audio-recognizer')

    def __init__(self, config_path, audio_db, local=False, use_cuda=True, use_onnx=False):
        config = yaml.safe_load(open(config_path, 'r'))
        self.audio_db = audio_db
        self.local = local

        if self.local:
            self.audio_inferer = AudioInferer(config_path, use_cuda=use_cuda, use_onnx=use_onnx)
        else:
            # self.server_host = socket.gethostbyname(config['Server']['server_host'])
            self.audio_inferer_url = resolve_url(config['Server']['asr']['audio_inference'])
            print(f"Audio inferer URL: {self.audio_inferer_url}")

        self.registered_speaker_names = []
        self.registered_speaker_features = None
        self.speaker_feature_count = {}
        self._load_audio_db(self.audio_db)
        self.logger.info(f"Successfully initialize audio recognizer, with local set to {local}.")

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
                    feature = self.audio_inferer.infer(audio_path)[0]
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

    def register(self, path, user_name):
        user_dir = os.path.join(self.audio_db, user_name)
        segment_wav(input_file=path, output_dir=user_dir)
        audio_files = os.listdir(user_dir)
        features = []
        for audio_file in audio_files:
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(user_dir, audio_file)
                feature = self.audio_inferer.infer(audio_path)[0]
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
            feature = self.audio_inferer.infer(path)[0]
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
            feature = self.audio_inferer.infer(path)[0]
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

    def _update_features(self, speaker_name, new_feature):
        speaker_index = self.registered_speaker_names.index(speaker_name)
        new_feature_normalized = new_feature / np.linalg.norm(new_feature, ord=2)
        num_existing_features = self.speaker_feature_count[speaker_name]
        total_features = num_existing_features + 1
        current_avg_feature = self.registered_speaker_features[speaker_index] * num_existing_features
        new_avg_feature = (current_avg_feature + new_feature_normalized) / total_features
        self.registered_speaker_features[speaker_index] = new_avg_feature
        self.speaker_feature_count[speaker_name] = total_features
