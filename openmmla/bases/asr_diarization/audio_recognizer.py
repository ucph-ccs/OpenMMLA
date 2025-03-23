import os
import pickle
import tempfile
import time

import numpy as np
import yaml

from openmmla.services.asr_diarization.requests import request_audio_inference
from openmmla.utils.audio.files import segment_wav
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url


class AudioRecognizer:
    """The audio recognizer compares audio segments against the known speakers' embeddings in the speaker library and
    gives a label. During recognition, it will update the speaker profile when new recognized speaker's audio has high
    similarity by incorporating the new features into the old one.
    """
    logger = get_logger('audio-recognizer')

    def __init__(self, config_path: str, audio_db: str, keep_audio: bool = False):
        config = yaml.safe_load(open(config_path, 'r'))
        self.audio_db = audio_db
        self.audio_inferer_url = resolve_url(config['Server']['asr']['audio_inference'])
        self.keep_audio = keep_audio
        print(f"Audio inferer URL: {self.audio_inferer_url}")

        self.speaker_names = []  # List of speaker names
        self.speaker_features = None  # np.array of origin embeddings (each row is a speaker's embedding)
        self.speaker_adaptive_features = None  # np.array of updated embeddings (same shape/order as speaker_features)
        self.speaker_adaptive_features_counts = {}  # Dict of speaker names to the weight of updated embeddings

        self._load_audio_db(self.audio_db)
        self.logger.info("Successfully initialize audio recognizer.")

    def register(self, path: str, user_name: str):
        # Check if user already exists
        if user_name in self.speaker_names:
            self.logger.info(f"User '{user_name}' already exists. Adding new embeddings to existing profile.")
            user_index = self.speaker_names.index(user_name)
        else:
            self.logger.info(f"Registering new user '{user_name}'.")
            user_index = None

        # Create embeddings directory for user
        user_embeddings_dir = os.path.join(self.audio_db, user_name)
        os.makedirs(user_embeddings_dir, exist_ok=True)

        # Create a temporary directory to store segmented audio
        with tempfile.TemporaryDirectory() as temp_dir:
            self.logger.info(f"Created temporary directory for audio segmentation: {temp_dir}")

            # Segment audio into temporary directory
            segment_wav(input_file=path, output_dir=temp_dir)

            # Process each segmented audio file and extract embeddings
            audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
            self.logger.info(f"Found {len(audio_files)} audio segments.")

            features = []
            new_embedding_filenames = []

            for audio_file in audio_files:
                audio_path = os.path.join(temp_dir, audio_file)
                # Infer embedding from audio
                feature = self._infer(audio_path)
                features.append(feature)

                # Save embedding with timestamp to ensure unique filenames
                timestamp = int(time.time() * 1000)
                base_filename = os.path.splitext(audio_file)[0]
                embedding_filename = f"{base_filename}_{timestamp}.pkl"
                embedding_path = os.path.join(user_embeddings_dir, embedding_filename)
                new_embedding_filenames.append(embedding_filename)

                # Save the embedding
                with open(embedding_path, 'wb') as f:
                    pickle.dump(feature, f)

                # Optionally, keep a copy of the audio file
                if self.keep_audio:
                    audio_filename = f"{base_filename}_{timestamp}.wav"
                    audio_save_path = os.path.join(user_embeddings_dir, audio_filename)
                    with open(audio_path, 'rb') as src_file:
                        audio_data = src_file.read()
                    with open(audio_save_path, 'wb') as dst_file:
                        dst_file.write(audio_data)
                    self.logger.debug(f"Saved audio file: {audio_filename}")

            self.logger.info(
                f"Processed {len(features)} audio segments and saved embeddings{' and audio files' if self.keep_audio else ''}.")

        if not features:
            self.logger.warning(f"No features could be extracted for user '{user_name}'.")
            return

        # Normalize and compute the reference embedding
        features = np.array(features)
        features = features / (np.linalg.norm(features, ord=2, axis=-1, keepdims=True))
        num_features = len(features)
        reference_emb = np.sum(features, axis=0) / num_features

        if user_index is not None:
            # For an existing user, combine new features with existing ones
            existing_features = []
            existing_embeddings = [f for f in os.listdir(user_embeddings_dir) if f.endswith('.pkl')]
            existing_embeddings = [f for f in existing_embeddings if f not in new_embedding_filenames]
            for embedding_file in existing_embeddings:
                embedding_path = os.path.join(user_embeddings_dir, embedding_file)
                try:
                    with open(embedding_path, 'rb') as f:
                        feature = pickle.load(f)
                        existing_features.append(feature)
                except Exception as e:
                    self.logger.warning(f"Error loading embedding {embedding_path}: {e}")
            all_features = existing_features + list(features)
            num_features = len(all_features)
            all_features = np.array(all_features)
            all_features = all_features / (np.linalg.norm(all_features, ord=2, axis=-1, keepdims=True))
            reference_emb = np.sum(all_features, axis=0) / num_features

            # Update the existing origin embedding
            self.speaker_features[user_index] = reference_emb
            self.speaker_adaptive_features[user_index] = reference_emb
            self.speaker_adaptive_features_counts[user_name] = num_features
        else:
            # For a new user, append the reference embedding to both origin and updated arrays.
            self.speaker_names.append(user_name)
            self.speaker_adaptive_features_counts[user_name] = num_features
            if self.speaker_features is None:
                self.speaker_features = reference_emb[np.newaxis, :]
                self.speaker_adaptive_features = reference_emb[np.newaxis, :]
            else:
                self.speaker_features = np.vstack([self.speaker_features, reference_emb])
                self.speaker_adaptive_features = np.vstack([self.speaker_adaptive_features, reference_emb])

        self.logger.info(f"Successfully registered/updated user '{user_name}' with {num_features} embeddings.")

    def recognize(self, path: str, update_threshold: float = 0.6) -> tuple[str, float]:
        try:
            feature = self._infer(path)
            feature = feature / np.linalg.norm(feature, ord=2)

            # Compute similarity scores for both origin and updated embeddings in a vectorized way
            origin_scores = np.dot(self.speaker_features, feature)
            updated_scores = np.dot(self.speaker_adaptive_features, feature)
            scores = np.maximum(origin_scores, updated_scores)

            max_index = np.argmax(scores)
            max_similarity_name = self.speaker_names[max_index]
            max_similarity = scores[max_index]

            if max_similarity > update_threshold:
                self._update_features(max_similarity_name, feature)

            return max_similarity_name, max_similarity
        except TypeError as e:
            self.logger.warning(f'{e} happens when inferring, discard this segment')
            return '', -1

    def recognize_among_candidates(self, path: str, candidates: list[str], origin_label: str,
                                   keep_threshold: float = 0.1) -> tuple[str, float]:
        # Filter out non-meaningful candidates
        candidates = [candidate for candidate in candidates if candidate not in ['unknown', 'silent']]
        if not candidates:
            return origin_label, 0
        if not set(candidates).issubset(set(self.speaker_names)):
            raise ValueError("Some candidates are not registered in the database.")

        try:
            feature = self._infer(path)
            feature = feature / np.linalg.norm(feature, ord=2)

            indices = [self.speaker_names.index(candidate) for candidate in candidates]
            origin_scores = np.dot(self.speaker_features[indices], feature)
            updated_scores = np.dot(self.speaker_adaptive_features[indices], feature)
            candidate_similarities = np.maximum(origin_scores, updated_scores)

            max_index = np.argmax(candidate_similarities)
            max_similarity = candidate_similarities[max_index]

            if origin_label in ['unknown', 'silent']:
                max_similarity_name = candidates[max_index] if max_similarity > keep_threshold else origin_label
            else:
                max_similarity_name = candidates[max_index]

            return max_similarity_name, max_similarity
        except TypeError as e:
            self.logger.warning(f'{e} happens when inferring, discard this segment')
            return '', -1

    def reset_db(self, audio_db: str):
        self.audio_db = audio_db
        self._load_audio_db(self.audio_db)
        self.logger.info("Successfully reload audio database.")

    def _load_audio_db(self, audio_db_path: str):
        # Initialize/reinitialize speaker profiles
        self.speaker_names = []
        self.speaker_features = None
        self.speaker_adaptive_features = None
        self.speaker_adaptive_features_counts = {}

        if not os.path.exists(audio_db_path):
            os.makedirs(audio_db_path)

        # List directories (each speaker has its own directory)
        speaker_dirs = [d for d in os.listdir(audio_db_path)
                        if os.path.isdir(os.path.join(audio_db_path, d)) and not d.startswith('.')]
        for speaker_name in speaker_dirs:
            speaker_dir = os.path.join(audio_db_path, speaker_name)
            self.logger.info(f"Loading embeddings for {speaker_name}")
            person_features = []

            # Load all existing PKL embedding files
            embedding_files = [f for f in os.listdir(speaker_dir) if f.endswith('.pkl')]
            embedding_basenames = set(os.path.splitext(f)[0] for f in embedding_files)
            for embedding_file in embedding_files:
                embedding_path = os.path.join(speaker_dir, embedding_file)
                try:
                    with open(embedding_path, 'rb') as f:
                        feature = pickle.load(f)
                        person_features.append(feature)
                except Exception as e:
                    self.logger.warning(f"Error loading embedding {embedding_path}: {e}")

            # Create embeddings for audio files that do not yet have a corresponding embedding file
            audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            for audio_file in audio_files:
                audio_basename = os.path.splitext(audio_file)[0]
                if audio_basename not in embedding_basenames:
                    audio_path = os.path.join(speaker_dir, audio_file)
                    try:
                        self.logger.info(f"Creating embedding for existing audio file: {audio_file}")
                        feature = self._infer(audio_path)
                        embedding_filename = f"{audio_basename}.pkl"
                        embedding_path = os.path.join(speaker_dir, embedding_filename)
                        with open(embedding_path, 'wb') as f:
                            pickle.dump(feature, f)
                        person_features.append(feature)
                    except Exception as e:
                        self.logger.warning(f"Error processing audio file {audio_path}: {e}")

            if not person_features:
                self.logger.warning(f"No embeddings could be loaded for {speaker_name}")
                continue

            person_features = np.array(person_features)
            person_features = person_features / (np.linalg.norm(person_features, ord=2, axis=-1, keepdims=True))
            num_features = len(person_features)
            reference_emb = np.sum(person_features, axis=0) / num_features

            self.speaker_names.append(speaker_name)
            self.speaker_adaptive_features_counts[speaker_name] = num_features
            if self.speaker_features is None:
                self.speaker_features = reference_emb[np.newaxis, :]
                self.speaker_adaptive_features = reference_emb[np.newaxis, :]
            else:
                self.speaker_features = np.vstack([self.speaker_features, reference_emb])
                self.speaker_adaptive_features = np.vstack([self.speaker_adaptive_features, reference_emb])

            self.logger.info(f"Loaded {num_features} embeddings for {speaker_name}")

    def _infer(self, audio_path: str) -> np.ndarray:
        return request_audio_inference(audio_path, os.path.basename(self.audio_db), self.audio_inferer_url)[0]

    def _update_features(self, speaker_name: str, new_feature: np.ndarray):
        new_feature_normalized = new_feature / np.linalg.norm(new_feature, ord=2)
        idx = self.speaker_names.index(speaker_name)
        count = self.speaker_adaptive_features_counts[speaker_name]
        current_updated = self.speaker_adaptive_features[idx]
        new_updated = (current_updated * count + new_feature_normalized) / (count + 1)
        self.speaker_adaptive_features[idx] = new_updated
        self.speaker_adaptive_features_counts[speaker_name] = count + 1
