import os
import pickle
import tempfile
import time

import numpy as np
import yaml

from openmmla.services.asr.requests import request_audio_inference
from openmmla.utils.audio.files import segment_wav
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import resolve_url, build_service_url


class AudioRecognizer:
    """A speaker recognition system that manages speaker profiles and performs real-time speaker identification.

    This class provides a complete framework for speaker recognition, including:
    1. Speaker Profile Management:
        - Registration of new speakers
        - Storage of speaker embeddings
        - Maintenance of both original and adaptive speaker profiles
        - Support for audio file and embedding persistence

    2. Speaker Recognition Features:
        - Real-time speaker identification
        - Adaptive speaker profiles that update during recognition
        - Support for both general and candidate-specific recognition
        - Similarity score calculation using both original and adapted embeddings

    3. Key Components:
        - Embedding Generation: Converts audio segments into speaker embeddings
        - Profile Storage: Manages speaker profiles in a directory structure
        - Adaptive Learning: Updates speaker profiles based on recognition confidence
        - Vectorized Operations: Efficient similarity computations using NumPy

    The system maintains two sets of embeddings for each speaker:
    - Original embeddings: Initial speaker profiles from registration
    - Adaptive embeddings: Continuously updated profiles during recognition

    Recognition decisions are made by comparing input audio against both original and adaptive embeddings, choosing the
    highest similarity score.
    """

    logger = get_logger('audio-recognizer')

    def __init__(self, config_path: str, profiles_dir: str, store: bool = False, selected_speakers: list[str] = None):
        """Initialize the AudioRecognizer.

        Args:
            config_path: path to the configuration file
            profiles_dir: path to the profiles directory containing speaker profiles
            store: whether to store audio files or not (default: False)
            selected_speakers: list of speaker names to load (default: None, loads all)
        """
        config = yaml.safe_load(open(config_path, 'r'))
        self.profiles_dir = profiles_dir
        self.audio_inferer_url = build_service_url(config, config['Server']['asr']['audio_inferer'])
        self.store = store
        print(f"Audio inferer URL: {self.audio_inferer_url}")

        self.speaker_names = []  # List of speaker names
        self.speaker_features = None  # np.array of origin embeddings (each row is a speaker's embedding)
        self.speaker_adaptive_features = None  # np.array of updated embeddings (same shape/order as speaker_features)
        self.speaker_adaptive_features_counts = {}  # Dict of speaker names to the weight of updated embeddings
        self.selected_speakers = selected_speakers  # Track which speakers are currently selected

        self._load_profiles(self.profiles_dir, selected_speakers)
        self.logger.info("Successfully initialize audio recognizer.")

    def register(self, path: str, user_name: str):
        """Register a new user or add new embeddings to an existing user, it will normalize the audio and extract the embeddings.
        
        Args:
            path: path to the audio file to register
            user_name: name of the user to register
        """
        # Check if user already exists
        if user_name in self.speaker_names:
            self.logger.info(f"User '{user_name}' already exists. Adding new embeddings to existing profile.")
            user_index = self.speaker_names.index(user_name)
        else:
            self.logger.info(f"Registering new user '{user_name}'.")
            user_index = None

        # Create embeddings directory for user
        user_embeddings_dir = os.path.join(self.profiles_dir, user_name)
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
                # Get embedding
                audio_path = os.path.join(temp_dir, audio_file)
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
                if self.store:
                    audio_filename = f"{base_filename}_{timestamp}.wav"
                    audio_save_path = os.path.join(user_embeddings_dir, audio_filename)
                    with open(audio_path, 'rb') as src_file:
                        audio_data = src_file.read()
                    with open(audio_save_path, 'wb') as dst_file:
                        dst_file.write(audio_data)
                    self.logger.debug(f"Saved audio file: {audio_filename}")

            self.logger.info(
                f"Processed {len(features)} audio segments and saved embeddings{' and audio files' if self.store else ''}.")

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

    def reset_profiles(self, profiles_dir: str, selected_speakers: list[str] = None):
        """Reset and reload speaker profiles from the profiles directory.
        
        Args:
            profiles_dir: path to the profiles directory
            selected_speakers: list of speaker names to load (default: None, loads all)
        """
        self.profiles_dir = profiles_dir
        self.selected_speakers = selected_speakers
        self._load_profiles(self.profiles_dir, selected_speakers)
        self.logger.info("Successfully reload audio database.")

    def deregister(self, speaker_name: str):
        """Deregister a speaker from the recognizer.
        
        Args:
            speaker_name: name of the speaker to deregister
        """
        if speaker_name not in self.speaker_names:
            self.logger.warning(f"Speaker '{speaker_name}' not found in recognizer.")
            return
            
        # Find the index of the speaker
        speaker_index = self.speaker_names.index(speaker_name)
        
        # Remove from all arrays
        self.speaker_names.pop(speaker_index)
        
        if len(self.speaker_names) == 0:
            # If no speakers left, reset arrays
            self.speaker_features = None
            self.speaker_adaptive_features = None
        else:
            # Remove the corresponding row from feature arrays
            self.speaker_features = np.delete(self.speaker_features, speaker_index, axis=0)
            self.speaker_adaptive_features = np.delete(self.speaker_adaptive_features, speaker_index, axis=0)
        
        # Remove from counts dictionary
        if speaker_name in self.speaker_adaptive_features_counts:
            del self.speaker_adaptive_features_counts[speaker_name]
            
        self.logger.info(f"Successfully deregistered speaker '{speaker_name}'.")

    def delete_speaker_profile(self, speaker_name: str):
        """Delete a speaker's profile files from disk and deregister from recognizer.
        
        Args:
            speaker_name: name of the speaker to delete
        """
        speaker_dir = os.path.join(self.profiles_dir, speaker_name)
        
        if not os.path.exists(speaker_dir):
            self.logger.warning(f"Speaker profile directory not found: {speaker_dir}")
            return
            
        try:
            # Delete the entire speaker directory
            import shutil
            shutil.rmtree(speaker_dir)
            self.logger.info(f"Deleted speaker profile directory: {speaker_dir}")
            
            # Deregister from recognizer
            self.deregister(speaker_name)
            
        except Exception as e:
            self.logger.error(f"Error deleting speaker profile '{speaker_name}': {e}")

    def _load_profiles(self, profiles_dir: str, selected_speakers: list[str] = None):
        """Load speaker profiles from the profiles directory.
        
        Args:
            profiles_dir: path to the profiles directory
            selected_speakers: list of speaker names to load (default: None, loads all)
        """
        # Initialize/reinitialize speaker profiles
        self.speaker_names = []
        self.speaker_features = None
        self.speaker_adaptive_features = None
        self.speaker_adaptive_features_counts = {}

        if not os.path.exists(profiles_dir):
            os.makedirs(profiles_dir)

        # List directories (each speaker has its own directory)
        speaker_dirs = [d for d in os.listdir(profiles_dir)
                        if os.path.isdir(os.path.join(profiles_dir, d)) and not d.startswith('.')]
        
        # Filter by selected speakers if specified
        if selected_speakers is not None:
            speaker_dirs = [d for d in speaker_dirs if d in selected_speakers]
        for speaker_name in speaker_dirs:
            speaker_dir = os.path.join(profiles_dir, speaker_name)
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
        emb = request_audio_inference(audio_path, os.path.basename(self.profiles_dir), self.audio_inferer_url)
        # Server may return (1, D) batched or (D,) flat; always return the 1-D embedding vector
        return emb[0] if emb.ndim > 1 else emb

    def _update_features(self, speaker_name: str, new_feature: np.ndarray):
        new_feature_normalized = new_feature / np.linalg.norm(new_feature, ord=2)
        idx = self.speaker_names.index(speaker_name)
        count = self.speaker_adaptive_features_counts[speaker_name]
        current_updated = self.speaker_adaptive_features[idx]
        new_updated = (current_updated * count + new_feature_normalized) / (count + 1)
        self.speaker_adaptive_features[idx] = new_updated
        self.speaker_adaptive_features_counts[speaker_name] = count + 1
