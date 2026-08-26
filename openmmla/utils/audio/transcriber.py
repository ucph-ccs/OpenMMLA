"""Transcriber classes for transcribing audio files using Hugging Face and OpenAI Whisper models."""
from abc import ABC, abstractmethod

import librosa
import torch
import whisper
import whisperx
from transformers import pipeline


class Transcriber(ABC):
    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        self.model_name = model_name
        self.language = language
        self.word_level = word_level
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    @abstractmethod
    def transcribe(self, audio_path, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class WhisperXTranscriber(Transcriber):
    """WhisperX transcriber with word-level timestamps and speaker diarization support."""

    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        super().__init__(model_name, language, word_level, use_cuda)

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16

        if model_name.startswith('whisperx/'):
            self.whisperx_model_name = model_name[9:]  # remove 'whisperx/' prefix

        self._load_models()

    def _load_models(self):
        """Load WhisperX models with PyTorch 2.6+ compatibility fix."""
        try:
            # fix for PyTorch 2.6+ weights_only=True default behavior
            original_load = torch.load

            def patched_load(f, map_location=None, pickle_module=None, weights_only=None, mmap=None, **kwargs):
                return original_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False,
                                     mmap=mmap, **kwargs)

            torch.load = patched_load
            try:
                self.model = whisperx.load_model(self.whisperx_model_name, self.device, compute_type=self.compute_type, language=self.language)

                # load align model during initialization if word_level is true
                if self.word_level:
                    # for auto-detection, we'll load it lazily on first transcription
                    if self.language and self.language != 'null':
                        self.align_model, self.align_metadata = whisperx.load_align_model(
                            language_code=self.language,
                            device=self.device
                        )
                    else:
                        self.align_model = None
                        self.align_metadata = None
                else:
                    self.align_model = None
                    self.align_metadata = None
            finally:
                # restore original torch.load function
                torch.load = original_load

        except ImportError as e:
            raise ImportError(f"WhisperX not installed. Install with: pip install whisperx") from e

    def transcribe(self, audio_path):
        """Transcribe audio with optional alignment for word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments (ignored, for compatibility)
            
        Returns:
            Dict with transcribed text and optionally word-level timestamps based on self.word_level
        """
        audio = whisperx.load_audio(audio_path)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        text = ''
        words = []
        
        if self.word_level and result.get("segments"):
            # load alignment model if not already loaded (for auto-detection)
            if self.align_model is None:
                detected_language = result["language"]
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )
            # perform alignment
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            segments = result.get("segments", [])
            text = ''.join([segment["text"] for segment in segments])
            words = [word for segment in segments for word in segment["words"]]
        else:
            segments = result.get("segments", [])
            text = ''.join([segment["text"] for segment in segments])
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return text, words


class WhisperTranscriber(Transcriber):
    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        super().__init__(model_name, language, word_level, use_cuda)
        self.transcriber = whisper.load_model(model_name, device=self.device)
        if self.word_level:
            raise ValueError("Word-level timestamps are not supported for Local Whisper models")

    def transcribe(self, audio_path, fp16=False):
        text = self.transcriber.transcribe(audio_path, fp16=fp16, language=self.language)["text"]
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return text


# https://billtcheng2013.medium.com/faster-audio-transcribing-with-openai-whisper-and-huggingface-transformers-dc088243803d
class WhisperTransformerTranscriber(Transcriber):
    def __init__(self, model_name, language, task="transcribe", word_level=False, use_cuda=True):
        super().__init__(model_name, language, word_level, use_cuda)
        self.multilingual = False if model_name.endswith('en') else True
        self.task = task
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=self.device,
        )
        if self.word_level:
            raise ValueError(
                "Word-level timestamps are not supported for Hugging Face Transformers Whisper models. Use WhisperX models (whisperx/model-name) or Azure backend instead.")

    def transcribe(self, audio_path, sampling_rate=16000, return_timestamps=False, start_time=None, end_time=None):
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        if end_time and end_time > start_time:
            start_sample = librosa.time_to_samples(start_time, sr=sampling_rate)
            end_sample = librosa.time_to_samples(end_time, sr=sampling_rate)
            audio = audio[start_sample:end_sample]

        if self.multilingual:
            transcription = self.transcriber(audio, batch_size=8, return_timestamps=return_timestamps,
                                             generate_kwargs={"task": self.task,
                                                              "language": f"<|{self.language}|>"})
        else:
            transcription = self.transcriber(audio, batch_size=8, return_timestamps=return_timestamps)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return transcription["text"] if not return_timestamps else transcription["chunks"]


class RoestTransformerTranscriber(Transcriber):
    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        super().__init__(model_name, language, word_level, use_cuda)
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name, device=self.device)
        if self.word_level:
            raise ValueError("Word-level timestamps are not supported for Roest models")

    def transcribe(self, audio_path, sampling_rate=16000):
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        transcription = self.transcriber(audio)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return transcription["text"]


class UnsupportedModelError(Exception):
    pass


# Direct mapping of model names to transcriber classes
TRANSCRIBER_MAP = {
    'openai/whisper-tiny': WhisperTransformerTranscriber,
    'openai/whisper-tiny.en': WhisperTransformerTranscriber,
    'openai/whisper-base': WhisperTransformerTranscriber,
    'openai/whisper-base.en': WhisperTransformerTranscriber,
    'openai/whisper-small': WhisperTransformerTranscriber,
    'openai/whisper-small.en': WhisperTransformerTranscriber,
    'openai/whisper-medium': WhisperTransformerTranscriber,
    'openai/whisper-medium.en': WhisperTransformerTranscriber,
    'openai/whisper-large': WhisperTransformerTranscriber,
    'openai/whisper-large-v2': WhisperTransformerTranscriber,
    'openai/whisper-large-v3': WhisperTransformerTranscriber,
    'alexandrainst/roest-315m': RoestTransformerTranscriber,
    'tiny': WhisperTranscriber,
    'tiny.en': WhisperTranscriber,
    'base': WhisperTranscriber,
    'base.en': WhisperTranscriber,
    'small': WhisperTranscriber,
    'small.en': WhisperTranscriber,
    'medium': WhisperTranscriber,
    'medium.en': WhisperTranscriber,
    'large': WhisperTranscriber,
    'large-v2': WhisperTranscriber,
    'large-v3': WhisperTranscriber,
    'whisperx/tiny': WhisperXTranscriber,
    'whisperx/tiny.en': WhisperXTranscriber,
    'whisperx/base': WhisperXTranscriber,
    'whisperx/base.en': WhisperXTranscriber,
    'whisperx/small': WhisperXTranscriber,
    'whisperx/small.en': WhisperXTranscriber,
    'whisperx/medium': WhisperXTranscriber,
    'whisperx/medium.en': WhisperXTranscriber,
    'whisperx/large': WhisperXTranscriber,
    'whisperx/large-v2': WhisperXTranscriber,
    'whisperx/large-v3': WhisperXTranscriber,
    # Add more models as needed
}


def get_transcriber(model_name, language="en", word_level=False, use_cuda=True):
    transcriber_class = TRANSCRIBER_MAP.get(model_name)

    if transcriber_class is None:
        raise UnsupportedModelError(f"Unsupported model: {model_name}")

    return transcriber_class(model_name, language, word_level=word_level, use_cuda=use_cuda)
