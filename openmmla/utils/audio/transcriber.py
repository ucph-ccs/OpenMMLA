"""Transcriber classes for transcribing audio files using Hugging Face and OpenAI Whisper models."""
import contextlib
from abc import ABC, abstractmethod

import librosa
import torch
import whisper
import whisperx
from transformers import pipeline

from openmmla.utils.audio.languages import language_code


@contextlib.contextmanager
def _torch_load_full():
    """torch.load as it was before 2.6 (weights_only=False), which the model files of WhisperX and
    of its alignment models need."""
    original_load = torch.load

    def patched_load(f, map_location=None, pickle_module=None, weights_only=None, mmap=None, **kwargs):
        return original_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False,
                             mmap=mmap, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


class Transcriber(ABC):
    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        self.model_name = model_name
        self.language = language
        self.word_level = word_level
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    @abstractmethod
    def transcribe(self, audio_path, **kwargs):
        """Transcribe `audio_path`; `language` transcribes one file in another language than the
        one this transcriber was made with (a base's -lang), and None keeps that one."""
        raise NotImplementedError("Subclasses must implement this method")


class WhisperXTranscriber(Transcriber):
    """WhisperX transcriber with word-level timestamps and speaker diarization support."""

    def __init__(self, model_name, language, word_level=False, use_cuda=True):
        super().__init__(model_name, language, word_level, use_cuda)

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16

        if model_name.startswith('whisperx/'):
            self.whisperx_model_name = model_name[9:]  # remove 'whisperx/' prefix

        # one alignment model per language (word_level), made as a language turns up, and the
        # languages WhisperX has none for
        self.align_models = {}
        self.align_missing = set()
        self._load_models()

    def _load_models(self):
        """Load WhisperX models with PyTorch 2.6+ compatibility fix."""
        try:
            with _torch_load_full():
                self.model = whisperx.load_model(self.whisperx_model_name, self.device, compute_type=self.compute_type, language=language_code(self.language))

                # the configured language's alignment model, so that the first transcription does
                # not wait for it; another language's is made when it is asked for
                if self.word_level:
                    self._align_model(self.language)

        except ImportError as e:
            raise ImportError(f"WhisperX not installed. Install with: pip install whisperx") from e

    def _align_model(self, language):
        """the alignment model that gives `language` its word timestamps, made once per language;
        None when WhisperX has none for it, and the transcript then comes without words."""
        code = language_code(language)
        if not code or code in self.align_missing:
            return None
        if code not in self.align_models:
            try:
                with _torch_load_full():
                    self.align_models[code] = whisperx.load_align_model(language_code=code, device=self.device)
            except Exception as e:
                self.align_missing.add(code)
                print(f"WhisperX has no alignment model for '{code}' ({e}): its transcripts come without "
                      f"word timestamps")
                return None
        return self.align_models[code]

    def transcribe(self, audio_path, language=None):
        """Transcribe audio with optional alignment for word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: the language to transcribe this file in; None keeps the configured one
            
        Returns:
            (text, words), the words empty unless self.word_level
        """
        audio = whisperx.load_audio(audio_path)
        # the language is always named, as the pipeline keeps the one of its last call otherwise, and
        # so is the task: WhisperX 3.8 compares the task it is given to the token id of the tokenizer's
        # ('50359' is not a valid task), which a language of its own would run into
        result = self.model.transcribe(audio, batch_size=self.batch_size, task="transcribe",
                                       language=language_code(language) or language_code(self.language))
        text = ''
        words = []
        
        if self.word_level and result.get("segments"):
            align = self._align_model(result.get("language") or language or self.language)
            if align is not None:
                # perform alignment
                result = whisperx.align(
                    result["segments"],
                    align[0],
                    align[1],
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

    def transcribe(self, audio_path, fp16=False, language=None):
        text = self.transcriber.transcribe(audio_path, fp16=fp16,
                                           language=language_code(language) or language_code(self.language))["text"]
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

    def transcribe(self, audio_path, sampling_rate=16000, return_timestamps=False, start_time=None, end_time=None,
                   language=None):
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        if end_time and end_time > start_time:
            start_sample = librosa.time_to_samples(start_time, sr=sampling_rate)
            end_sample = librosa.time_to_samples(end_time, sr=sampling_rate)
            audio = audio[start_sample:end_sample]

        code = language_code(language) or language_code(self.language)
        if self.multilingual and code:
            transcription = self.transcriber(audio, batch_size=8, return_timestamps=return_timestamps,
                                             generate_kwargs={"task": self.task,
                                                              "language": f"<|{code}|>"})
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

    def transcribe(self, audio_path, sampling_rate=16000, language=None):
        # a Danish model: it transcribes Danish whatever language is asked for
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
