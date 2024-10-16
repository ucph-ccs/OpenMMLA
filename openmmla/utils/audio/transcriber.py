from abc import ABC, abstractmethod

import librosa
import torch
import whisper
from transformers import pipeline


class Transcriber(ABC):
    def __init__(self, model_name, language, use_cuda=True):
        self.model_name = model_name
        self.language = language
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    @abstractmethod
    def transcribe(self, audio_path, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class WhisperTranscriber(Transcriber):
    def __init__(self, model_name, language, use_cuda=True):
        super().__init__(model_name, language, use_cuda)
        self.transcriber = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path, fp16=False):
        text = self.transcriber.transcribe(audio_path, fp16=fp16, language=self.language)["text"]
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return text


# https://billtcheng2013.medium.com/faster-audio-transcribing-with-openai-whisper-and-huggingface-transformers-dc088243803d
class WhisperTransformerTranscriber(Transcriber):
    def __init__(self, model_name, language, task="transcribe", use_cuda=True):
        super().__init__(model_name, language, use_cuda)
        self.multilingual = False if model_name.endswith('en') else True
        self.task = task
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=self.device,
        )

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
    def __init__(self, model_name, language, use_cuda=True):
        super().__init__(model_name, language, use_cuda)
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name, device=self.device)

    def transcribe(self, audio_path, sampling_rate=16000):
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        transcription = self.transcriber(audio)
        print(transcription)
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
    # Add more models as needed
}


def get_transcriber(model_name, language="en", use_cuda=True):
    transcriber_class = TRANSCRIBER_MAP.get(model_name)

    if transcriber_class is None:
        raise UnsupportedModelError(f"Unsupported model: {model_name}")

    return transcriber_class(model_name, language, use_cuda=use_cuda)


# Usage example:
# transcriber = get_transcriber("openai/whisper-tiny.en", language="en", use_cuda=False)
# result = transcriber.transcribe("path/to/audio/file.wav")
# print(result)

# For Roest model:
# transcriber = get_transcriber("alexandrainst/roest-315m", language="da", use_cuda=False)
# result = transcriber.transcribe("path/to/audio/file.wav")
# print(result)
