# https://billtcheng2013.medium.com/faster-audio-transcribing-with-openai-whisper-and-huggingface-transformers-dc088243803d
import librosa
import torch
import whisper
from transformers import pipeline


class Transcriber:
    def __init__(self, model_name, language="en", use_cuda=True):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.transcriber = whisper.load_model(model_name, device=self.device)
        self.language = language

    def transcribe(self, audio_path, fp16=False):
        text = self.transcriber.transcribe(audio_path, fp16=fp16, language=self.language)["text"]
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return text


class TransformerTranscriber:
    def __init__(self, model_name, language="en", task="transcribe", use_cuda=False):
        self.multilingual = False if model_name.endswith('en') else True
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.language = language
        self.task = task
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_name}",
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

# if __name__ == "__main__":
#     transcriber = TransformerTranscriber("large-v3", language="en", use_cuda=False)
#     path = '../audio/post-time/chunks/audio20231128_10_20/chunk_11_jesper.wav'
#     result = transcriber.transcribe(path)
#     print(result)
