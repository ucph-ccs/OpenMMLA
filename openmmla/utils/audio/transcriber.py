"""Transcriber classes for transcribing audio files using Hugging Face and OpenAI Whisper models."""
import contextlib
import inspect
import math
import os
import zlib
from abc import ABC, abstractmethod

import librosa
import torch
import whisper
import whisperx
from transformers import pipeline

from openmmla.utils.audio.languages import language_code


def _unfilled(value) -> bool:
    """an <...> placeholder the user never replaced counts as no value at all."""
    text = str(value or "").strip()
    return not text or (text.startswith("<") and text.endswith(">"))


def _speaker_bound(value):
    """a min/max speaker count as a positive int, or None: an unfilled <int> placeholder or any
    other non-number is simply no bound."""
    if _unfilled(value):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4  # Whisper's own


def _compression_ratio(text: str) -> float:
    """how many times zlib shrinks `text`: Whisper's own measure of a degenerate transcript, a
    token or a phrase written over and over compressing many times better than speech does."""
    data = text.encode("utf-8")
    return len(data) / max(len(zlib.compress(data)), 1)


def _degenerate_threshold(value) -> float | None:
    """the compression ratio above which a segment, its loops cut, is dropped as a hallucination:
    nothing or an unfilled placeholder keeps Whisper's 2.4, a number is taken as given, 0 (or
    less) turns the dropping off; a non-number keeps the default too."""
    text = str(value if value is not None else "").strip()  # 0 is a value here, not nothing
    if not text or (text.startswith("<") and text.endswith(">")):
        return DEFAULT_COMPRESSION_RATIO_THRESHOLD
    try:
        number = float(text)
    except ValueError:
        return DEFAULT_COMPRESSION_RATIO_THRESHOLD
    return number if number > 0 else None


LOOP_MAX_NGRAM = 8  # a repeated phrase of up to this many words is a loop
LOOP_KEEP = 2  # the turns of it that are kept


def collapse_loops(text: str) -> str:
    """the text with any phrase of up to LOOP_MAX_NGRAM words said more than LOOP_KEEP times in a
    row cut down to LOOP_KEEP turns: a stretch of noise can make WhisperX write a word or a phrase
    over and over, most often at the end of a segment that began as real speech, and the turns
    after the second are the model's, not the speaker's."""
    words = text.split()
    lead = ' ' if text[:1].isspace() else ''
    out, i = [], 0
    while i < len(words):
        cut = False
        for n in range(1, LOOP_MAX_NGRAM + 1):
            phrase = words[i:i + n]
            if len(phrase) < n:
                break
            repeats = 1
            while words[i + repeats * n:i + (repeats + 1) * n] == phrase:
                repeats += 1
            if repeats > LOOP_KEEP:
                out.extend(phrase * LOOP_KEEP)
                i += repeats * n
                cut = True
                break
        if not cut:
            out.append(words[i])
            i += 1
    return lead + ' '.join(out) if out else text


def _terminal_diarize_error(error: Exception) -> bool:
    """whether an error making the diarization pipeline will not go away by trying again:
    WhisperX without pyannote, a model that is not there, or one this token may not fetch."""
    if isinstance(error, ImportError):
        return True
    try:
        from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        EntryNotFoundError = GatedRepoError = RepositoryNotFoundError = ()
        HfHubHTTPError = ()
    if isinstance(error, (RepositoryNotFoundError, GatedRepoError, EntryNotFoundError)):
        return True
    if isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        return status in (401, 403, 404)
    return False


def _takes(function, name: str) -> bool:
    """whether `function` (or a callable object) names the parameter `name` in its signature; one
    that only takes **kwargs does not count, as it may pass the name on to what rejects it."""
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def speaker_embeddings(embeddings) -> dict[str, list[float]] | None:
    """the speaker embeddings of a diarized file as {speaker: [float]}, rounded to 5 decimals, the
    speakers whose vector is empty, not finite or all zero (pyannote pads a speaker it has no
    centroid for with zeros, and has NaN for one it had no clean window of) left out; None when
    the pipeline returned none."""
    if not isinstance(embeddings, dict):
        return None
    out = {}
    for speaker, vector in embeddings.items():
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError):
            continue
        if values and all(math.isfinite(value) for value in values) and any(values):
            out[str(speaker)] = [round(value, 5) for value in values]
    return out


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
    """WhisperX transcriber with word-level timestamps and anonymous speaker diarization.

    Diarization (pyannote, through WhisperX) tells who-of-how-many spoke when, as SPEAKER_00,
    SPEAKER_01 ... within one file: no names, no profiles. It is on for every file with
    `diarize`, and a single call can ask for it (a base's -dia)."""

    # the compression ratio above which a segment is a hallucination (_degenerate_threshold)
    compression_ratio_threshold = DEFAULT_COMPRESSION_RATIO_THRESHOLD

    def __init__(self, model_name, language, word_level=False, use_cuda=True, diarize=False,
                 diarize_model=None, hf_token=None, min_speakers=None, max_speakers=None,
                 compression_ratio_threshold=None):
        super().__init__(model_name, language, word_level, use_cuda)

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16

        if model_name.startswith('whisperx/'):
            self.whisperx_model_name = model_name[9:]  # remove 'whisperx/' prefix

        # one alignment model per language (word_level), made as a language turns up, and the
        # languages WhisperX has none for
        self.align_models = {}
        self.align_missing = set()

        # the diarization pipeline: made once, at start when every file is diarized, else at the
        # first call that asks; diarize_failed says why the last attempt could not make it,
        # diarize_terminal that no attempt will (no pyannote, no such model, no access), and
        # diarize_attempts how many were made (a transient error is tried again, a few times)
        self.diarize = bool(diarize)
        self.diarize_model_name = None if _unfilled(diarize_model) else str(diarize_model).strip()
        self.hf_token = ((None if _unfilled(hf_token) else str(hf_token).strip())
                         or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or None)
        self.min_speakers = _speaker_bound(min_speakers)
        self.max_speakers = _speaker_bound(max_speakers)
        self.compression_ratio_threshold = _degenerate_threshold(compression_ratio_threshold)
        self.diarize_pipeline = None
        self.diarize_failed = None
        self.diarize_terminal = False
        self.diarize_attempts = 0
        self._load_models()
        if self.diarize:
            self._diarize_pipeline()

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

    def _clean_segments(self, segments):
        """the segments with their loops cut (collapse_loops), minus the ones whose text still
        compresses more than the threshold allows (a run of one character, and the like); the
        console says what was cut and what was dropped."""
        threshold = self.compression_ratio_threshold
        kept = []
        for segment in segments:
            text = str(segment.get("text") or "")
            cleaned = collapse_loops(text)
            if cleaned != text:
                cut = len(text.split()) - len(cleaned.split())
                print(f"WhisperX looped: {cut} repeated words cut from a segment, now {cleaned.strip()[:80]!r}")
                segment = dict(segment, text=cleaned)
                text = cleaned
            ratio = _compression_ratio(text) if text.strip() else 0.0
            if threshold and ratio > threshold:
                print(f"WhisperX dropped a segment as a hallucination (its text compresses {ratio:.1f} times, "
                      f"over {threshold}): {text.strip()[:80]!r}")
                continue
            kept.append(segment)
        return kept

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

    DIARIZE_MAX_ATTEMPTS = 3

    def _diarize_pipeline(self):
        """the pyannote diarization pipeline, made once; None when it cannot be made (WhisperX
        without pyannote, no Hugging Face token for a gated model, no such model), and
        diarize_failed then says why. A transient error (the hub not answering, the network
        down) is tried again at the next call, up to DIARIZE_MAX_ATTEMPTS times."""
        if self.diarize_pipeline is not None or self.diarize_terminal:
            return self.diarize_pipeline
        if self.diarize_attempts >= self.DIARIZE_MAX_ATTEMPTS:
            return None
        self.diarize_attempts += 1
        try:
            from whisperx.diarize import DiarizationPipeline
            kwargs = {}
            if self.diarize_model_name:
                kwargs['model_name'] = self.diarize_model_name
            # the token goes by the name this WhisperX gives it
            parameters = inspect.signature(DiarizationPipeline.__init__).parameters
            token_key = next((key for key in ('use_auth_token', 'token') if key in parameters), None)
            if token_key and self.hf_token:
                kwargs[token_key] = self.hf_token
            with _torch_load_full():
                self.diarize_pipeline = DiarizationPipeline(device=self.device, **kwargs)
        except Exception as e:
            self.diarize_failed = f"{type(e).__name__}: {e}"
            self.diarize_terminal = _terminal_diarize_error(e)
            if self.diarize_terminal:
                print(f"WhisperX cannot make the diarization pipeline ({self.diarize_failed}): transcripts "
                      f"come without speaker turns. A pyannote pipeline is gated: accept its terms on "
                      f"huggingface.co and give that account's token as hf_token or HF_TOKEN; and the "
                      f"model name must be one that is there.")
            elif self.diarize_attempts >= self.DIARIZE_MAX_ATTEMPTS:
                print(f"WhisperX could not make the diarization pipeline ({self.diarize_failed}) "
                      f"{self.diarize_attempts} times, and gives up: transcripts come without speaker "
                      f"turns until the speech transcriber is started again.")
            else:
                print(f"WhisperX could not make the diarization pipeline ({self.diarize_failed}); "
                      f"it tries again at the next file that asks.")
        return self.diarize_pipeline

    def _diarize(self, audio, segments):
        """the speaker turns of `audio`, the segments with their speakers, and one embedding per
        speaker: (turns, segments, embeddings), the turns None when the pipeline could not be made,
        [] when it heard no one; the embeddings {speaker: [float]} (speaker_embeddings) when this
        WhisperX can return them, else None."""
        pipeline = self._diarize_pipeline()
        if pipeline is None:
            return None, segments, None
        kwargs = {key: value for key, value in (('min_speakers', self.min_speakers),
                                                ('max_speakers', self.max_speakers)) if value}
        if _takes(pipeline, 'return_embeddings'):
            # a WhisperX whose pipeline takes return_embeddings (3.8.6 does): pyannote's centroid of
            # each speaker of the file comes along
            kwargs['return_embeddings'] = True
        try:
            with _torch_load_full():
                answer = pipeline(audio, **kwargs)
            diarization, embeddings = answer if isinstance(answer, tuple) else (answer, None)
            turns = [{'start': round(float(row.start), 3), 'end': round(float(row.end), 3),
                      'speaker': str(row.speaker)} for row in diarization.itertuples()]
            if segments:
                try:
                    from whisperx.diarize import assign_word_speakers
                except ImportError:
                    assign_word_speakers = whisperx.assign_word_speakers
                segments = assign_word_speakers(diarization, {'segments': segments}).get('segments', segments)
        except Exception as e:
            # this file goes without turns; the transcript itself is kept, and the next file is tried
            self.diarize_failed = f"{type(e).__name__}: {e}"
            print(f"WhisperX could not diarize this file ({self.diarize_failed}): its transcript comes "
                  f"without speaker turns.")
            return None, segments, None
        return turns, segments, speaker_embeddings(embeddings)

    def transcribe(self, audio_path, language=None, diarize=None):
        """Transcribe audio with optional alignment for word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: the language to transcribe this file in; None keeps the configured one
            diarize: whether to diarize this file; None keeps the configured `diarize`
            
        Returns:
            (text, words, turns, segments, embeddings): the words empty unless self.word_level; the
            turns the speaker turns [{start, end, speaker}] in seconds from the start of the file,
            None when not diarized; the segments WhisperX's, each with its speaker when diarized; the
            embeddings one vector per speaker of the turns ({speaker: [float]}), None when the file
            was not diarized or this WhisperX returns none
        """
        audio = whisperx.load_audio(audio_path)
        # the language is always named, as the pipeline keeps the one of its last call otherwise, and
        # so is the task: WhisperX 3.8 compares the task it is given to the token id of the tokenizer's
        # ('50359' is not a valid task), which a language of its own would run into
        result = self.model.transcribe(audio, batch_size=self.batch_size, task="transcribe",
                                       language=language_code(language) or language_code(self.language))
        # a stretch of noise can come back as a word or a phrase written over and over, which
        # the batched pipeline does not notice: the loop is cut to two turns, and a segment that
        # is garbage still (a run of one character) goes, before alignment and diarization
        result["segments"] = self._clean_segments(result.get("segments") or [])
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
        turns, embeddings = None, None
        if self.diarize if diarize is None else diarize:
            if segments:
                turns, segments, embeddings = self._diarize(audio, segments)
                if words:
                    words = [word for segment in segments for word in segment.get("words", [])]
            elif self._diarize_pipeline() is not None:
                # nothing was transcribed, so there is no one to attribute: the pipeline is fine,
                # and the file simply has no turns
                turns = []
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return text, words, turns, segments, embeddings


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


def get_transcriber(model_name, language="en", word_level=False, use_cuda=True, **diarization):
    """the transcriber of `model_name`; `diarization` (diarize, diarize_model, hf_token,
    min_speakers, max_speakers, and compression_ratio_threshold) goes to a WhisperX transcriber, the
    only kind that diarizes and drops hallucinated segments."""
    transcriber_class = TRANSCRIBER_MAP.get(model_name)

    if transcriber_class is None:
        raise UnsupportedModelError(f"Unsupported model: {model_name}")

    if transcriber_class is WhisperXTranscriber:
        return transcriber_class(model_name, language, word_level=word_level, use_cuda=use_cuda, **diarization)
    return transcriber_class(model_name, language, word_level=word_level, use_cuda=use_cuda)
