import gc
import json
import math
import os
import threading

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from flask import request, jsonify
from http import HTTPStatus

from openmmla.services.server import Server
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.io import write_bytes_to_wav
from openmmla.utils.audio.languages import azure_locale, language_code
from openmmla.utils.audio.transcriber import WhisperXTranscriber, get_transcriber


def _as_bool(value, default=False) -> bool:
    """a config or request value as a bool: true/false, 1/0, yes/no, on/off; `default` for none."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def _filled(value):
    """a config value, or None for an <...> placeholder the user never replaced (or nothing)."""
    text = str(value or "").strip()
    return None if not text or (text.startswith("<") and text.endswith(">")) else value


class SpeechTranscriber(Server):
    """SpeechTranscriber transcribes the audio signal. It receives audio signal from base station and sends back the
    transcribed text. Supports local model, Azure, and DashScope (Paraformer / Qwen3-ASR) backends.

    A request may name the language to transcribe it in (a base's -lang), which holds for that
    request alone; without one it is the language of this service's config. The answer says which
    language was used.

    A request may also ask for anonymous speaker turns (a base's -dia): a local WhisperX model
    diarizes the file with pyannote and the answer carries `diarization`, [{start, end, speaker}]
    as SPEAKER_00, SPEAKER_01 ... in seconds from the start of the file, and `diarized`, which
    says whether it could (the other backends cannot). `diarize` in the local config does it for
    every file."""

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the speech transcriber.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        self._setup_yaml()
        self._setup_objects()

    def _setup_yaml(self):
        """Load configuration settings from YAML."""
        config = self.config['SpeechTranscriber']
        self.cuda = config.get('cuda', True)
        self.cuda = self.cuda and torch.cuda.is_available()
        self.backend = config.get('backend', 'local')

        if self.backend == 'azure':
            # azure configuration
            azure_config = config.get('azure', {})
            self.subscription_key = azure_config.get('subscription_key')
            self.region = azure_config.get('region')
            self.language = azure_config.get('language', 'en-US')
            self.profanity_option = azure_config.get('profanity_option', 'masked')
            self.word_level = azure_config.get('word_level', False)

            if not self.subscription_key or not self.region:
                raise ValueError("Azure Speech requires subscription_key and region to be configured")

            self.logger.info(
                f"Using Azure Speech-to-Text backend, region: {self.region}, language: {self.language}, word_level: {self.word_level}")
        elif self.backend in ('dashscope', 'paraformer'):
            if self.backend == 'paraformer':
                self.logger.warning("backend 'paraformer' is deprecated, use 'dashscope' instead")
            self.backend = 'dashscope'

            # read from 'dashscope' section, fall back to legacy 'paraformer' section
            ds_config = config.get('dashscope', config.get('paraformer', {}))

            self.api_key = ds_config.get('api_key')
            self.model = ds_config.get('model', 'paraformer-realtime-v2')
            self.word_level = ds_config.get('word_level', False)

            # determine model family for API routing
            self._is_paraformer = self.model.startswith('paraformer')
            self._is_qwen_asr = self.model.startswith('qwen3-asr') or self.model.startswith('qwen-asr')

            if not self._is_paraformer and not self._is_qwen_asr:
                raise ValueError(
                    f"Unsupported DashScope model '{self.model}'. "
                    "Model name must start with 'paraformer' or 'qwen3-asr'."
                )

            if not self.api_key:
                self.api_key = os.environ.get('DASHSCOPE_API_KEY')
                if not self.api_key:
                    raise ValueError(
                        "DashScope backend requires api_key in config or DASHSCOPE_API_KEY environment variable"
                    )

            language = ds_config.get('language', ds_config.get('language_hints', ['zh', 'en', 'ja']))

            if self._is_paraformer:
                self.language_hints = language if isinstance(language, list) else [language]
                self.logger.info(
                    f"Using DashScope Paraformer backend, model: {self.model}, "
                    f"language_hints: {self.language_hints}, word_level: {self.word_level}")

            if self._is_qwen_asr:
                self.ds_region = ds_config.get('region', 'intl')
                self.language = language[0] if isinstance(language, list) else language
                self.enable_itn = ds_config.get('enable_itn', False)
                if self.word_level:
                    self.logger.warning(
                        "word_level timestamps are not supported by Qwen3-ASR-Flash in synchronous mode; ignoring"
                    )
                    self.word_level = False
                self.logger.info(
                    f"Using DashScope Qwen-ASR backend, model: {self.model}, "
                    f"region: {self.ds_region}, language: {self.language}, enable_itn: {self.enable_itn}")
        else:
            # local model configuration
            local_config = config.get('local', {})
            self.tr_model = local_config.get('model', 'base.en')
            self.language = local_config.get('language', 'en')
            self.word_level = local_config.get('word_level', False)

            if self.word_level and not self.tr_model.startswith('whisperx/'):
                raise ValueError(
                    "Word-level timestamps are only supported with Azure backend or WhisperX models. "
                    f"Current model '{self.tr_model}' does not support word-level timestamps. "
                    "Use Azure backend or switch to WhisperX model with format 'whisperx/model-name' (e.g., 'whisperx/large-v3')"
                )

            # anonymous speaker turns with every file (a request can ask for them on its own)
            self.diarize = _as_bool(local_config.get('diarize'), False)
            self.diarize_model = _filled(local_config.get('diarize_model'))
            self.hf_token = _filled(local_config.get('hf_token'))
            self.min_speakers = _filled(local_config.get('min_speakers'))
            self.max_speakers = _filled(local_config.get('max_speakers'))
            if self.diarize and not self.tr_model.startswith('whisperx/'):
                raise ValueError(
                    "Diarization needs a WhisperX model (whisperx/model-name), which runs pyannote on the file. "
                    f"Current model '{self.tr_model}' cannot diarize: switch the model or set diarize: false"
                )

            self.logger.info(
                f"Using local transcription model: {self.tr_model}, language: {self.language}, "
                f"word_level: {self.word_level}, diarize: {self.diarize}")

    def _setup_objects(self):
        """Initialize necessary objects based on the selected backend."""
        self.nr_model = pretrained.dns64().cuda() if self.cuda else pretrained.dns64()
        if self.backend == 'azure':
            try:
                import azure.cognitiveservices.speech as speechsdk
                self.speechsdk = speechsdk

                # create speech config
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.subscription_key,
                    region=self.region
                )
                self.speech_config.speech_recognition_language = self.language

                # enable detailed results for word-level timestamps if requested
                if self.word_level:
                    self.speech_config.request_word_level_timestamps()
                    self.speech_config.enable_dictation()

                # set profanity filter if specified
                if hasattr(speechsdk.ProfanityOption, self.profanity_option.upper()):
                    profanity_enum = getattr(speechsdk.ProfanityOption, self.profanity_option.upper())
                    self.speech_config.set_profanity(profanity_enum)

                self.logger.info("Azure Speech SDK initialized successfully")
            except ImportError:
                self.logger.error(
                    "Failed to import Azure Speech SDK. Install it with 'pip install azure-cognitiveservices-speech'")
                raise
        elif self.backend == 'dashscope':
            try:
                import dashscope
                dashscope.api_key = self.api_key

                if self._is_paraformer:
                    self.recognition = dashscope.audio.asr.Recognition(
                        model=self.model,
                        format='wav',
                        sample_rate=16000,
                        language_hints=self.language_hints,
                        callback=None
                    )
                    # one more per language a request asks for that the hints do not hold
                    self.recognitions = {}
                    self.logger.info("DashScope Paraformer Recognition initialized successfully")

                elif self._is_qwen_asr:
                    region_urls = {
                        'intl': 'https://dashscope-intl.aliyuncs.com/api/v1',
                        'cn': 'https://dashscope.aliyuncs.com/api/v1',
                    }
                    base_url = region_urls.get(self.ds_region, region_urls['intl'])
                    dashscope.base_http_api_url = base_url
                    self.logger.info(
                        f"DashScope Qwen-ASR initialized (region={self.ds_region}, endpoint={base_url})")

            except ImportError:
                self.logger.error(
                    "Failed to import DashScope SDK. Install it with 'pip install dashscope'")
                raise
        else:
            # local model - automatically handles both regular whisper and whisperx
            self.transcriber = get_transcriber(self.tr_model, self.language, word_level=self.word_level,
                                               use_cuda=self.cuda, diarize=self.diarize,
                                               diarize_model=self.diarize_model, hf_token=self.hf_token,
                                               min_speakers=self.min_speakers, max_speakers=self.max_speakers)
            self.logger.info("Local speech transcription models initialized")

        # common lock for thread safety
        self.transcriber_lock = threading.Lock()


    def describe(self) -> dict:
        """What this transcriber runs with, for GET /transcribe/info: the backend and its
        model and language, as resolved at start (see openmmla.utils.session_provenance)."""
        info = {'service': self.__class__.__name__, 'backend': self.backend, 'cuda': bool(self.cuda),
                'word_level': bool(getattr(self, 'word_level', False))}
        if self.backend == 'azure':
            info.update(region=getattr(self, 'region', None), language=getattr(self, 'language', None),
                        profanity_option=getattr(self, 'profanity_option', None))
        elif self.backend == 'dashscope':
            info.update(model=getattr(self, 'model', None), region=getattr(self, 'ds_region', None),
                        language=getattr(self, 'language_hints', None) or getattr(self, 'language', None),
                        enable_itn=getattr(self, 'enable_itn', None))
        else:
            info.update(model=getattr(self, 'tr_model', None), language=getattr(self, 'language', None),
                        diarize=bool(getattr(self, 'diarize', False)),
                        diarize_model=getattr(self, 'diarize_model', None))
        return {name: value for name, value in info.items() if value is not None}

    def process_request(self):
        """Transcribe the audio using the configured backend.

        Returns:
            A tuple containing the JSON response (transcribed text) and status code.
        """
        if request.files:
            audio_file_path = None
            try:
                with self.transcriber_lock:  # acquire lock
                    base_id = request.values.get('base_id')
                    fr = int(request.values.get('fr', 16000))
                    # the language this request is to be transcribed in, for this request alone
                    language = (request.values.get('language') or '').strip() or None
                    # whether this request asks for speaker turns; None leaves it to the config
                    asked = request.values.get('diarize')
                    diarize = _as_bool(asked) if asked not in (None, '') else None
                    audio_file = request.files['audio']
                    audio_file_path = self._get_temp_file_path('transcribe_audio', base_id, 'wav')
                    write_bytes_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                    # route to appropriate transcription method
                    used = self._language_used(language)
                    self.logger.info(f"Starting transcription for {base_id}"
                                     f"{f' in {used}' if used else ''}...")
                    if self.backend == 'azure':
                        self._apply_nr(audio_file_path)
                        normalize_decibel(infile=audio_file_path, rms_level=-20)
                        response = self._transcribe_with_azure(audio_file_path, language)
                    elif self.backend == 'dashscope':
                        if self._is_paraformer:
                            response = self._transcribe_with_paraformer(audio_file_path, language)
                        else:
                            response = self._transcribe_with_qwen_asr(audio_file_path, language)
                    else:
                        self._apply_nr(audio_file_path)
                        normalize_decibel(infile=audio_file_path, rms_level=-20)
                        response = self._transcribe_with_local_model(audio_file_path, language, diarize)

                    # what it was transcribed in, so that a base sees whether its -lang was taken
                    if used:
                        response.setdefault("language", used)
                    # and whether it was diarized, so that a base sees whether its -dia was taken
                    if diarize:
                        response.setdefault("diarized", False)
                    self.logger.info(f"Finished transcription for {base_id}")
                    return jsonify(response), 200

            except Exception as e:
                self.logger.error(f"Exception during transcribing", exc_info=True)
                return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500
            finally:
                # clean up
                if audio_file_path and os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary file {audio_file_path}: {e}")

                if self.backend not in ['azure', 'dashscope']:
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400

    def _language_used(self, language: str | None) -> str | None:
        """the language the backend is asked to transcribe in: the request's, else this service's
        own. Each backend in its own form (Azure a locale, the others a code)."""
        if self.backend == 'azure':
            return azure_locale(language) or self.language
        if self.backend == 'dashscope' and self._is_paraformer:
            return language_code(language) or (self.language_hints[0] if self.language_hints else None)
        return language_code(language) or language_code(getattr(self, 'language', None))

    def _transcribe_with_local_model(self, audio_file_path, language=None, diarize=None):
        """Transcribe audio using local model.
        
        Args:
            audio_file_path: Path to the audio file
            language: the language to transcribe this file in; None takes the configured one
            diarize: whether to diarize this file; None takes the configured `diarize`
            
        Returns:
            Dict with the transcribed text, its words when word_level, and when diarized its
            speaker turns (`diarization`) and `diarized: True`
        """
        if isinstance(self.transcriber, WhisperXTranscriber):
            result = self.transcriber.transcribe(audio_file_path, language=language_code(language), diarize=diarize)
        else:
            result = self.transcriber.transcribe(audio_file_path, language=language_code(language))
        # WhisperX answers (text, words, turns, segments), the other transcribers the text alone
        if isinstance(result, tuple):
            text, words, turns = (tuple(result) + (None, None, None))[:3]
            words = words or []
        else:
            text, words, turns = result, [], None
        response = {"text": text}
        if self.word_level:
            response["words"] = words
        if turns is not None:
            response["diarization"] = turns
            response["diarized"] = True
        return response

    def _transcribe_with_azure(self, audio_file_path, language=None):
        """Transcribe audio using Azure Speech-to-Text service.
        
        Args:
            audio_file_path: Path to the audio file
            language: the language to transcribe this file in; None takes the configured one
            
        Returns:
            Dict containing transcribed text and optionally word-level timestamps if word_level=True
        """
        audio_config = self.speechsdk.audio.AudioConfig(filename=audio_file_path)
        speech_recognizer = self.speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
            # the recognizer's own language, where speech_config carries the configured one
            language=azure_locale(language)
        )
        result = speech_recognizer.recognize_once_async().get()

        if result.reason == self.speechsdk.ResultReason.RecognizedSpeech:
            text = result.text

            # base response
            response = {"text": text}

            # add word-level timestamps if requested and available
            if self.word_level and hasattr(result, 'json') and result.json:
                try:
                    json_result = json.loads(result.json)

                    # extract word-level timestamps from NBest results
                    if 'NBest' in json_result and json_result['NBest']:
                        nbest = json_result['NBest'][0]
                        if 'Words' in nbest and nbest['Words']:
                            words = []
                            for word_data in nbest['Words']:
                                word_info = {
                                    'word': word_data.get('Word', ''),
                                    'start': word_data.get('Offset', 0) / 10000000,  # convert ticks to seconds
                                    'end': (word_data.get('Offset', 0) + word_data.get('Duration', 0)) / 10000000,
                                    'confidence': word_data.get('Confidence', 0.0)
                                }
                                words.append(word_info)
                            response["words"] = words
                            self.logger.debug(f"Extracted {len(words)} word-level timestamps")
                        else:
                            self.logger.warning("Word-level timestamps requested but not available in Azure response")
                except Exception as e:
                    self.logger.error(f"Failed to parse Azure JSON result for word timestamps: {e}")

            return response

        else:
            error_msg = f"Azure recognition failed with reason: {result.reason}"
            raise RuntimeError(error_msg)

    def _transcribe_with_paraformer(self, audio_file_path, language=None):
        """Transcribe audio using Paraformer (DashScope) service with synchronous call.
        
        NOTE: Paraformer Recognition instances are single-use only (cannot be reused like Azure's speech_config).
        We create a new instance for each request and properly clean it up afterward to avoid file descriptor leaks.
        
        Args:
            audio_file_path: Path to the audio file
            language: the language to transcribe this file in; None takes the configured hints
            
        Returns:
            Dict containing transcribed text and optionally word-level timestamps if word_level=True
        """
        try:
            # perform recognition
            result = self._paraformer(language).call(audio_file_path)
            
            # check if recognition was successful
            if result.status_code != HTTPStatus.OK:
                error_msg = f"Paraformer recognition failed: {result.message}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # get sentence result - returns a list of sentence dictionaries
            sentences = result.get_sentence()
            
            if not sentences or not isinstance(sentences, list):
                self.logger.warning(f"No sentences found in result")
                return {"text": ""}
            
            self.logger.debug(f"Paraformer returned {len(sentences)} sentences")
            
            # combine all sentences
            all_text = []
            all_words = []
            
            for sentence in sentences:
                if 'text' in sentence and sentence['text']:
                    all_text.append(sentence['text'])
                    
                    # collect word-level timestamps if requested and available
                    if self.word_level and 'words' in sentence:
                        for word in sentence['words']:
                            # extract and convert timestamps, keeping only relevant fields
                            word_info = {
                                'text': word.get('text', ''),
                                'start_time': word.get('begin_time', 0) / 1000.0,
                                'end_time': word.get('end_time', 0) / 1000.0,
                                'punctuation': word.get('punctuation', '')
                            }
                            all_words.append(word_info)
            
            # build response
            full_text = ' '.join(all_text)
            response = {"text": full_text}
            
            if self.word_level and all_words:
                response['words'] = all_words
                self.logger.debug(f"Extracted {len(all_words)} word-level timestamps")
            
            self.logger.debug(f"Paraformer transcription completed: {full_text}")
            return response
            
        finally:
            gc.collect()

    def _paraformer(self, language=None):
        """the Recognition that hears `language` (one per language asked for, as the hints are
        given when it is made); the configured one for None."""
        code = language_code(language)
        if not code or code in self.language_hints:
            return self.recognition
        if code not in self.recognitions:
            import dashscope
            self.recognitions[code] = dashscope.audio.asr.Recognition(
                model=self.model, format='wav', sample_rate=16000, language_hints=[code], callback=None)
        return self.recognitions[code]

    def _transcribe_with_qwen_asr(self, audio_file_path, language=None):
        """Transcribe audio using DashScope Qwen-ASR model via MultiModalConversation API.
        
        Args:
            audio_file_path: Path to the audio file
            language: the language to transcribe this file in; None takes the configured one
            
        Returns:
            Dict containing transcribed text and optional language/emotion metadata
        """
        try:
            import dashscope

            messages = [
                {"role": "user", "content": [{"audio": audio_file_path}]}
            ]

            asr_options = {}
            asked = language_code(language) or self.language
            if asked:
                asr_options["language"] = asked
            if self.enable_itn:
                asr_options["enable_itn"] = self.enable_itn

            kwargs = dict(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                result_format="message",
            )
            if asr_options:
                kwargs["asr_options"] = asr_options

            response = dashscope.MultiModalConversation.call(**kwargs)

            if response.status_code != HTTPStatus.OK:
                error_msg = (
                    f"Qwen-ASR recognition failed: code={response.status_code}, "
                    f"message={response.message}"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            choices = response.output.get("choices", [])
            if not choices:
                self.logger.warning("No choices in Qwen-ASR response")
                return {"text": ""}

            message = choices[0].get("message", {})
            content = message.get("content", [])
            text = content[0].get("text", "") if content else ""

            result = {"text": text}

            annotations = message.get("annotations", [])
            if annotations:
                ann = annotations[0]
                if "language" in ann:
                    result["language"] = ann["language"]
                if "emotion" in ann:
                    result["emotion"] = ann["emotion"]

            self.logger.debug(f"Qwen-ASR transcription completed: {text}")
            return result

        finally:
            gc.collect()

    def _apply_nr(self, input_path: str):
        """Apply noise reduction to the audio.
        
        Args:
            input_path: Path to the audio file
        """
        chunk_size = 30
        sr = torchaudio.info(input_path).sample_rate
        total_duration = torchaudio.info(input_path).num_frames / sr
        output_chunks = []

        if total_duration == 0:
            raise ValueError("Total duration of the audio is zero.")

        try:
            for start in range(0, math.ceil(total_duration), chunk_size):
                chunk, _ = torchaudio.load(input_path, num_frames=int(chunk_size * sr), frame_offset=int(start * sr))
                if chunk.nelement() == 0:
                    continue  # Skip empty chunks

                if self.cuda:
                    chunk = chunk.cuda()
                chunk = convert_audio(chunk, sr, self.nr_model.sample_rate, self.nr_model.chin)

                with torch.no_grad():
                    denoised_chunk = self.nr_model(chunk[None])[0]
                output_chunks.append(denoised_chunk.cpu())

                if self.cuda:
                    torch.cuda.empty_cache()

            if not output_chunks:
                raise RuntimeError("No chunks were processed. Check the audio file and processing steps.")

            processed_audio = torch.cat(output_chunks, dim=1)  # Concatenate and save the processed chunks
            torchaudio.save(input_path, processed_audio, sample_rate=self.nr_model.sample_rate, bits_per_sample=16)
        except Exception as e:
            raise RuntimeError("Error in apply_nr") from e
        finally:
            torch.cuda.empty_cache()
            gc.collect()
