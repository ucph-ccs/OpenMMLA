import gc
import math
import os
import threading

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from flask import request, jsonify

from openmmla.services.server import Server
from openmmla.utils.audio.auga import normalize_decibel
from openmmla.utils.audio.io import write_bytes_to_wav
from openmmla.utils.audio.transcriber import get_transcriber


class SpeechTranscriber(Server):
    """SpeechTranscriber transcribes the audio signal. It receives audio signal from base station and sends back the
    transcribed text. Supports both local model and Azure Speech-to-Text backend."""

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
        
        # determine which backend to use (local or azure)
        self.backend = config.get('backend', 'local')
        
        if self.backend == 'azure':
            # azure configuration
            azure_config = config.get('azure', {})
            self.subscription_key = azure_config.get('subscription_key')
            self.region = azure_config.get('region')
            self.language = azure_config.get('language', 'en-US')
            self.profanity_option = azure_config.get('profanity_option', 'masked')
            
            if not self.subscription_key or not self.region:
                raise ValueError("Azure Speech requires subscription_key and region to be configured")
                
            self.logger.info(f"Using Azure Speech-to-Text backend, region: {self.region}, language: {self.language}")
        else:
            # local model configuration
            local_config = config.get('local', {})
            self.cuda = local_config.get('cuda', True)
            self.cuda = self.cuda and torch.cuda.is_available()
            self.tr_model = local_config.get('model', 'base.en')
            self.language = local_config.get('language', 'en')
            self.logger.info(f"Using local transcription model: {self.tr_model}, language: {self.language}")

    def _setup_objects(self):
        """Initialize necessary objects based on the selected backend."""
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
                
                # set profanity filter if specified
                if hasattr(speechsdk.ProfanityOption, self.profanity_option.upper()):
                    profanity_enum = getattr(speechsdk.ProfanityOption, self.profanity_option.upper())
                    self.speech_config.set_profanity(profanity_enum)
                
                self.logger.info("Azure Speech SDK initialized successfully")
            except ImportError:
                self.logger.error("Failed to import Azure Speech SDK. Install it with 'pip install azure-cognitiveservices-speech'")
                raise
        else:
            # initialize local models
            self.nr_model = pretrained.dns64().cuda() if self.cuda else pretrained.dns64()
            self.transcriber = get_transcriber(self.tr_model, self.language, use_cuda=self.cuda)
            self.logger.info("Local speech transcription models initialized")
            
        # common lock for thread safety
        self.transcriber_lock = threading.Lock()

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
                    audio_file = request.files['audio']
                    audio_file_path = self._get_temp_file_path('transcribe_audio', base_id, 'wav')
                    write_bytes_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                    self.logger.info(f"Starting transcription for {base_id}...")
                    
                    # route to appropriate transcription method
                    if self.backend == 'azure':
                        text = self._transcribe_with_azure(audio_file_path)
                    else:
                        text = self._transcribe_with_local_model(audio_file_path)
                    
                    self.logger.info(f"Finished transcription for {base_id}")
                    return jsonify({"text": text}), 200
                    
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
                
                if self.backend != 'azure':
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400

    def _transcribe_with_local_model(self, audio_file_path):
        """Transcribe audio using local model.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        self._apply_nr(audio_file_path)
        normalize_decibel(infile=audio_file_path, rms_level=-20)
        return self.transcriber.transcribe(audio_file_path)

    def _transcribe_with_azure(self, audio_file_path):
        """Transcribe audio using Azure Speech-to-Text service.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        self.logger.info(f"Transcribing with Azure: {audio_file_path}")
        
        # create audio configuration from file
        audio_config = self.speechsdk.audio.AudioConfig(filename=audio_file_path)
        
        # create speech recognizer
        speech_recognizer = self.speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        
        # start recognition and get result
        self.logger.debug("Starting Azure speech recognition")
        result = speech_recognizer.recognize_once_async().get()
        
        # check results
        if result.reason == self.speechsdk.ResultReason.RecognizedSpeech:
            self.logger.debug(f"Azure recognition successful: {result.text}")
            return result.text
        elif result.reason == self.speechsdk.ResultReason.NoMatch:
            self.logger.warning("Azure could not recognize speech")
            return ""
        elif result.reason == self.speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            self.logger.error(f"Azure speech recognition canceled: {cancellation.reason}")
            if cancellation.reason == self.speechsdk.CancellationReason.Error:
                self.logger.error(f"Azure error details: {cancellation.error_details}")
            return ""
        
        # fallback
        return ""

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
