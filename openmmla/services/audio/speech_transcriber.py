import configparser
import gc
import math
import os
import threading

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from flask import request, jsonify

from openmmla.utils.audio.processing import write_frames_to_wav, normalize_rms
from openmmla.utils.audio.transcriber import Transcriber
from openmmla.utils.logger import get_logger


class SpeechTranscriber:
    """Speech transcriber transcribe the audio signal into speech. It receives audio signal from base station and sends
     back transcription text"""

    def __init__(self, project_dir, config_path, use_cuda=True):
        """Initialize the speech transcriber.

        Args:
            project_dir: the project directory.
            use_cuda: whether to use CUDA or not.
        """
        # Check if the project directory exists
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Project directory not found at {project_dir}")

        if os.path.isabs(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(project_dir, config_path)

        # Check if the configuration file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        config = configparser.ConfigParser()
        config.read(self.config_path)
        tr_model = config['SpeechTranscriber']['model']
        language = config['SpeechTranscriber']['language']

        self.server_logger_dir = os.path.join(project_dir, 'logger')
        self.server_file_folder = os.path.join(project_dir, 'temp')
        os.makedirs(self.server_logger_dir, exist_ok=True)
        os.makedirs(self.server_file_folder, exist_ok=True)

        self.logger = get_logger('whisper', os.path.join(self.server_logger_dir, 'transcribe_server.log'))
        self.cuda_enable = use_cuda and torch.cuda.is_available()

        # Initialize denoiser and transcriber
        self.nr_model = pretrained.dns64().cuda() if self.cuda_enable else pretrained.dns64()
        self.transcriber = Transcriber(tr_model, language=language, use_cuda=self.cuda_enable)
        self.transcriber_lock = threading.Lock()

    def apply_nr(self, input_path: str) -> str:
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

                if self.cuda_enable:
                    chunk = chunk.cuda()
                chunk = convert_audio(chunk, sr, self.nr_model.sample_rate, self.nr_model.chin)

                with torch.no_grad():
                    denoised_chunk = self.nr_model(chunk[None])[0]
                output_chunks.append(denoised_chunk.cpu())

                if self.cuda_enable:
                    torch.cuda.empty_cache()

            if not output_chunks:
                raise RuntimeError("No chunks were processed. Check the audio file and processing steps.")

            processed_audio = torch.cat(output_chunks, dim=1)  # Concatenate and save the processed chunks
            torchaudio.save(input_path, processed_audio, sample_rate=self.nr_model.sample_rate, bits_per_sample=16)
        except Exception as e:
            raise RuntimeError(f"Error in apply_nr: {e}")
        finally:
            torch.cuda.empty_cache()
            gc.collect()
        return input_path

    def transcribe_audio(self):
        """Transcribe the audio.

        Returns:
            A tuple containing the JSON response and status code.
        """
        try:
            with self.transcriber_lock:  # Acquire lock
                base_id = request.values.get('base_id')
                fr = int(request.values.get('fr', 16000))
                audio_file = request.files['audio']
                audio_file_path = os.path.join(self.server_file_folder, f'transcribe_audio_{base_id}.wav')
                write_frames_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                self.logger.info(f"starting transcribe for {base_id}...")
                if fr == 16000:
                    self.apply_nr(audio_file_path)
                normalize_rms(infile=audio_file_path, rms_level=-20)
                text = self.transcriber.transcribe(audio_file_path)
                self.logger.info(f"finished transcribe for {base_id}.")

            return jsonify({"text": text}), 200
        except Exception as e:
            self.logger.error(f"during transcribing, {e} happens.")
            return jsonify({"error": str(e)}), 500
        finally:
            torch.cuda.empty_cache()
            gc.collect()
