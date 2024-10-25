import math
import socket
import wave
from typing import Union

import pyaudio
import yaml

from openmmla.utils.audio.auga import apply_gain
from openmmla.utils.logger import get_logger
from openmmla.utils.requests import request_speech_enhancement, request_voice_activity_detection
from openmmla.utils.sockets import clear_socket_udp, read_frames_udp, read_frames_tcp

# Try importing the modules separately
try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    from denoiser import pretrained
except ImportError:
    pretrained = None

try:
    from denoiser.dsp import convert_audio
except ImportError:
    convert_audio = None

try:
    from silero_vad import load_silero_vad, read_audio, save_audio, get_speech_timestamps, collect_chunks
except ImportError:
    load_silero_vad = None
    read_audio = None
    get_speech_timestamps = None
    save_audio = None
    collect_chunks = None


class AudioRecorder:
    logger = get_logger('audio-recorder')

    def __init__(self, config_path, vad_enable, nr_enable, vad_local=False, nr_local=False, use_onnx=True,
                 use_cuda=True):
        config = yaml.safe_load(open(config_path, 'r'))
        self.vad_enable = vad_enable
        self.nr_enable = nr_enable
        self.vad_local = vad_local
        self.nr_local = nr_local
        self.use_onnx = use_onnx
        self.cuda_enable = use_cuda and torch is not None and torch.cuda.is_available()

        if not vad_local or not nr_local:
            self.audio_server_host = socket.gethostbyname(config['Server']['audio_server_host'])
        self.record_rate = int(config['Recorder']['record_rate'])
        self.chunk_size = int(config['Recorder']['chunk_size'])
        self.channels = int(config['Recorder']['channels'])
        self.sample_width = int(config['Recorder']['sample_width'])
        self.format = pyaudio.paInt16  # local recording format
        self.input_device_index = None
        self.output_device_index = None

        self.p = None
        self.stream = None
        self.continuous_recording = False

        self._setup_objects()

    def _setup_objects(self):
        self.vad_model = None
        self.nr_model = None
        if self.vad_enable and self.vad_local:
            try:
                if load_silero_vad is None:
                    raise ImportError
                self.vad_model = load_silero_vad(onnx=self.use_onnx)
                self.logger.info(f"Silero VAD is enabled with onnx set to {self.use_onnx}.")
            except ImportError as e:
                self.logger.warning(f"Error %s occurs while importing torch and torchaudio, disabling VAD.", e,
                                    exc_info=True)
                self.vad_enable = False

        if self.nr_enable and self.nr_local:
            try:
                if pretrained is None or torch is None or torchaudio is None:
                    raise ImportError
                if self.cuda_enable:
                    self.nr_model = pretrained.dns64().cuda()
                else:
                    self.nr_model = pretrained.dns64()
                # self.nr_model = pretrained.dns48()  # Use this for faster computation but lower performance
                self.logger.info("Denoiser is enabled.")
            except ImportError as e:
                self.logger.warning(f"Error %s occurs while importing Denoiser, disabling Denoiser.", e, exc_info=True)
                self.nr_enable = False

    def post_processing(self, input_path: str, sampling_rate: int, inplace: int, base_id: str = None) \
            -> Union[str, None]:
        # Apply audio enhancing, took ~0.1 seconds on Mac, and 2 seconds on Raspberry Pi
        self.apply_nr(input_path=input_path, base_id=base_id)
        # Apply voice activity detection, took ~0.0063 seconds
        return self.apply_vad(input_path=input_path, sampling_rate=sampling_rate, base_id=base_id, inplace=inplace)

    def apply_vad(self, input_path: str, sampling_rate: int, inplace: int, base_id: str = None) -> Union[str, None]:
        if self.vad_enable:
            if self.vad_local:
                wav = read_audio(input_path, sampling_rate=sampling_rate)
                speech_timestamps = get_speech_timestamps(wav, self.vad_model, sampling_rate=sampling_rate)
                if not speech_timestamps:
                    return None
                if inplace:
                    save_audio(input_path, collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate)
                if self.cuda_enable:
                    torch.cuda.empty_cache()
            else:
                if not base_id:
                    raise Exception('Error when requesting speech enhancing service, missing base id.')
                return request_voice_activity_detection(input_path, base_id, inplace, self.audio_server_host)

        return input_path

    def apply_nr(self, input_path: str, base_id: str = None) -> str:
        if self.nr_enable:
            if self.nr_local:
                chunk_size = 30
                sr = torchaudio.info(input_path).sample_rate
                total_duration = torchaudio.info(input_path).num_frames / sr
                output_chunks = []

                for start in range(0, math.ceil(total_duration), chunk_size):
                    chunk, _ = torchaudio.load(input_path, num_frames=int(chunk_size * sr),
                                               frame_offset=int(start * sr))
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

                if not output_chunks:  # Check if the list is empty
                    raise RuntimeError(
                        "No audio chunks were processed. The input audio file might be too short or silent.")

                processed_audio = torch.cat(output_chunks, dim=1)  # Concatenate and save the processed chunks
                torchaudio.save(input_path, processed_audio, sample_rate=self.nr_model.sample_rate, bits_per_sample=16)
            else:
                if not base_id:
                    raise Exception('Error when requesting speech enhancing service, missing base id.')
                request_speech_enhancement(input_path, base_id, self.audio_server_host)

        return input_path

    def start_recording_stream(self):
        """Initialize and start the continuous recording stream."""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format, channels=self.channels,
                                  input_device_index=self.input_device_index,
                                  output_device_index=self.output_device_index, rate=self.record_rate,
                                  input=True, output=False, frames_per_buffer=self.chunk_size)
        self.continuous_recording = True

    def stop_recording_stream(self):
        """Stop and close the continuous recording stream."""
        self.continuous_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

    def read_frames_stream(self, duration: float) -> bytes:
        """Read frames from PyAudio stream for a given duration."""
        expected_bytes = self.record_rate * self.sample_width * duration
        byte_samples = bytearray()
        while len(byte_samples) < expected_bytes:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            byte_samples.extend(data)
        return byte_samples

    def record_registry_stream(self, duration: float, output_path: str, base_id: str) -> str:
        """Record audio using PyAudio stream and save it to a wave file for registering."""
        self.p = pyaudio.PyAudio()
        self.recording_prompt(duration)
        print("Start recording...")

        self.stream = self.p.open(format=self.format, channels=self.channels,
                                  input_device_index=self.input_device_index,
                                  output_device_index=self.output_device_index, rate=self.record_rate,
                                  input=True, output=False, frames_per_buffer=self.chunk_size)
        frames = self.read_frames_stream(duration)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.p.get_sample_size(self.format))
            wav_file.setframerate(self.record_rate)
            wav_file.writeframes(frames)

        print("Recording ended.")
        return self.post_processing(output_path, self.record_rate, 1, base_id)  # Normalization will be done in
        # register's segmentation

    def record_registry_udp(self, duration: float, output_path: str, base_id: str, sock) -> str:
        """Record audio from socket UDP and save it to a wave file for registering"""
        self.recording_prompt(duration)
        clear_socket_udp(sock)
        print("Start recording...")
        frames = read_frames_udp(sock, duration)
        with (wave.open(output_path, 'wb') as wav_file):
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.record_rate)
            wav_file.writeframes(frames)

        print("Recording ended.")
        apply_gain(output_path)
        return self.post_processing(output_path, self.record_rate, 1, base_id)

    def record_registry_tcp(self, duration: float, output_path: str, base_id: str, sock) -> str:
        """Record audio from socket TCP and save it to a wave file for registering"""
        self.recording_prompt(duration)
        sock.listen(1)
        conn, addr = sock.accept()
        print("Start recording...")
        frames = read_frames_tcp(sock, conn, duration)
        with (wave.open(output_path, 'wb') as wav_file):
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.record_rate)
            wav_file.writeframes(frames)

        print("Recording ended.")
        apply_gain(output_path)
        return self.post_processing(output_path, self.record_rate, 1, base_id)

    @staticmethod
    def recording_prompt(seconds: float):
        """Reading prompt for registering."""
        input(f"Press the Enter key to start recording, and read the following sentence in {seconds} seconds:\n"
              "1. The boy was there when the sun rose.\n"
              "2. A rod is used to catch pink salmon.\n"
              "3. The source of the huge river is the clear spring.\n"
              "4. Kick the ball straight and follow through.\n"
              "5. Help the woman get back to her feet.\n"
              "6. A pot of tea helps to pass the evening.\n"
              "7. Smoky fires lack flame and heat.\n"
              "8. The soft cushion broke the man's fall.\n"
              "9. The salt breeze came across from the sea.\n"
              "10. The girl at the booth sold fifty bonds."
              )
        print("------------------------------------------------")
