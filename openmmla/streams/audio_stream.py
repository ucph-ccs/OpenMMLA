import socket

import pyaudio

from openmmla.streams.stream_receiver import StreamReceiver


class AudioStream(StreamReceiver):
    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        self.format = kwargs.get('format', pyaudio.paInt16)
        self.channels = kwargs.get('channels', 1)
        self.rate = kwargs.get('rate', 16000)
        self.chunk_size = kwargs.get('chunk_size', 1024)
        self.sample_width = kwargs.get('sample_width', 2)
        self.input_device_index = kwargs.get('input_device_index', None)
        self.output_device_index = kwargs.get('output_device_index', None)
        self.stream = None
        self.running = False

        if source == 'pyaudio':
            self.p = pyaudio.PyAudio()
        elif source == 'socket':
            self.socket = kwargs.get('socket', None)
            if self.socket is None:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_type = kwargs.get('socket_type', 'tcp')
            if self.socket_type == 'udp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError(f"Unsupported source '{source}' for AudioStream.")

    def start(self):
        if self.source == 'pyaudio':
            self.stream = self.p.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      input_device_index=self.input_device_index,
                                      frames_per_buffer=self.chunk_size)
            self.running = True
        elif self.source == 'socket':
            if self.socket_type == 'tcp':
                self.socket.listen(1)
                self.conn, self.addr = self.socket.accept()
            self.running = True

    def stop(self):
        self.running = False
        if self.source == 'pyaudio':
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        elif self.source == 'socket':
            if self.socket_type == 'tcp':
                self.conn.close()
            self.socket.close()

    def read(self, duration=None):
        if self.source == 'pyaudio':
            if duration is None:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                return data
            else:
                frames = []
                total_frames = int(self.rate / self.chunk_size * duration)
                for _ in range(total_frames):
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                return b''.join(frames)
        elif self.source == 'socket':
            if self.socket_type == 'tcp':
                data = self.conn.recv(self.chunk_size)
                return data
            elif self.socket_type == 'udp':
                data, _ = self.socket.recvfrom(self.chunk_size)
                return data

    def write(self, data):
        if self.source == 'pyaudio':
            # Implement playback if needed
            pass
        elif self.source == 'socket':
            if self.socket_type == 'tcp':
                self.conn.sendall(data)
            elif self.socket_type == 'udp':
                # Destination address should be specified
                dest_addr = self.config.get('dest_addr', None)
                if dest_addr:
                    self.socket.sendto(data, dest_addr)
                else:
                    raise ValueError("Destination address must be specified for UDP socket.")
