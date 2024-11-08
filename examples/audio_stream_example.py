import wave
from pathlib import Path

import pyaudio

from openmmla.streams.audio_stream import AudioStream


def main():
    # Configuration
    config = {
        'format': pyaudio.paInt16,
        'channels': 1,
        'rate': 16000,
        'chunk_size': 1024,
        'buffer_duration': 5.0,
        'resample_method': 'audio_librosa',
    }

    # Initialize stream
    stream = AudioStream(source='pyaudio', **config)

    try:
        print("Starting audio stream...")
        stream.start()

        print("Reading 6 seconds of audio...")
        target_rate = 16000
        audio_frame = stream.read(duration=12.0, target_rate=target_rate)

        if audio_frame:
            # Save to WAV file
            output_path = Path("output/recorded_audio.wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving audio to {output_path}...")
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(audio_frame.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(target_rate)
                wav_file.writeframes(audio_frame.to_bytes())

            print("Audio saved successfully!")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping audio stream...")
        stream.stop()
        print("Stream stopped.")


if __name__ == "__main__":
    main()
