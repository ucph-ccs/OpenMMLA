import gc
import io

from flask import request, jsonify, send_file

from openmmla.services.server import Server
from openmmla.utils.audio.io import resample_audio_file, write_frames_to_wav


class AudioResampler(Server):
    """Audio resampler receives audio signals from audio base stations, and then resample them and send it back."""

    def __init__(self, project_dir):
        """Initialize the audio resampler.

        Args:
            project_dir: the project directory.
        """
        super().__init__(project_dir=project_dir)

    def process_request(self):
        """Resample the audio.

        Returns:
            A tuple containing the response (resample audio bytes) and status code.
        """
        if request.files:
            try:
                base_id = request.values.get('base_id')
                fr = int(request.values.get('fr'))
                target_fr = int(request.values.get('target_fr'))
                audio_file = request.files['audio']
                audio_file_path = self._get_temp_file_path('resample_audio', base_id, 'wav')
                write_frames_to_wav(audio_file_path, audio_file.read(), 1, 2, fr)

                self.logger.info(f"starting resampling for {base_id}...")
                resample_audio_file(audio_file_path, target_fr)
                self.logger.info(f"finished resampling for {base_id}...")

                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()
                return send_file(io.BytesIO(audio_data), mimetype="audio/wav"), 200
            except Exception as e:
                self.logger.error(f"during resampling, {e} happens.")
                return jsonify({"error": str(e)}), 500
            finally:
                gc.collect()
        else:
            return jsonify({"error": "No audio file provided"}), 400
