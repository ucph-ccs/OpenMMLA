"""This script runs the audio resampler server."""
import os

from openmmla.services.asr_diarization import AudioResampler
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
app = create_app(
    class_type=AudioResampler,
    endpoint='resample',
    method_name='process_request',
    class_args={'project_dir': project_dir},
)

# gunicorn -w 1 -b 0.0.0.0:5001 serve_audio_resampler:app
# hypercorn -w 1 -b 0.0.0.0:5001 serve_audio_resampler:app
# kill -9 $(lsof -ti:5001)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)
