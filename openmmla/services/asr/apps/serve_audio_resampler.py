"""This script runs the audio resampler server."""
import os

from openmmla.services.asr.audio_resampler import AudioResampler
from openmmla.utils.apps import create_app

project_dir = os.environ.get('PROJECT_DIR')
if not project_dir:
    raise RuntimeError("Environment variable PROJECT_DIR must be set, please set it via export or -c.")
app = create_app(
    class_type=AudioResampler,
    endpoint='resample',
    method_name='process_request',
    class_args={'project_dir': project_dir},
)

# gunicorn -w 1 -b 0.0.0.0:5002 serve_audio_resampler:app
# kill -9 $(lsof -ti:5002)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)
