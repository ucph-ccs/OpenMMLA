"""This script runs the speech transcriber server."""
import os

from openmmla.services.asr.speech_transcriber import SpeechTranscriber
from openmmla.utils.apps import create_app

project_dir = os.environ.get('PROJECT_DIR')
config_path = os.environ.get('CONFIG_PATH')
if not project_dir:
    raise RuntimeError("Environment variable PROJECT_DIR must be set, please set it via export or -c.")
if not config_path:
    raise RuntimeError("Environment variable CONFIG_PATH must be set, please set it via export or -c.")
app = create_app(
    class_type=SpeechTranscriber,
    endpoint='transcribe',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# gunicorn -w 1 -b 0.0.0.0:5005 serve_speech_transcriber:app
# kill -9 $(lsof -ti:5005)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True)
