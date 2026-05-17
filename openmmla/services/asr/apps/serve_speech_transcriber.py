"""This script runs the speech transcriber server."""
import os

from openmmla.services.asr.speech_transcriber import SpeechTranscriber
from openmmla.utils.apps import create_app

default_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
project_dir = os.environ.get('OPENMMLA_PROJECT_DIR', default_project_dir)
config_path = os.environ.get('OPENMMLA_CONFIG_PATH', os.path.join(project_dir, 'config.yml'))
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
