"""This script runs the voice activity detector server."""
import os

from openmmla.services.asr.voice_activity_detector import VoiceActivityDetector
from openmmla.utils.apps import create_app

default_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
project_dir = os.environ.get('OPENMMLA_PROJECT_DIR', default_project_dir)
config_path = os.environ.get('OPENMMLA_CONFIG_PATH', os.path.join(project_dir, 'config.yml'))
app = create_app(
    class_type=VoiceActivityDetector,
    endpoint='vad',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# gunicorn -w 1 -b 0.0.0.0:5006 serve_voice_activity_detector:app
# kill -9 $(lsof -ti:5006)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, threaded=True)
