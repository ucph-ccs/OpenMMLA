"""This script runs the speech separator server."""
import os

from openmmla.services.asr.speech_separator import SpeechSeparator
from openmmla.utils.apps import create_app

default_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
project_dir = os.environ.get('OPENMMLA_PROJECT_DIR', default_project_dir)
config_path = os.environ.get('OPENMMLA_CONFIG_PATH', os.path.join(project_dir, 'config.yml'))
app = create_app(
    class_type=SpeechSeparator,
    endpoint='separate',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# gunicorn -w 1 -b 0.0.0.0:5004 serve_speech_separator:app
# kill -9 $(lsof -ti:5004)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, threaded=True)
