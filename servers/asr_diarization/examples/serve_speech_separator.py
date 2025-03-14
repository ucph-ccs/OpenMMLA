"""This script runs the speech separator server."""
import os

from openmmla.services.asr_diarization import SpeechSeparator
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
app = create_app(
    class_type=SpeechSeparator,
    endpoint='separate',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path, 'use_cuda': True},
)

# gunicorn -w 1 -b 0.0.0.0:5003 serve_speech_separator:app
# hypercorn -w 1 -b 0.0.0.0:5003 serve_speech_separator:app
# kill -9 $(lsof -ti:5003)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, threaded=True)
