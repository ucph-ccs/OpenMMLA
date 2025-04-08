"""This script runs the audio enhancer server."""
import os

from openmmla.services.asr import SpeechEnhancer
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
app = create_app(
    class_type=SpeechEnhancer,
    endpoint='enhance',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# gunicorn -w 1 -b 0.0.0.0:5003 serve_speech_enhancer:app
# kill -9 $(lsof -ti:5003)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, threaded=True)
