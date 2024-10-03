"""This script runs the speech transcriber server."""
import os

from openmmla.services.audio import SpeechTranscriber
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
app = create_app(
    class_type=SpeechTranscriber,
    endpoint='transcribe',
    method_name='transcribe_audio',
    class_args={'project_dir': project_dir, 'config_path': config_path, 'use_cuda': True},
)

# gunicorn -w 1 -b 0.0.0.0:5000 transcribe_server:app
# hypercorn -w 1 -b 0.0.0.0:5000 transcribe_server:app
# kill -9 $(lsof -ti:5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
