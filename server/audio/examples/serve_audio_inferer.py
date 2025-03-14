"""This script runs the audio inferer server."""
import os

from openmmla.services.audio import AudioInferer
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')
app = create_app(
    class_type=AudioInferer,
    endpoint='infer',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path, 'use_cuda': True},
)

# gunicorn -w 1 -b 0.0.0.0:5000 serve_audio_inferer:app
# hypercorn -w 1 -b 0.0.0.0:5000 serve_audio_inferer:app
# kill -9 $(lsof -ti:5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
