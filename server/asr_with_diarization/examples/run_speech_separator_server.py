"""This script runs the speech separator server."""
import os

from openmmla.services.audio import SpeechSeparator
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.ini')
app = create_app(
    class_type=SpeechSeparator,
    endpoint='separate',
    method_name='separate_speech',
    class_args={'project_dir': project_dir, 'config_path': config_path, 'use_cuda': True},
)

# gunicorn -w 1 -b 0.0.0.0:5001 separate_server:app
# hypercorn -w 1 -b 0.0.0.0:5001 separate_server:app
# kill -9 $(lsof -ti:5001)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
