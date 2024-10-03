"""This script runs the voice activity detector server."""
import os

from openmmla.services.audio import VoiceActivityDetector
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
app = create_app(
    class_type=VoiceActivityDetector,
    endpoint='vad',
    method_name='vad',
    class_args={'project_dir': project_dir, 'use_cuda': True, 'use_onnx': True},
)

# gunicorn -w 1 -b 0.0.0.0:5004 vad_server:app
# hypercorn -w 1 -b 0.0.0.0:5004 vad_server:app
# kill -9 $(lsof -ti:5004)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, threaded=True)
