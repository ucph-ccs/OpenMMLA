"""This script runs the audio enhancer server."""
import os

from openmmla.services.audio import AudioEnhancer
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
app = create_app(
    class_type=AudioEnhancer,
    endpoint='enhance',
    method_name='enhance_audio',
    class_args={'project_dir': project_dir, 'use_cuda': True},
)

# gunicorn -w 1 -b 0.0.0.0:5000 enhance_server:app
# hypercorn -w 1 -b 0.0.0.0:5000 enhance_server:app
# kill -9 $(lsof -ti:5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
