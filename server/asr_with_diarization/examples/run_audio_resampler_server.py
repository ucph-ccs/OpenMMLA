"""This script runs the audio resampler server."""
import os

from openmmla.services.audio import AudioResampler
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
app = create_app(
    class_type=AudioResampler,
    endpoint='resample',
    method_name='resample_audio',
    class_args={'project_dir': project_dir},
)

# gunicorn -w 1 -b 0.0.0.0:5005 resample_server:app
# hypercorn -w 1 -b 0.0.0.0:5005 resample_server:app
# kill -9 $(lsof -ti:5005)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
