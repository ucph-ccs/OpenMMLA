"""This script runs the video frame analyzer server."""
import os
from openmmla_vision.services.video_frame_analyzer import VideoFrameAnalyzer
from openmmla_vision.utils.server_utils import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

app = create_app(
    class_type=VideoFrameAnalyzer,
    endpoint='process',
    method_name='process_image',
    class_args={'project_dir': project_dir},
)

# gunicorn -w 1 -b 0.0.0.0:5000 image_processor_server:app
# hypercorn -w 1 -b 0.0.0.0:5000 image_processor_server:app
# To kill the server: kill -9 $(lsof -ti:5000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
