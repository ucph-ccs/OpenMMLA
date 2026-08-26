"""This script runs the Multi-Angle VLLM frame analyzer server."""
import os

from openmmla.services.vfa.multi_angle_vllm_frame_analyzer import MultiAngleVLLMFrameAnalyzer
from openmmla.utils.apps import create_app

project_dir = os.environ.get('PROJECT_DIR')
config_path = os.environ.get('CONFIG_PATH')
if not project_dir:
    raise RuntimeError("Environment variable PROJECT_DIR must be set, please set it via export or -c.")
if not config_path:
    raise RuntimeError("Environment variable CONFIG_PATH must be set, please set it via export or -c.")

app = create_app(
    class_type=MultiAngleVLLMFrameAnalyzer,
    endpoint='vllm',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# Run with: python serve_multi_angle_vllm_frame_analyzer.py
# Or with gunicorn: gunicorn -w 1 -b 0.0.0.0:5007 serve_multi_angle_vllm_frame_analyzer:app
# To kill the server: kill -9 $(lsof -ti:5007)  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
