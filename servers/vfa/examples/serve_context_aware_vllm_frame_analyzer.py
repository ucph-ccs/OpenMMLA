"""This script runs the Context-Aware VLLM frame analyzer server."""
import os

from openmmla.services.vfa.context_aware_vllm_frame_analyzer import ContextAwareVLLMFrameAnalyzer
from openmmla.utils.apps import create_app

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'config.yml')  # Use your own config file

app = create_app(
    class_type=ContextAwareVLLMFrameAnalyzer,
    endpoint='vllm',
    method_name='process_request',
    class_args={'project_dir': project_dir, 'config_path': config_path},
)

# Run with: python serve_context_aware_vllm_frame_analyzer.py
# Or with gunicorn: gunicorn -w 1 -b 0.0.0.0:5007 serve_context_aware_vllm_frame_analyzer:app
# To kill the server: kill -9 $(lsof -ti:5007)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007) 