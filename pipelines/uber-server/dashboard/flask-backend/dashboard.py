import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from threading import Event

import yaml
from celery import Celery
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_socketio import SocketIO

from openmmla.analytics.asr.analyze import asr_session_analysis
from openmmla.analytics.ips.analyze import ips_session_analysis
from openmmla.utils.constants import EVENT_TYPE_ASR_RECOGNITION, EVENT_TYPE_ASR_TRANSCRIPTION, EVENT_TYPE_IPS_RELATION
from openmmla.utils.querys import fetch_latest_entry, get_node_positions
from openmmla.utils.client import InfluxDBClientWrapper

project_dir = os.getcwd()
config_path = os.path.join(project_dir, 'config.yml')
static_dir = os.path.join(project_dir, 'static')
frontend_dir = os.path.abspath(os.path.join(project_dir, '..', 'frontend'))
logs_dir = os.path.join(static_dir, 'logs')
visualizations_dir = os.path.join(static_dir, 'visualizations')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
influx_client = InfluxDBClientWrapper(config_path)

with open(config_path, 'r') as f:
    redis_config = yaml.safe_load(f)['Redis']

post_time_visualization_timestamps = {}  # session_id -> last generation time
real_time_visualization_threads = {}  # session_id -> (thread, stop_event)
active_sessions = {}  # session_id -> set of client ids
last_sent_data = {}  # session_id -> last sent timestamps per data type


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}",
        broker=f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
    )
    celery.conf.update(app.config)
    return celery


def clear_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


clear_folder_contents(visualizations_dir)
clear_folder_contents(logs_dir)
celery = make_celery(app)


# ======= Frontend pages =======
@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')


@app.route('/realtime')
def realtime_page():
    return send_from_directory(frontend_dir, 'realtime.html')


@app.route('/posttime')
def posttime_page():
    return send_from_directory(frontend_dir, 'posttime.html')


@app.route('/css/<path:filename>')
def frontend_css(filename):
    return send_from_directory(os.path.join(frontend_dir, 'css'), filename)


# ======= API =======
@app.route('/api/get_sessions')
def get_sessions():
    return jsonify(influx_client.get_all_session_ids())


@celery.task
def generate_post_time_visualization(session_id):
    try:
        print("Generating post-time visualization...")
        asr_session_analysis(static_dir, session_id, influx_client)
        ips_session_analysis(static_dir, session_id, influx_client)
    except KeyError as e:
        print(f"Key not found, {e}")


@app.route('/api/post_time_visualize', methods=['POST'])
def post_time_visualize():
    """Start the post-time visualization generation task (throttled)."""
    session_id = request.json['session_id']
    last_visualized = post_time_visualization_timestamps.get(session_id)
    visualization_age = datetime.now() - last_visualized if last_visualized else timedelta.max
    if visualization_age > timedelta(minutes=2):
        generate_post_time_visualization.delay(session_id)
        post_time_visualization_timestamps[session_id] = datetime.now()
        print("Post-time visualization task started")
    return jsonify({'message': "Post-time visualization started"})


@app.route('/api/get_post_time_visualizations/<session_id>')
def get_post_time_visualizations(session_id):
    visualization_dir = os.path.join(visualizations_dir, session_id, 'post-time')
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.html')
    try:
        files = [f for f in os.listdir(visualization_dir)
                 if os.path.isfile(os.path.join(visualization_dir, f)) and f.endswith(valid_extensions)]
    except FileNotFoundError:
        files = []
    file_urls = [url_for('static', filename=f"visualizations/{session_id}/post-time/{f}") for f in files]
    return jsonify({'files': file_urls})


@app.route('/api/get_logs/<session_id>')
def get_logs(session_id):
    log_path = os.path.join(logs_dir, session_id)
    try:
        log_files = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
        log_urls = [{'name': f, 'url': f"/logs/{session_id}/{f}"} for f in log_files]
        return jsonify({'logs': log_urls})
    except FileNotFoundError:
        return jsonify({'message': 'Log files not found', 'logs': []}), 404


@app.route('/logs/<session_id>/<filename>')
def download_log(session_id, filename):
    return send_from_directory(os.path.join(logs_dir, session_id), filename, as_attachment=True)


@app.route('/api/real_time_visualize/<session_id>')
def real_time_visualize(session_id):
    """Start the real-time data emitter for the specified session."""
    if session_id not in real_time_visualization_threads:
        stop_event = Event()
        thread = threading.Thread(target=emit_realtime_data, args=(session_id, stop_event))
        thread.daemon = True
        thread.start()
        real_time_visualization_threads[session_id] = (thread, stop_event)
        print(f"Starting real-time visualization thread for session: {session_id}")
    return jsonify({'message': "Real-time visualization started"})


def emit_realtime_data(session_id, stop_event):
    while not stop_event.is_set():
        try:
            recognition = fetch_latest_entry(session_id, EVENT_TYPE_ASR_RECOGNITION, influx_client)
            transcription = fetch_latest_entry(session_id, EVENT_TYPE_ASR_TRANSCRIPTION, influx_client)
            relations = fetch_latest_entry(session_id, EVENT_TYPE_IPS_RELATION, influx_client)
            graph = relations.get('graph') if relations else None
            graph_timestamp = relations.get('window_start_time') if relations else None
            positions = get_node_positions(session_id, influx_client, graph_timestamp) if graph_timestamp else {}

            sent = last_sent_data.setdefault(session_id, {})
            data = {'recognition': None, 'transcription': None, 'graph': None, 'positions': None}

            if recognition and recognition['window_start_time'] != sent.get('recognition'):
                sent['recognition'] = recognition['window_start_time']
                data['recognition'] = recognition

            if transcription and transcription['window_start_time'] != sent.get('transcription'):
                sent['transcription'] = transcription['window_start_time']
                data['transcription'] = transcription

            if graph_timestamp and graph_timestamp != sent.get('graph'):
                sent['graph'] = graph_timestamp
                data['graph'] = graph
                data['positions'] = positions

            if any(v is not None for v in data.values()):
                socketio.emit('realtime_data', {'session': session_id, 'data': data})

        except Exception as e:
            print(f"An unexpected error occurred during data fetching: {e}")

        time.sleep(1)


# ======= SocketIO events =======
@socketio.on('join_session')
def handle_join_session(data):
    session_id = data['session_id']
    client_id = data['client_id']
    active_sessions.setdefault(session_id, set()).add(client_id)
    print(f"Client {client_id} joined session: {session_id}")


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    for session_id, clients in active_sessions.items():
        if request.sid in clients:
            clients.remove(request.sid)
            print(f"{request.sid} is removed for {session_id}.")
            if not clients and session_id in real_time_visualization_threads:
                thread, stop_event = real_time_visualization_threads.pop(session_id)
                stop_event.set()
                thread.join()
                print(f"{session_id} has no clients connected and its thread has been cleaned up.")
    for session_id in [s for s, clients in active_sessions.items() if not clients]:
        del active_sessions[session_id]


# gunicorn -k gevent -w 1 -b 0.0.0.0:5050 dashboard:app
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5050, host='0.0.0.0')
