import os
import shutil
import threading
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
logs_dir = os.path.join(static_dir, 'logs')
visualizations_dir = os.path.join(static_dir, 'visualizations')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
influx_client = InfluxDBClientWrapper(config_path)

# Load Redis configuration
with open(config_path, 'r') as f:
    redis_config = yaml.safe_load(f)['Redis']

post_time_visualization_timestamps = {}  # Global dictionary to keep track of visualization generation timestamps
real_time_visualization_threads = {}
active_buckets = {}  # Maps bucket names to sets of client IDs
last_sent_data = {}  # Cache to store the last sent data


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
celery = make_celery(app)  # Initialize Celery


@app.route('/api/get_buckets')
def get_buckets():
    session_ids = influx_client.get_all_session_ids()
    return jsonify(session_ids)


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
    """Start the post-time visualization generation task"""
    data = request.json
    session_id = data['session_id']
    # Check if visualizations exist and were generated recently
    last_visualized = post_time_visualization_timestamps.get(session_id)
    visualization_age = datetime.now() - last_visualized if last_visualized else timedelta.max
    if visualization_age > timedelta(minutes=2):  # At least 1 minute old data to regenerate
        generate_post_time_visualization.delay(session_id)
        print("Post-time visualization task started")
        post_time_visualization_timestamps[session_id] = datetime.now()
    return jsonify({'message': "Post-time visualization started"})


@app.route('/api/get_post_time_visualizations/<session_id>')
def get_post_time_visualizations(session_id):
    post_time_visualization_path = os.path.join(visualizations_dir, session_id, 'post-time')
    valid_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.html']
    files = []
    try:
        files = [f for f in os.listdir(post_time_visualization_path) if
                 os.path.isfile(os.path.join(post_time_visualization_path, f)) and any(
                     f.endswith(ext) for ext in valid_extensions)]
    except FileNotFoundError:
        pass
    finally:
        # file_urls is the flask route to access the static folder files, not the actual file path
        file_urls = [url_for('static', filename=f"visualizations/{session_id}/post-time/{file}") for file in files]
    return jsonify({'files': file_urls})


@app.route('/api/get_logs/<session_id>')
def get_logs(session_id):
    log_path = os.path.join(logs_dir, session_id)
    try:
        log_files = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
        # log_urls is the flask route to call the download_log function, not the actual file path
        log_urls = [{'name': file, 'url': f"/logs/{session_id}/{file}"} for file in log_files]
        return jsonify({'logs': log_urls})
    except FileNotFoundError:
        return jsonify({'message': 'Log files not found', 'logs': []}), 404


@app.route('/logs/<session_id>/<filename>')
def download_log(session_id, filename):
    log_path = os.path.join(logs_dir, session_id)
    return send_from_directory(log_path, filename, as_attachment=True)


@app.route('/api/real_time_visualize/<session_id>')
def real_time_visualize(session_id):
    """Start real-time visualization for the specified bucket"""
    # Check if we need to start a new thread for this bucket
    if session_id not in real_time_visualization_threads:
        stop_event = Event()  # Create a new stop event for this thread
        thread = threading.Thread(target=emit_realtime_data, args=(session_id, stop_event))
        thread.daemon = True
        thread.start()
        real_time_visualization_threads[session_id] = (thread, stop_event)
        print(f"Starting real-time visualization thread for bucket: {session_id}")
    return jsonify({'message': "Real-time visualization started"})


def emit_realtime_data(session_id, stop_event):
    global last_sent_data

    while not stop_event.is_set():
        try:
            recognition_data = fetch_latest_entry(session_id, EVENT_TYPE_ASR_RECOGNITION, influx_client)
            transcription_data = fetch_latest_entry(session_id, EVENT_TYPE_ASR_TRANSCRIPTION, influx_client)
            relations = fetch_latest_entry(session_id, EVENT_TYPE_IPS_RELATION, influx_client)
            graph_data = relations.get('graph', None) if relations else None
            timestamp = relations.get('window_start_time', None) if relations else None
            position_data = {}
            if timestamp:
                position_data = get_node_positions(session_id, influx_client, timestamp)

            data = {'recognition': None, 'transcription': None, 'graph': None, 'positions': None}
            recognition_new = False
            transcription_new = False
            graph_new = False

            if session_id not in last_sent_data:
                last_sent_data[session_id] = {}

            # Update recognition data
            if recognition_data and (session_id not in last_sent_data or recognition_data['window_start_time']
                                     != last_sent_data[session_id].get('recognition_last_timestamp')):
                last_sent_data[session_id]['recognition_last_timestamp'] = recognition_data['window_start_time']
                data['recognition'] = recognition_data
                recognition_new = True

            # Update transcription data
            if transcription_data and (session_id not in last_sent_data or transcription_data['window_start_time']
                                       != last_sent_data[session_id].get('transcription_last_timestamp')):
                last_sent_data[session_id]['transcription_last_timestamp'] = transcription_data['window_start_time']
                data['transcription'] = transcription_data
                transcription_new = True

            # Update graph and position data
            if timestamp and (session_id not in last_sent_data or timestamp != last_sent_data[session_id].get(
                    'graph_last_timestamp')):
                last_sent_data[session_id]['graph_last_timestamp'] = timestamp
                data['graph'] = graph_data
                data['positions'] = position_data
                graph_new = True

            if recognition_new or transcription_new or graph_new:
                socketio.emit('realtime_data', {'bucket': session_id, 'data': data})

        except Exception as e:
            print(f"An unexpected error occurred during data fetching: {e}")


@socketio.on('join_bucket')
def handle_join_bucket(data):
    session_id = data['session_id']
    client_id = data['client_id']  # Assuming client sends this
    print(f"Client {client_id} joined bucket: {session_id}")
    # Initialize the set for the bucket if it doesn't exist
    if session_id not in active_buckets:
        active_buckets[session_id] = set()
    # Add the client ID to the set for the bucket
    active_buckets[session_id].add(client_id)


@socketio.on('connect')
def handle_connect():
    """SocketIO event handler for when a client connects"""
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    # Iterate over active_buckets to remove this client's ID
    for bucket, clients in active_buckets.items():
        if request.sid in clients:
            clients.remove(request.sid)
            print(f"{request.sid} is removed for {bucket}.")
            if not clients:
                if bucket in real_time_visualization_threads:
                    thread, stop_event = real_time_visualization_threads[bucket]
                    stop_event.set()  # Signal the thread to stop
                    thread.join()
                    del real_time_visualization_threads[bucket]
                    print(f"{bucket} has no clients connected and its thread has been cleaned up.")
    # Clean up empty entries
    for bucket in list(active_buckets.keys()):
        if not active_buckets[bucket]:
            del active_buckets[bucket]


# gunicorn -k gevent -w 1 -b 0.0.0.0:5050 app:app
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5050, host='0.0.0.0')
