import os
import shutil
import threading
from datetime import datetime, timedelta
from threading import Event

from celery import Celery
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_socketio import SocketIO

from utils.analyze_utils import session_analysis
from utils.influx_client import InfluxDBClientWrapper
from utils.influx_utils import fetch_latest_entry, get_node_positions

project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
config_path = os.path.join(project_dir, 'conf/uber_base.ini')
logs_dir = os.path.join(project_dir, 'flask-backend/static/logs')
visualizations_dir = os.path.join(project_dir, 'flask-backend/static/visualizations')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
influx_client = InfluxDBClientWrapper(config_path)

post_time_visualization_timestamps = {}  # Global dictionary to keep track of visualization generation timestamps
real_time_visualization_threads = {}
active_buckets = {}  # Maps bucket names to sets of client IDs
last_sent_data = {}  # Cache to store the last sent data


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend='redis://localhost:6379/1',  # Use Redis as the result backend
        broker='redis://localhost:6379/1'  # Use Redis as the broker
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
    bucket_list = influx_client.get_buckets()
    bucket_names = [bucket.name for bucket in bucket_list.buckets if bucket.name not in ['_tasks', '_monitoring']]
    return jsonify(bucket_names)


@celery.task
def generate_post_time_visualization(bucket_name):
    try:
        session_analysis(project_dir, bucket_name, influx_client)
    except KeyError as e:
        print(f"Key not found, {e}")


@app.route('/api/post_time_visualize', methods=['POST'])
def post_time_visualize():
    """Start the post-time visualization generation task"""
    data = request.json
    bucket_name = data['bucket_name']
    # Check if visualizations exist and were generated recently
    last_visualized = post_time_visualization_timestamps.get(bucket_name)
    visualization_age = datetime.now() - last_visualized if last_visualized else timedelta.max
    if visualization_age > timedelta(minutes=2):  # At least 1 minute old data to regenerate
        generate_post_time_visualization.delay(bucket_name)
        post_time_visualization_timestamps[bucket_name] = datetime.now()
    return jsonify({'message': "Post-time visualization started"})


@app.route('/api/get_post_time_visualizations/<bucket_name>')
def get_post_time_visualizations(bucket_name):
    post_time_visualization_path = os.path.join(visualizations_dir, bucket_name, 'post-time')
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
        file_urls = [url_for('static', filename=f"visualizations/{bucket_name}/post-time/{file}") for file in files]
    return jsonify({'files': file_urls})


@app.route('/api/get_logs/<bucket_name>')
def get_logs(bucket_name):
    log_path = os.path.join(logs_dir, bucket_name)
    try:
        log_files = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
        # log_urls is the flask route to call the download_log function, not the actual file path
        log_urls = [{'name': file, 'url': f"/logs/{bucket_name}/{file}"} for file in log_files]
        return jsonify({'logs': log_urls})
    except FileNotFoundError:
        return jsonify({'message': 'Log files not found', 'logs': []}), 404


@app.route('/logs/<bucket_name>/<filename>')
def download_log(bucket_name, filename):
    log_path = os.path.join(logs_dir, bucket_name)
    return send_from_directory(log_path, filename, as_attachment=True)


@app.route('/api/real_time_visualize/<bucket_name>')
def real_time_visualize(bucket_name):
    """Start real-time visualization for the specified bucket"""
    # Check if we need to start a new thread for this bucket
    if bucket_name not in real_time_visualization_threads:
        stop_event = Event()  # Create a new stop event for this thread
        thread = threading.Thread(target=emit_realtime_data, args=(bucket_name, stop_event))
        thread.daemon = True
        thread.start()
        real_time_visualization_threads[bucket_name] = (thread, stop_event)
        print(f"Starting real-time visualization thread for bucket: {bucket_name}")
    return jsonify({'message': "Real-time visualization started"})


def emit_realtime_data(bucket_name, stop_event):
    global last_sent_data

    while not stop_event.is_set():
        try:
            recognition_data = fetch_latest_entry(bucket_name, "speaker recognition", influx_client)
            transcription_data = fetch_latest_entry(bucket_name, "speaker transcription", influx_client)
            relations = fetch_latest_entry(bucket_name, "badge relations", influx_client)
            graph_data, segment_time = relations if relations else (None, None)
            position_data = {}
            if segment_time:
                position_data = get_node_positions(bucket_name, influx_client, segment_time)

            data = {'recognition': None, 'transcription': None, 'graph': None, 'positions': None}
            recognition_new = False
            transcription_new = False
            graph_new = False

            if bucket_name not in last_sent_data:
                last_sent_data[bucket_name] = {}

            # Check for new recognition data based on segment_start_time
            if recognition_data and (bucket_name not in last_sent_data or recognition_data['segment_start_time']
                                     != last_sent_data[bucket_name].get('recognition_segment_start_time')):
                last_sent_data[bucket_name]['recognition_segment_start_time'] = recognition_data['segment_start_time']
                data['recognition'] = recognition_data
                recognition_new = True

            # Check for new transcription data based on chunk_start_time
            if transcription_data and (bucket_name not in last_sent_data or transcription_data['chunk_start_time']
                                       != last_sent_data[bucket_name].get('transcription_chunk_start_time')):
                last_sent_data[bucket_name]['transcription_chunk_start_time'] = transcription_data['chunk_start_time']
                data['transcription'] = transcription_data
                transcription_new = True

            # Update graph and position data if there's a new segment
            if segment_time and (bucket_name not in last_sent_data or segment_time != last_sent_data[bucket_name].get(
                    'graph_segment_time')):
                last_sent_data[bucket_name]['graph_segment_time'] = segment_time
                data['graph'] = graph_data
                data['positions'] = position_data
                graph_new = True

            if recognition_new or transcription_new or graph_new:
                socketio.emit('realtime_data', {'bucket': bucket_name, 'data': data})

        except Exception as e:
            print(f"An unexpected error occurred during data fetching: {e}")


@socketio.on('join_bucket')
def handle_join_bucket(data):
    bucket_name = data['bucket_name']
    client_id = data['client_id']  # Assuming client sends this
    print(f"Client {client_id} joined bucket: {bucket_name}")
    # Initialize the set for the bucket if it doesn't exist
    if bucket_name not in active_buckets:
        active_buckets[bucket_name] = set()
    # Add the client ID to the set for the bucket
    active_buckets[bucket_name].add(client_id)


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


# gunicorn -k gevent -w 1 -b 0.0.0.0:5000 app:app
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')
