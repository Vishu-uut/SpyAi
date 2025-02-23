import os
import time
from flask import Flask, jsonify, send_from_directory, request
from flask_socketio import SocketIO
from flask_cors import CORS
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the folder where videos are stored
UPLOAD_FOLDER = "videos"

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend (React) to access backend
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket support

# Ensure the videos directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to get the list of videos
def get_video_list():
    return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

# API route to get the list of videos
@app.route('/videos', methods=['GET'])
def list_videos():
    return jsonify(get_video_list())

# API route to serve video files
@app.route('/videos/<filename>')
def get_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# WebSocket event handler for new videos
class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(('.mp4', '.avi', '.mov')):
            time.sleep(1)  # Wait to ensure file is fully written
            filename = os.path.basename(event.src_path)
            print(f"New video detected: {filename}")
            socketio.emit('new_video', {'filename': filename})

# Start watching the directory for new files
observer = Observer()
observer.schedule(VideoHandler(), path=UPLOAD_FOLDER, recursive=False)
observer.start()

# API route to handle "Alert" and "Deny" actions
@app.route('/action/<action>/<filename>', methods=['POST'])
def video_action(action, filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if action not in ["alert", "deny"]:
        return jsonify({"error": "Invalid action"}), 400

    if action == "deny":
        if os.path.exists(filepath):
            os.remove(filepath)  # Delete the file
            print(f"Deleted: {filename}")
            socketio.emit('remove_video', {'filename': filename})  # Notify frontend
            return jsonify({"message": f"Deleted {filename}"}), 200
        else:
            return jsonify({"error": "File not found"}), 404

    print(f"User action: {action.upper()} on {filename}")
    return jsonify({"message": f"{action.capitalize()} action received for {filename}"}), 200

# Run the Flask app
if __name__ == '__main__':
    print("Starting Flask server...")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
