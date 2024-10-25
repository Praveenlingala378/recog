from flask import (
    Flask,
    render_template,
    Response,
    request,
    redirect,
    url_for,
    send_from_directory,
    abort,
    flash,
    jsonify,
)


import face_recognition
import asyncio
import aiohttp
from imouapi.api import ImouAPIClient
from imouapi.device import ImouDiscoverService, ImouDevice
import numpy as np
from sklearn.cluster import DBSCAN
from face_recognition import load_image_file, face_encodings
import requests
import json
import cv2
import os
import time
import threading
import queue
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

app = Flask(__name__)

load_dotenv()

# Replace with your actual appId and appSecret
APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")


# Function to get face embeddings using face_recognition library
def get_face_embeddings(image_path):
    try:
        img = load_image_file(image_path)
        encoding = face_encodings(img)
        if encoding:
            return encoding[0]
    except Exception as e:
        print(f"Error getting embedding for {image_path}: {e}")
    return None


# Function to cluster face images
def cluster_faces(image_paths):
    embeddings = [get_face_embeddings(image) for image in image_paths]
    embeddings = np.array(
        [e for e in embeddings if e is not None]
    )  # Filter valid embeddings
    if embeddings.size == 0:
        return {}  # Return empty if no valid embeddings
    # Apply DBSCAN clustering
    clustering_model = DBSCAN(eps=0.6, min_samples=2).fit(embeddings)
    labels = clustering_model.labels_
    # Grouping images by clusters
    clustered_faces = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # Noise, can be ignored or treated separately
        if label not in clustered_faces:
            clustered_faces[label] = []
        clustered_faces[label].append(image_paths[idx])
    return clustered_faces


class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


async def discover_and_control_ptz(h, v, z):
    async with aiohttp.ClientSession() as session:
        api_client = ImouAPIClient(APP_ID, APP_SECRET, session)
        discover_service = ImouDiscoverService(api_client)

        # Discover devices
        discovered_devices = await discover_service.async_discover_devices()
        if (
            discovered_devices
            and "imou" in discovered_devices
            and isinstance(discovered_devices["imou"], ImouDevice)
        ):
            device = discovered_devices["imou"]
            device_id = device.get_device_id()

            # Control PTZ
            await api_client.async_api_controlLocationPTZ(device_id, h, v, z)
            return {"device_id": device_id, "status": "success"}
        else:
            return {"status": "error", "message": "No device found or invalid format"}


# Set a secret key for session handling
app.secret_key = "123456789"

# Directory where captured faces are stored
captured_faces_dir = "captured_faces"
snapshot_dir = "static\\SNAPSHOT"

# Ensure the directories exist
os.makedirs(captured_faces_dir, exist_ok=True)
os.makedirs(snapshot_dir, exist_ok=True)

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Video stream variable and control
rtsp_url = None
video_stream = None
video_stream2 = None
streaming_paused = False
stream_lock = threading.Lock()


def process_stream():
    global video_stream
    if video_stream:
        start_time = time.time()
        frame = video_stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = frame[y : y + h, x : x + w]
            filename = f"face_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(captured_faces_dir, filename), face)
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        print(time.time() - start_time)


def generate_frames():
    global video_stream, streaming_paused
    while True:
        with stream_lock:
            if streaming_paused:
                time.sleep(0.01)
                continue

        frame = video_stream.read()
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global video_stream, rtsp_url
        rtsp_url = request.form["rtsp_url"]
        video_stream = VideoCapture(rtsp_url)
        return redirect(url_for("stream"))
    return render_template("index.html")


@app.route("/stream")
def stream():
    images = os.listdir(captured_faces_dir)
    return render_template("stream.html", images=images)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/captured_images")
def captured_images():
    try:
        clusters = (
            process_images()
        )  # Call the clustering logic or load pre-clustered data
        # Assume clusters.json returns a valid JSON object
        return render_template(
            "captured_images.html", images=clusters
        )  # Pass clusters to template
    except Exception as e:
        print(f"Error in /captured_images: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/captured_faces/<filename>")
def get_captured_face(filename):
    try:
        return send_from_directory(captured_faces_dir, filename)
    except FileNotFoundError:
        abort(404)


@app.route("/capture_snapshot", methods=["POST"])
def capture_snapshot():
    global video_stream

    success, frame = video_stream.read()
    if success:
        filename = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(snapshot_dir, filename), frame)
        flash(f"Snapshot saved successfully as {filename} in the SNAPSHOT folder!")
    else:
        flash("Failed to capture snapshot.")

    return redirect(url_for("stream"))


@app.route("/delete_faces", methods=["POST"])
def delete_faces():
    for filename in os.listdir(captured_faces_dir):
        file_path = os.path.join(captured_faces_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return redirect(url_for("stream"))


@app.route("/pause_stream", methods=["POST"])
def pause_stream():
    global streaming_paused
    with stream_lock:
        streaming_paused = True
    return redirect(url_for("stream"))


@app.route("/play_stream", methods=["POST"])
def play_stream():
    global streaming_paused
    with stream_lock:
        streaming_paused = False
    return redirect(url_for("stream"))


@app.route("/snapshot_images")
def snapshot_images():
    snapshots = [
        snap
        for snap in os.listdir(snapshot_dir)
        if snap.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]
    return render_template("snapshot_images.html", images=snapshots)


@app.route("/api/controlLocationPTZ", methods=["POST"])
def control_location_ptz():
    data = request.get_json()
    h = data.get("h")
    v = data.get("v")
    z = data.get("z")

    # Run PTZ control in asyncio loop
    try:
        result = asyncio.run(discover_and_control_ptz(h, v, z))
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/next_page")
def next_page():
    return render_template("next_page.html")


@app.route("/create_group", methods=["POST"])
def create_group():
    data = request.get_json()
    group_name = data.get("name")
    description = data.get("description", "na")

    if not group_name:
        return jsonify({"error": "Group name is required"}), 400

    url = "http://122.166.149.171:4000/create_group"
    payload = json.dumps({"name": group_name, "description": description})
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        response_data = response.json()
        return jsonify(
            {"status": response_data.get("status"), "body": response_data.get("body")}
        )
    else:
        return jsonify({"error": "Failed to create group"}), response.status_code


@app.route("/list_groups", methods=["GET"])
def list_groups():
    url = "http://122.166.149.171:4000/list_group"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200 and data.get("body"):
            return jsonify({"status": "success", "body": data["body"]})
        else:
            return jsonify({"status": "failed", "message": "No groups found"})
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Error fetching group list: {str(e)}"}
        )


# @app.route("/trigger_face_match", methods=["POST"])
# def trigger_face_match():
#     directory = "captured_faces"  # Set your directory path here
#     url = "http://122.166.149.171:4000/api/storeImage"
#     payload = {
#         "type": "beatuser",
#         "user": "650085038edf02001cc51ed1",
#         "company": "650084388edf02001cc517d7",
#     }

#     urls = []
#     try:
#         for file_name in os.listdir(directory):
#             if file_name.endswith(".jpg"):
#                 path = os.path.join(directory, file_name)
#                 files = [("image", (file_name, open(path, "rb"), "image/jpeg"))]
#                 headers = {}
#                 response = requests.post(
#                     url, headers=headers, data=payload, files=files
#                 )
#                 response_data = response.json()
#                 if response.ok and "url" in response_data:
#                     urls.append(response_data["url"])
#                     print(response_data["url"])  # Print each URL to the terminal
#     except Exception as e:
#         print(f"Error in trigger_face_match: {e}")
#         return jsonify({"error": str(e)}), 500

#     return jsonify({"urls": urls})


# New code starts here


# Function to upload images and store URLs
def store_images(directory, url, payload):
    urls = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            path = os.path.join(directory, file_name)
            files = [("image", (file_name, open(path, "rb"), "image/jpeg"))]
            response = requests.post(url, data=payload, files=files)
            image_url = response.json().get("url")
            if image_url:
                urls.append(image_url)
                print(f"Image URL: {image_url}")
            else:
                print(f"Failed to upload {file_name}")
    return urls


# Function to match faces using the URLs
def match_faces(urls, match_url):
    results = []
    for image_url in urls:
        payload = json.dumps({"image_url": image_url, "group_name": "testface"})
        headers = {"Content-Type": "application/json"}
        response = requests.post(match_url, headers=headers, data=payload)
        results.append({"image_url": image_url, "match_result": response.json()})
        print(f"Match Result: {response.text}")
    return results



#------------------------------------------------------
UPLOAD_FOLDER = 'captured_faces'
CLUSTER_FOLDER = 'clustered_faces'
os.makedirs(CLUSTER_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(img_path)
        images.append(image)
        filenames.append(filename)
    return images, filenames

def get_face_encodings(images):
    encodings = []
    for img in images:
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            encodings.append(enc[0])
    return np.array(encodings)

def perform_clustering(face_encodings, eps=0.5, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(face_encodings)
    return clustering

def save_cluster_first_images(images, filenames, labels):
    unique_labels = set(labels)
    saved_filenames = set()

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        cluster_images = [images[i] for i in range(len(labels)) if labels[i] == label]
        cluster_filenames = [filenames[i] for i in range(len(labels)) if labels[i] == label]

        # Save the first image of the cluster
        if cluster_filenames:
            first_image = cluster_images[0]
            first_filename = secure_filename(cluster_filenames[0])
            output_path = os.path.join(CLUSTER_FOLDER, first_filename)
            face_recognition.save_image_file(output_path, first_image)
            saved_filenames.add(first_filename)

@app.route("/process_images", methods=["POST"])
def process_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')

    # Save uploaded images to the folder
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))

    # Load images
    images, filenames = load_images_from_folder(UPLOAD_FOLDER)

    # Get face encodings
    face_encodings = get_face_encodings(images)

    if len(face_encodings) == 0:
        return jsonify({'error': 'No faces found in the uploaded images'}), 400

    # Perform clustering
    clustering_model = perform_clustering(face_encodings)

    # Save first images of each cluster
    save_cluster_first_images(images, filenames, clustering_model.labels_)

    return jsonify({'message': 'Images processed and clustered successfully.'}), 200

#------------------------------------------------------


# API endpoint to trigger the process and return the results
@app.route("/process_imagesv2", methods=["POST"])
def process_images():
    # Extract directory path from the request
    data = request.json
    directory = data.get("directory", r"captured_faces")
    # Get the list of images from the directory
    image_paths = [
        os.path.join(directory, img)
        for img in os.listdir(directory)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    # Cluster the images
    clustered_faces = cluster_faces(image_paths)
    result = []
    for group, images in clustered_faces.items():
        representative_image = images[
            0
        ]  # Select the first image in each cluster as representative
        result.append(
            {
                "image_url": url_for(
                    "get_captured_face", filename=os.path.basename(representative_image)
                ),
                "group_images": [
                    os.path.basename(img) for img in images
                ],  # Store only the filenames for simplicity
            }
        )
    return jsonify(result)


# Serve the HTML page
@app.route("/new_index")
def new_index():
    return render_template("index.html")


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(process_stream, "interval", seconds=1)
    scheduler.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
