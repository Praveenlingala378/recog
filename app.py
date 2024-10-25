from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    Response,
)
import cv2
import dlib
import os
import requests
import json

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["DETECTED_FOLDER"] = "detected_faces"

# Create upload and detected faces folders if they do not exist
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
if not os.path.exists(app.config["DETECTED_FOLDER"]):
    os.makedirs(app.config["DETECTED_FOLDER"])

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Global storage for URLs (consider replacing with a more persistent solution)
detected_face_urls = []


def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    store_image_url = "http://122.166.149.171:3000/api/storeImage"
    match_faces_url = "http://122.166.149.171:4000/match_faces"
    payload = {
        "type": "beatuser",
        "user": "650085038edf02001cc51ed1",
        "company": "650084388edf02001cc517d7",
    }
    global detected_face_urls  # Access global variable

    # try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for i, face in enumerate(faces):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Ensure coordinates are within frame boundaries
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the detected face image
            face_img = frame[y : y + h, x : x + w]
            face_filename = os.path.join(
                app.config["DETECTED_FOLDER"],
                f"{video_name}_frame_{frame_number}_face_{i}.jpg",
            )

            cv2.imwrite(face_filename, face_img)

            # Send the face image to the store image API
            # with open(face_filename, "rb") as img_file:
            #     files = [("image", (face_filename, img_file, "image/jpeg"))]
            #     response = requests.post(
            #         store_image_url, headers={}, data=payload, files=files
            #     )
            #     if response.status_code == 200:
            #         face_url = response.json().get("url")
            #         if face_url:
            #             detected_face_urls.append(face_url)  # Store the URL
            #             print("Current URLs:", detected_face_urls)

            #             # Call the face match API
            #             match_payload = json.dumps(
            #                 {"image_url": face_url, "group_name": "Rezlers"}
            #             )
            #             match_headers = {"Content-Type": "application/json"}
            #             match_response = requests.post(
            #                 match_faces_url,
            #                 headers=match_headers,
            #                 data=match_payload,
            #             )
            #             print("Face Match Response:", match_response.text)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        frame_number += 1
    # finally:
    #     cap.release()


# Updated detected_faces route to use face match API
@app.route("/detected_faces")
def detected_faces():
    faces = os.listdir(app.config["DETECTED_FOLDER"])  # Detected faces in local storage
    matched_faces = []

    # Fetch match results for each detected face URL
    for url in detected_face_urls:  # Use the global list of detected face URLs
        match_payload = json.dumps({"image_url": url, "group_name": "Rezlers"})
        match_headers = {"Content-Type": "application/json"}
        match_response = requests.post(
            "http://122.166.149.171:4000/match_faces",
            headers=match_headers,
            data=match_payload,
        )
        match_result = match_response.json()

        # If matches are found, retrieve registered face image URLs from the match result
        if match_result["status"] == "success" and "matches" in match_result["body"]:
            matches = match_result["body"]["matches"]
            for match in matches:
                registered_image = match["url"]  # URL to registered image
                match_score = match["score"]

                # Append both detected and registered image URLs to matched_faces
                registered_image = registered_image.replace(
                    "localhost", "122.166.149.171"
                )
                # registered_image.replace("localhost", "122.166.149.171")
                # print(registered_image)
                matched_faces.append(
                    {
                        "detected_image": url,  # URL to detected image
                        "registered_image": registered_image,  # URL to registered image
                        "score": match_score,
                    }
                )

    # Pass both detected faces and matched faces to the template
    return render_template(
        "detected_faces.html", faces=faces, matched_faces=matched_faces
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        return redirect(url_for("play_video", filename=file.filename))


@app.route("/play_video/<filename>")
def play_video(filename):
    return render_template("play_video.html", filename=filename)


@app.route("/video_feed/<filename>")
def video_feed(filename):
    return Response(
        generate_frames(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/detected_faces/<filename>")
def send_face(filename):
    return send_from_directory(app.config["DETECTED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
