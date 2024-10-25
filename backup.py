# #codebackup
#     #from flask import (
#         Flask,
#         render_template,
#         request,
#         redirect,
#         url_for,
#         send_from_directory,
#         Response,
#     )
#     import cv2
#     import dlib
#     import os
#     import requests

#     app = Flask(__name__)
#     app.config["UPLOAD_FOLDER"] = "uploads"
#     app.config["DETECTED_FOLDER"] = "detected_faces"

#     # Create upload and detected faces folders if they do not exist
#     if not os.path.exists(app.config["UPLOAD_FOLDER"]):
#         os.makedirs(app.config["UPLOAD_FOLDER"])
#     if not os.path.exists(app.config["DETECTED_FOLDER"]):
#         os.makedirs(app.config["DETECTED_FOLDER"])

#     # Initialize dlib's face detector (HOG-based)
#     detector = dlib.get_frontal_face_detector()


#     def generate_frames(video_path):
#         cap = cv2.VideoCapture(video_path)
#         frame_number = 0
#         video_name = os.path.splitext(os.path.basename(video_path))[0]

#         url = "http://122.166.149.171:3000/api/storeImage"
#         payload = {
#             "type": "beatuser",
#             "user": "650085038edf02001cc51ed1",
#             "company": "650084388edf02001cc517d7",
#         }
#         urls = []

#         try:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 faces = detector(gray)
#                 for i, face in enumerate(faces):
#                     x, y, w, h = face.left(), face.top(), face.width(), face.height()
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     # Save the detected face image
#                     face_img = frame[y : y + h, x : x + w]
#                     face_filename = os.path.join(
#                         app.config["DETECTED_FOLDER"],
#                         f"{video_name}_frame_{frame_number}_face_{i}.jpg",
#                     )
#                     cv2.imwrite(face_filename, face_img)

#                     # Send the face image to the API
#                     with open(face_filename, "rb") as img_file:
#                         files = [("image", (face_filename, img_file, "image/jpeg"))]
#                         response = requests.post(url, headers={}, data=payload, files=files)
#                         if response.status_code == 200:
#                             face_url = response.json().get("url")
#                             if face_url:
#                                 urls.append(face_url)
#                                 print("Current URLs:", urls)  # Print URLs incrementally

#                 ret, buffer = cv2.imencode(".jpg", frame)
#                 frame = buffer.tobytes()
#                 yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

#                 frame_number += 1
#         finally:
#             cap.release()
#             print(
#                 "===================Final URLs:=================", urls
#             )  # Ensure to print whatever URLs have been captured


#     @app.route("/")
#     def index():
#         return render_template("index.html")


#     @app.route("/upload", methods=["POST"])
#     def upload_file():
#         if "file" not in request.files:
#             return redirect(request.url)

#         file = request.files["file"]
#         if file.filename == "":
#             return redirect(request.url)

#         if file:
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#             file.save(file_path)
#             return redirect(url_for("play_video", filename=file.filename))


#     @app.route("/play_video/<filename>")
#     def play_video(filename):
#         return render_template("play_video.html", filename=filename)


#     @app.route("/video_feed/<filename>")
#     def video_feed(filename):
#         return Response(
#             generate_frames(os.path.join(app.config["UPLOAD_FOLDER"], filename)),
#             mimetype="multipart/x-mixed-replace; boundary=frame",
#         )


#     @app.route("/uploads/<filename>")
#     def uploaded_file(filename):
#         return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


#     @app.route("/detected_faces")
#     def detected_faces():
#         faces = os.listdir(app.config["DETECTED_FOLDER"])
#         return render_template("detected_faces.html", faces=faces)


#     @app.route("/detected_faces/<filename>")
#     def send_face(filename):
#         return send_from_directory(app.config["DETECTED_FOLDER"], filename)


#     if __name__ == "__main__":
#         app.run(debug=True)
