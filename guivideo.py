import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import os

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Ensure the detected_faces folder exists
DETECTED_FOLDER = "detected_faces"
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection App")
        self.video_path = ""
        self.detected_faces = []

        # Create GUI elements
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack()

        self.canvas = tk.Canvas(root, width=600, height=400)
        self.canvas.pack()

        self.view_faces_button = tk.Button(root, text="View Detected Faces", command=self.view_detected_faces)
        self.view_faces_button.pack()

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if self.video_path:
            self.play_video()

    def play_video(self):
        cap = cv2.VideoCapture(self.video_path)
        self.detected_faces = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for i, face in enumerate(faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                # Ensure coordinates are within frame boundaries
                if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the detected face image
                    face_img = frame[y:y + h, x:x + w]
                    face_filename = os.path.join(DETECTED_FOLDER, f"detected_face_{i}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    self.detected_faces.append(face_filename)

            # Convert frame to ImageTk format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.update()

        cap.release()

    def view_detected_faces(self):
        if not self.detected_faces:
            messagebox.showinfo("Info", "No faces detected.")
            return

        faces_window = tk.Toplevel(self.root)
        faces_window.title("Detected Faces")

        for face_file in self.detected_faces:
            img = Image.open(face_file)
            imgtk = ImageTk.PhotoImage(image=img)
            label = tk.Label(faces_window, image=imgtk)
            label.image = imgtk
            label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()