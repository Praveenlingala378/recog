import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import os

# Ensure the detected_faces folder exists
DETECTED_FOLDER = "detected_faces"
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

# Initialize dlib's face detector (HOG-based) for video-based tracking
detector = dlib.get_frontal_face_detector()

# Initialize OpenCV's Haar Cascade for image-based tracking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ========================= VIDEO BASED FACE TRACKING =========================
class VideoFaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.video_path = ""
        self.detected_faces = []
        
        # Upload video button
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video)
        self.upload_button.pack()
        
        # Canvas for displaying video, resized to 500x500
        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()
        
        # Button to view detected faces
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
                if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the detected face image
                    face_img = frame[y:y + h, x:x + w]
                    face_filename = os.path.join(DETECTED_FOLDER, f"detected_face_video_{i}.jpg")
                    cv2.imwrite(face_filename, face_img)
                    self.detected_faces.append(face_filename)

            # Resize the frame to fit within 500x500, keeping the aspect ratio
            frame_resized = self.resize_frame(frame, 500, 500)
            
            # Convert frame to ImageTk format for Tkinter
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.update()

        cap.release()

    def resize_frame(self, frame, target_width, target_height):
        """Resizes the frame to fit within the specified target width and height while keeping the aspect ratio."""
        height, width, _ = frame.shape
        aspect_ratio = width / height
        
        if width > height:
            # Fit to width
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Fit to height
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        return cv2.resize(frame, (new_width, new_height))

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

# ========================= IMAGE BASED FACE TRACKING =========================
def image_based_face_tracking(root):
    # Image upload function
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        detect_faces(file_path)

    def detect_faces(file_path):
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Failed to open image!")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_count = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = img[y:y + h, x:x + w]
            face_filename = os.path.join(DETECTED_FOLDER, f"detected_face_image_{face_count}.jpg")
            cv2.imwrite(face_filename, face)
            face_count += 1

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((300, 300))  # Resize while preserving aspect ratio
        img_tk = ImageTk.PhotoImage(img_pil)

        display_image(img_tk)

    def display_image(img_tk):
        top = tk.Toplevel(root)
        top.title("Image-Based Face Tracking")
        canvas = tk.Canvas(top, width=300, height=300, bg="gray")
        canvas.pack(pady=20)
        canvas.create_image(150, 150, anchor="center", image=img_tk)
        canvas.image = img_tk

    # Set up the window for image face tracking
    top = tk.Toplevel(root)
    top.title("Image-Based Face Tracking")

    canvas = tk.Canvas(top, width=300, height=300, bg="gray")
    canvas.pack(pady=20)

    upload_button = tk.Button(top, text="Upload Image", command=upload_image)
    upload_button.pack()


# ========================= HOME SCREEN =========================
def home_screen():
    root = tk.Tk()
    root.title("Face Tracking")

    # Create the header label
    header_label = tk.Label(root, text="Face Tracking", font=("Arial", 24))
    header_label.pack(pady=20)

    # Button for Image-based Face Tracking
    image_button = tk.Button(root, text="Image", font=("Arial", 18), width=10,
                             command=lambda: image_based_face_tracking(root))
    image_button.pack(pady=10)

    # Button for Video-based Face Tracking
    video_button = tk.Button(root, text="Video", font=("Arial", 18), width=10,
                             command=lambda: VideoFaceDetectionApp(tk.Toplevel(root)))
    video_button.pack(pady=10)

    root.mainloop()


# Run the home screen
home_screen()
