import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import dlib
import os
import requests

# Ensure the detected_faces folder exists
DETECTED_FOLDER = "detected_faces"
if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

# Initialize dlib's face detector (HOG-based) for video-based tracking
detector = dlib.get_frontal_face_detector()

# Initialize OpenCV's Haar Cascade for image-based tracking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# API endpoints
STORE_IMAGE_API = "http://122.166.149.171:3000/api/storeimage"
REGISTER_IMAGE_API = "http://122.166.149.171:4000/register_face"

class VideoFaceDetectionApp:

    def __init__(self, root):
        self.root = root
        self.video_path = ""
        self.detected_faces = []

        # Clear the existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Set window size to 800x800
        self.root.geometry("800x800")

        # Create a frame to hold the buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Upload video button
        self.upload_button = tk.Button(button_frame, text="Upload Video", command=self.upload_video)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        # Button to view detected faces
        self.view_faces_button = tk.Button(button_frame, text="View Detected Faces", command=self.view_detected_faces)
        self.view_faces_button.pack(side=tk.LEFT, padx=5)

        # Register Image button
        self.register_button = tk.Button(button_frame, text="Register Image", command=self.open_register_window)
        self.register_button.pack(side=tk.LEFT, padx=5)

        # Canvas for displaying video, resized to 800x800
        self.canvas = tk.Canvas(root, width=800, height=800)
        self.canvas.pack()


    # Rest of the class remains unchanged

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

            # Resize the frame to fit within 800x800, keeping the aspect ratio
            frame_resized = self.resize_frame(frame, 800, 800)

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
        faces_window.geometry("800x800")

        for face_file in self.detected_faces:
            img = Image.open(face_file)
            imgtk = ImageTk.PhotoImage(image=img)
            label = tk.Label(faces_window, image=imgtk)
            label.image = imgtk
            label.pack()

    def open_register_window(self):
        """Opens a new window to upload an image and register it with a group name."""
        self.register_window = tk.Toplevel(self.root)
        self.register_window.title("Register Image")
        self.register_window.geometry("800x800")

        # Group Name input
        tk.Label(self.register_window, text="Group Name:").pack()
        self.group_name_entry = tk.Entry(self.register_window)
        self.group_name_entry.pack()

        # Image upload button
        self.upload_image_button = tk.Button(self.register_window, text="Select Image", command=self.select_image)
        self.upload_image_button.pack()

        # Label to display the selected image file name
        self.image_name_label = tk.Label(self.register_window, text="")
        self.image_name_label.pack()

        # Submit button
        self.submit_button = tk.Button(self.register_window, text="Upload and Register", command=self.upload_and_register_image)
        self.submit_button.pack()

    def select_image(self):
        """Open file dialog to select an image."""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            # Update the label with the selected file name
            self.image_name_label.config(text=os.path.basename(self.image_path))

    def upload_and_register_image(self):
        """Uploads the image to the Store Image API, then registers it using the Register Image API."""
        group_name = self.group_name_entry.get()

        if not self.image_path or not group_name:
            messagebox.showerror("Error", "Please select an image and enter a group name.")
            return

        # Upload to Store Image API
        files = {'image': open(self.image_path, 'rb')}
        data = {'type': 'beatuser', 'user': '650085038edf02001cc51ed1', 'company': '650084388edf02001cc517d7'}

        try:
            response = requests.post(STORE_IMAGE_API, files=files, data=data)
            if response.status_code == 200 and response.json().get("success"):
                image_url = response.json().get("url")

                # Register with Register Image API
                register_data = {"image_url": image_url, "group_name": group_name}
                register_response = requests.post(REGISTER_IMAGE_API, json=register_data)

                # Check if the request was successful and the status in the response is "success"
                if register_response.status_code == 200:
                    response_json = register_response.json()
                    if response_json.get("status") == "success":
                        messagebox.showinfo("Success", "Image registered successfully!")
                    else:
                        # Print error details to terminal and show messagebox error
                        print("Failed to register image:", response_json.get("message"))
                        messagebox.showerror("Error", "Failed to register image.")
                else:
                    # Print error details to terminal and show messagebox error
                    print("Failed to register image:", register_response.status_code, register_response.text)
                    messagebox.showerror("Error", "Failed to register image.")
            else:
                # Print error details to terminal and show messagebox error
                print("Failed to upload image:", response.status_code, response.text)
                messagebox.showerror("Error", "Failed to upload image.")
        except Exception as e:
            # Print exception details to terminal and show messagebox error
            print("Error during image upload/registration:", e)
            messagebox.showerror("Error", "An unexpected error occurred.")

def home_screen():
    root = tk.Tk()
    root.title("Face Tracking")
    root.geometry("800x800")

    # Create the header label
    header_label = tk.Label(root, text="Face Tracking", font=("Arial", 24))
    header_label.pack(pady=20)

    # Button for Video-based Face Tracking
    video_button = tk.Button(root, text="Video", font=("Arial", 18), width=10,
                             command=lambda: VideoFaceDetectionApp(root))
    video_button.pack(pady=10)

    root.mainloop()

# Run the home screen
home_screen()