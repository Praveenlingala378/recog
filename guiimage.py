import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

# Initialize the face detector (using Haar Cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a folder to save detected faces if it doesn't exist
detected_faces_dir = "detected_faces"
if not os.path.exists(detected_faces_dir):
    os.makedirs(detected_faces_dir)

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    
    # Load and display the image with face tracking
    detect_faces(file_path)

def detect_faces(file_path):
    # Read the image
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Failed to open image!")
        return
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_count = 0
    # Draw a green rectangle around each detected face and save the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the face region
        face = img[y:y+h, x:x+w]
        
        # Save the detected face in the 'detected_faces' folder
        face_filename = os.path.join(detected_faces_dir, f"detected_face_{face_count}.jpg")
        cv2.imwrite(face_filename, face)
        face_count += 1
    
    # Resize the image to fit within a 300x300 box, preserving the aspect ratio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((300, 300))  # Resize while preserving aspect ratio
    img_tk = ImageTk.PhotoImage(img_pil)
    
    # Display the image in the GUI
    display_image(img_tk)

def display_image(img_tk):
    # Clear the existing image
    canvas.delete("all")
    
    # Display the new image on the canvas, centered in the 300x300 box
    canvas.create_image(150, 150, anchor="center", image=img_tk)
    canvas.image = img_tk  # Store a reference to prevent garbage collection

# Set up the GUI
root = tk.Tk()
root.title("Face Tracker")
root.geometry("400x400")

# Create a canvas to display the 300x300 image box
canvas = tk.Canvas(root, width=300, height=300, bg="gray")
canvas.pack(pady=20)

# Add an upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Run the application
root.mainloop()
