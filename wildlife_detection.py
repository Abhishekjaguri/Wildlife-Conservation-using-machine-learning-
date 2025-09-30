import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Load the pre-trained YOLO model (YOLOv8)
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano version (lightweight)

def detect_animals_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    results = model(image)
    
    # Plot results on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} ({conf:.2f})"
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show image
    output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Processed image saved as {output_path}")


def detect_animals_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                label = f"{model.names[cls]} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Animal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        detect_animals_image(file_path)

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        detect_animals_video(file_path)

# Create UI
root = tk.Tk()
root.title("Wildlife Animal Detection")
root.geometry("300x150")

tk.Label(root, text="Upload Image or Video for Detection").pack(pady=10)
tk.Button(root, text="Upload Image", command=upload_image).pack(pady=5)
tk.Button(root, text="Upload Video", command=upload_video).pack(pady=5)
tk.Button(root, text="Exit", command=root.quit).pack(pady=5)

cv2.imshow('Window Name', image)
cv2.waitKey(0)
cv2.destroyAllWindows()