from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model.predict(source="bus.jpg", show=True)
