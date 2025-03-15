import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO("Model/ppe.pt")  # Ensure this path is correct

# Load an image (replace with an actual image path)
image_path = r"C:\Users\saatw\Downloads\cat.jpeg"  # Replace with a real image path
img = Image.open(image_path)
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Perform YOLO inference
results = model(img)

# Check detections
for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"✅ Detected {len(result.boxes)} objects!")
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} ({conf:.2f})"
            print(f"  - {label} at [{x1}, {y1}, {x2}, {y2}]")
    else:
        print("❌ No objects detected!")
