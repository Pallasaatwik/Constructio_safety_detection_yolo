from flask import Flask, request, jsonify, send_file
from io import BytesIO
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load YOLO model
model = YOLO("Model/ppe.pt")  # Replace with your custom model file if needed

# Helper function to draw bounding boxes
def draw_text_with_background(frame, text, position, font_scale=0.4, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.7, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    overlay = frame.copy()
    x, y = position
    cv2.rectangle(overlay, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

# Route to upload and process the image
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     # Read image from the uploaded file
#     img = Image.open(file.stream)
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
#     # Perform YOLO inference
#     results = model(img)

#     # Colors for bounding boxes
#     colors = [
#         (255, 0, 0),  # Hardhat (Blue)
#         (0, 255, 0),  # Mask (Green)
#         (0, 0, 255),  # NO-Hardhat (Red)
#         (255, 255, 0),  # NO-Mask (Cyan)
#         (255, 0, 255),  # NO-Safety Vest (Magenta)
#         (0, 255, 255),  # Person (Yellow)
#     ]
    
#     # Draw bounding boxes and labels
#     for result in results:
#         if result.boxes is not None:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                 confidence = box.conf[0]  # Confidence score
#                 cls = int(box.cls[0])  # Class ID
#                 label = f"{model.names[cls]} ({confidence:.2f})"
#                 color = colors[cls % len(colors)]  # Get color for bounding box

#                 # Draw the bounding box
#                 cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#                 draw_text_with_background(img, label, (x1, y1 - 10), font_scale=0.4, color=(255, 255, 255), bg_color=color, alpha=0.8, padding=4)

#     # Convert image back to PIL format for easy sending
#     img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#     # Convert to BytesIO to send as a response
#     img_io = BytesIO()
#     img_pil.save(img_io, 'JPEG')
#     img_io.seek(0)

#     return send_file(img_io, mimetype='image/jpeg')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print("❌ No file received!")  # Debug log
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("❌ Empty filename!")  # Debug log
        return jsonify({"error": "No selected file"}), 400
    
    print(f"✅ Received file: {file.filename}")  # Debug log

    try:
        img = Image.open(file.stream)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print("✅ Image successfully loaded and converted!")  # Debug log
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")  # Debug log
        return jsonify({"error": "Failed to load image"}), 400

    # Perform YOLO inference
    results = model(img)

    if len(results) == 0:
        print("❌ No detections found!")  # Debug log
        return jsonify({"error": "No objects detected"}), 400
    
    print(f"✅ Detected {len(results[0].boxes)} objects!")  # Debug log

    # Convert image to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')
if __name__ == '__main__':
    app.run(debug=True)
