from flask import Flask, request, jsonify
from PIL import Image
import io
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Color Blind Detection Server is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lazy import and model load
        from ultralytics import YOLO
        model = YOLO('best.pt')  # Only loads when this route is hit

        # Check if image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        # Read image
        img = Image.open(file.stream).convert('RGB')

        # Run YOLO detection
        results = model(img)

        # Extract and simplify detection data
        detections = []
        for box in results[0].boxes:
            det = {
                "box": box.xyxy[0].tolist(),       # [x1, y1, x2, y2]
                "confidence": float(box.conf[0]),  # e.g., 0.98
                "class_id": int(box.cls[0])        # e.g., 0
            }
            detections.append(det)

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port, debug=False)
