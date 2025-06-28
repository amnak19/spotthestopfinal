from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import traceback
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# ✅ Load the YOLO model once globally (better for Render memory and performance)
try:
    model = YOLO('best.pt')  # Replace with your correct model path if needed
except Exception as e:
    print("🔥 Failed to load YOLO model at startup:")
    traceback.print_exc()

# Health check route
@app.route('/')
def home():
    return "✅ SpotTheStop API is Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        # Read and preprocess the image
        img = Image.open(file.stream).convert('RGB')

        # Run YOLO detection
        results = model(img)

        # Format results
        detections = []
        for box in results[0].boxes:
            det = {
                "box": box.xyxy[0].tolist(),       # [x1, y1, x2, y2]
                "confidence": float(box.conf[0]),  # e.g., 0.98
                "class": model.names[int(box.cls[0])]  # e.g., 'Red light'
            }
            detections.append(det)

        return jsonify({"detections": detections})

    except Exception as e:
        print("🔥 Exception in /predict:")
        traceback.print_exc()  # ✅ Show full error in Render logs
        return jsonify({"error": str(e)}), 500

# App entry point for local/dev (Render will ignore this if using Gunicorn)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default to 10000
    app.run(host='0.0.0.0', port=port, debug=False)
