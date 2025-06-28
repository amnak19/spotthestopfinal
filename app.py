from flask import Flask, request, jsonify
from PIL import Image
import os

app = Flask(__name__)

# Load model once at startup (outside route)
from ultralytics import YOLO
model = YOLO('best.pt')

@app.route('/')
def home():
    return "Color Blind Detection Server is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        results = model(img)

        detections = []
        for box in results[0].boxes:
            det = {
                "box": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            }
            detections.append(det)

        return jsonify({"detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
