from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

@app.route('/')
def home():
    return "Color Blind Detection Server is Running"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    results = model(img)
    detections = results[0].boxes.data.tolist()

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True)
