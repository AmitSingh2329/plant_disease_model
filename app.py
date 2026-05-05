from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import json
import os
import tflite_runtime.interpreter as tflite   # ✅ CORRECT

app = Flask(__name__)
CORS(app)

# ✅ Paths
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'class_indices.json'

# ✅ Load TFLite model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Load labels
with open(LABELS_PATH) as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}


# ✅ Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype('float32')


# ✅ Health check
@app.route('/')
def home():
    return "ML API is running 🚀"


# ✅ Health check (used by frontend warm-up)
@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        img = Image.open(file).convert('RGB')
        processed = preprocess_image(img)

        # ✅ TFLite inference
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # 🔥 IMPORTANT: Confidence threshold
        if confidence < 0.9:
            return jsonify({
                'error': '⚠️ Please upload a valid plant leaf image',
                'confidence': round(confidence, 4)
            }), 200

        disease_name = labels.get(predicted_class, "Unknown Disease")
        disease_name = disease_name.replace("___", " - ").replace("_", " ")

        return jsonify({
            'disease': disease_name,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)