from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import json
import os
import tensorflow as tf

app = Flask(__name__)

# ✅ Paths
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'class_indices.json'

# ✅ Load TFLite model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
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


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        img = Image.open(file).convert('RGB')
        processed = preprocess_image(img)

        # ✅ TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

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