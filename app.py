from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# ✅ Load model (SavedModel folder OR .h5 both work)
model = tf.keras.models.load_model('plant_disease.h5', compile=False)# ✅ Load class labels
with open('class_indices.json') as f:
    class_indices = json.load(f)

# Reverse mapping: index → label
labels = {v: k for k, v in class_indices.items()}


# ✅ Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))   # ⚠️ change if your model used different size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ✅ Health check route
@app.route('/')
def home():
    return "ML API is running 🚀"


# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Read image
        img = Image.open(file).convert('RGB')

        # Preprocess
        processed = preprocess_image(img)

        # Predict
        prediction = model.predict(processed)

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        disease_name = labels.get(predicted_class, "Unknown Disease")

        # ✅ IMPORTANT: match Node.js expected format
        return jsonify({
            'disease': disease_name,
            'confidence': confidence
        })

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return jsonify({
            'error': 'Prediction failed'
        }), 500


# ✅ Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)