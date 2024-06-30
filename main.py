from flask import Flask, request, jsonify
import tensorflow as tf
from threading import Thread
import os
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Specify the correct paths to your model files
model_path_mlo = './models/Model_Binary_MLO.h5'
model_path_cc = './models/Model_Binary_CC.h5'

# Check if the model files exist
if not os.path.exists(model_path_mlo):
    raise FileNotFoundError(f"Model file does not exist at: {model_path_mlo}")
if not os.path.exists(model_path_cc):
    raise FileNotFoundError(f"Model file does not exist at: {model_path_cc}")

loaded_model_mlo = tf.keras.models.load_model(model_path_mlo)
loaded_model_cc = tf.keras.models.load_model(model_path_cc)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'view_type' not in request.form or 'image' not in request.files:
            return jsonify({'error': 'Missing view_type or image in the request'}), 400

        view_type = request.form['view_type']
        image_file = request.files['image']
        image_bytes = image_file.read()

        preprocessed_input = preprocess_image(image_bytes)

        if view_type == 'mlo':
            prediction = loaded_model_mlo(preprocessed_input)
        elif view_type == 'cc':
            prediction = loaded_model_cc(preprocessed_input)
        else:
            return jsonify({'error': 'Invalid view type specified'}), 400
        
        threshold = 0.5
        probability = float(prediction.numpy()[0][0])
        probability = round(probability, 2)
        predicted_class = 1 if probability >= threshold else 0
        predicted_class = int(tf.argmax(prediction, axis=1).numpy()[0])
        class_labels = ['Normal', 'Pathologique']
        predicted_class_label = class_labels[predicted_class]
        print("Predicted Class:", predicted_class_label)
        
        return jsonify(predicted_class=predicted_class, predicted_class_label=predicted_class_label, probability=probability)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask_app():
    app.run(port=5500)

if __name__ == '__main__':
    try:
        t = Thread(target=run_flask_app)
        t.start()
        try:
            while True:
                pass
        except KeyboardInterrupt:
            t.join()
            print("Server is down")
    except Exception as e:
        print('Error:', str(e))
