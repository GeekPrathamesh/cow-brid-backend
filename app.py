from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
import json

app = Flask(__name__)
CORS(app)

MODEL_PATH = "cow_model.h5"
CLASSES_JSON = "classes.json"
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_JSON, "r") as f:
    class_names = json.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return "No files uploaded", 400

    files = request.files.getlist("files")
    results = []

    for idx, file in enumerate(files):
        if file.filename == "":
            continue

        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        breed = class_names[class_index]

        results.append({"index": idx, "filename": file.filename, "breed": breed})
        os.remove(filepath)

    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
