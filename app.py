import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from io import BytesIO
import os
from PIL import Image

MODEL = tf.keras.models.load_model(r'C:\Users\AUDRIK\DSC_ANA\weather_prediction\models\1')

app = Flask(__name__)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

CLASS_NAMES = [['Dew',
 'Fogsmog',
 'Frost',
 'Glaze',
 'Hail',
 'Lightning',
 'Rain',
 'Rainbow',
 'Rime',
 'Sandstorm',
 'Snow']]

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = read_file_as_image(file.read())
            img_batch = np.expand_dims(image, 0)
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template("index.html", prediction=None, confidence=None)

if __name__ == "__main__":
    app.run()
