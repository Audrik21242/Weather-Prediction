import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from io import BytesIO
from PIL import Image

MODEL = tf.keras.models.load_model(r'C:\Users\AUDRIK\DSC_ANA\weather_prediction\models\2')

app = Flask(__name__)


def preprocess_image(image):
    image = image.resize((256, 256)) 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image

CLASS_NAMES = [
    'Dew', 'Fogsmog', 'Frost', 'Glaze', 'Hail',
    'Lightning', 'Rain', 'Rainbow', 'Rime', 'Sandstorm', 'Snow'
]

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(BytesIO(file.read())) 
            img_batch = preprocess_image(image)

            # Make predictions
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template("index.html", prediction=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
