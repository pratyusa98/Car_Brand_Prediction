from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os, sys, glob, re
import tensorflow_hub as hub

app = Flask(__name__)

model = load_model(('model_mobilenet.h5'), custom_objects={'KerasLayer': hub.KerasLayer})


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The Car IS BMW"
    elif preds == 1:
        preds = "The Car is Ferrari"
    elif preds == 2:
        preds = "The Car is Ford"
    elif preds == 3:
        preds = "The Car is Mercedes-Benz"
    elif preds == 4:
        preds = "The Car is Audi"
    else:
        preds = "The Car Is lamborghini"

    return preds


IMG_FOLDER = os.path.join('static', 'upload')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('static/upload', filename)
        file.save(file_path)

        pred = model_predict(file_path, model)

        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        return render_template('index.html', predict=pred, user_image=full_filename)


if __name__ == '__main__':
    app.run(debug=True)

