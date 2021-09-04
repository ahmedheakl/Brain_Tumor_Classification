import os
import cv2
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from utils import customize
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
import requests

app = Flask(__name__)

model = load_model('CategoricalBrainModel10epochs.h4')


def get_className(classNumnber):
    if classNumnber == 0:
        return 'No Brain Tumor'
    elif classNumnber == 1:
        return 'Yes Brain Tumor'
    else:
        raise Exception('classNumber must be 1 or 0')


def get_result(img):
    image = customize(img)
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(input_img))
    class_name = get_className(result)
    return class_name


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        n = request.form['file']
        name = n.split('/')[-1]

        response = requests.get(n)

        file = open(f"uploads/{name}", "wb")
        file.write(response.content)
        file.close()

        result = get_result(f"uploads/{name}")

        return render_template('index.html', path=n, prediction=result)
    return "Request method must be POST"


if __name__ == '__main__':
    app.run(debug=True)
