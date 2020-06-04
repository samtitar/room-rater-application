from flask import Flask, render_template, request, redirect, jsonify
from model import Classifier
from PIL import Image

import torch
import io
import os
import json
import torchvision.transforms as transforms

model = Classifier()
model.eval()

app = Flask(__name__)

def transform_image(image):
    im_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(io.BytesIO(image))
    return im_transforms(image).unsqueeze(0)

def get_prediction(image):
    inputs = transform_image(image)
    outputs = model(inputs)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return predicted_idx

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    return jsonify({ 'rating': get_prediction(img_bytes) })


if __name__ == '__main__':
    model_state = torch.load('model.tar')
    model.load_state_dict(model_state)
    app.run(host='0.0.0.0')