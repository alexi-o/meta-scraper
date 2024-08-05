from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import subprocess
import requests
import os
import json
import numpy as np
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)

# Load the model
model = MobileNetV2(weights='imagenet')

def process_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]

def extract_metadata_with_exiftool(file_path):
    result = subprocess.run(['exiftool', '-json', file_path], stdout=subprocess.PIPE)
    metadata = json.loads(result.stdout.decode('utf-8'))[0]
    return metadata

def fetch_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')  # Convert RGBA to RGB
    return img

@app.before_request
def log_request_info():
    logging.info("Headers: %s", request.headers)
    logging.info("Body: %s", request.get_data())

@app.after_request
def log_response_info(response):
    logging.info("Response status: %s", response.status)
    return response

@app.route('/')
def home():
    return "Welcome to the API", 200

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/image_recognition', methods=['POST'])
def extract_metadata():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400

    # Fetch the image from the URL
    try:
        img = fetch_image_from_url(image_url)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500

    # Extract metadata using TensorFlow
    try:
        metadata = process_image(img)
        metadata = json.loads(json.dumps(metadata, cls=NumpyEncoder))
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    return jsonify({"message": "Image recognition successful", "metadata": metadata})

@app.route('/extract_exif_metadata', methods=['POST'])
def extract_exif_metadata():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400

    # Fetch the image from the URL
    try:
        img = fetch_image_from_url(image_url)
        file_path = 'temp_image.jpg'
        img.save(file_path, format='JPEG')  # Ensure the image is saved as JPEG
    except Exception as e:
        return jsonify({"error": f"Failed to fetch or save image: {str(e)}"}), 500

    # Extract metadata using ExifTool
    try:
        metadata = extract_metadata_with_exiftool(file_path)
        os.remove(file_path)  # Clean up the temporary file
    except Exception as e:
        return jsonify({"error": f"Failed to extract metadata: {str(e)}"}), 500

    return jsonify({"message": "EXIF metadata extraction successful", "metadata": metadata})

@app.route('/extract_color_palette', methods=['POST'])
def extract_colors():
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400

    # Fetch the image from the URL
    try:
        img = fetch_image_from_url(image_url)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500

    # Extract color palette
    try:
        palette = extract_color_palette(img)
    except Exception as e:
        return jsonify({"error": f"Failed to extract color palette: {str(e)}"}), 500

    return jsonify({"message": "Color palette extraction successful", "palette": palette})

def extract_color_palette(image):
    # Example function to extract colors
    colors = image.getcolors(maxcolors=1000)  # Modify maxcolors as needed
    sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
    palette = [color[1] for color in sorted_colors[:5]]  # Get top 5 colors
    return palette

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
