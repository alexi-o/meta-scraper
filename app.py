from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import requests
import os
import json
import numpy as np
from io import BytesIO
import ExifRead

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)

model = MobileNetV2(weights='imagenet')

def process_image(image):
    """Process an image and return predictions using MobileNetV2."""
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0]

def extract_metadata_with_exifread(file_path):
    """Extract metadata using ExifRead."""
    with open(file_path, 'rb') as img_file:
        tags = ExifRead.process_file(img_file, details=False, strict=True)
    return {tag: str(tags[tag]) for tag in tags.keys()}

def fetch_image_from_url(url):
    """Fetch an image from a given URL."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

@app.route('/image_recognition', methods=['POST'])
def extract_metadata():
    """Endpoint for image recognition using MobileNetV2."""
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400

    try:
        img = fetch_image_from_url(image_url)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500

    metadata = process_image(img)
    metadata = json.loads(json.dumps(metadata, cls=NumpyEncoder))

    return jsonify({"metadata": metadata})

@app.route('/extract_exif_metadata', methods=['POST'])
def extract_exif_metadata():
    """Endpoint for extracting EXIF metadata."""
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400

    try:
        img = fetch_image_from_url(image_url)
        file_path = 'temp_image.jpg'
        img.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch or save image: {str(e)}"}), 500
    try:
        metadata = extract_metadata_with_exifread(file_path)
        os.remove(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to extract metadata: {str(e)}"}), 500

    return jsonify({"metadata": metadata})

@app.route('/extract_color_palette', methods=['POST'])
def extract_colors():
    """Endpoint for extracting color palette from an image."""
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']
    if not image_url:
        return jsonify({"error": "Invalid URL"}), 400
    try:
        img = fetch_image_from_url(image_url)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500
    try:
        palette = extract_color_palette(img)
    except Exception as e:
        return jsonify({"error": f"Failed to extract color palette: {str(e)}"}), 500

    return jsonify({"palette": palette})

def extract_color_palette(image):
    """Extract color palette from the image."""
    colors = image.getcolors(maxcolors=1000)  # Modify maxcolors as needed
    sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
    palette = [color[1] for color in sorted_colors[:5]]  # Get top 5 colors
    return palette

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
