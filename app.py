from flask import Flask, request, url_for, send_from_directory, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
from PIL import Image
import subprocess
import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)
app.config['UPLOADS_DEFAULT_DEST'] = 'uploads'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

model = MobileNetV2(weights='imagenet')

previous_uploads = []

def process_image(file):
    img = Image.open(file)
    img = img.convert('RGB')
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

@app.route('/')
def list_uploads():
    files = os.listdir(app.config['UPLOADS_DEFAULT_DEST'])
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    image_files = [file for file in files if file.lower().endswith(image_extensions)]
    image_urls = [url_for('uploaded_file', filename=file) for file in image_files]

    return jsonify({'image_urls': image_urls})

@app.route('/delete_all', methods=['POST'])
def delete_all_files():
    try:
        upload_dir = app.config['UPLOADS_DEFAULT_DEST']
        for file_name in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({'message': 'All files deleted successfully'})
    except Exception as e:
        return jsonify({'error': 'Failed to delete files', 'details': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADS_DEFAULT_DEST'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo:
            filename = photos.save(photo, folder=None)
            file_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], filename)
            metadata = process_image(file_path)
            metadata_json = json.dumps(metadata, cls=NumpyEncoder)
            previous_uploads.append({'image_url': url_for('uploaded_file', filename=filename), 'metadata': metadata_json})

            return {'metadata': metadata_json, 'image_url': url_for('uploaded_file', filename=filename)}

    return {'error': 'No file uploaded'}, 400

@app.route('/extract_metadata', methods=['POST'])
def extract_metadata():
    if 'photo' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], filename)
    file.save(file_path)

    # Extract metadata using TensorFlow
    metadata = process_image(file_path)
    metadata = json.loads(json.dumps(metadata, cls=NumpyEncoder))

    return jsonify({"metadata": metadata})

@app.route('/extract_exif_metadata', methods=['POST'])
def extract_exif_metadata():
    if 'photo' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOADS_DEFAULT_DEST'], filename)
    file.save(file_path)

    # Extract metadata using ExifTool
    metadata = extract_metadata_with_exiftool(file_path)

    return jsonify({"metadata": metadata})

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
