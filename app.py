from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
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

@app.route('/')
def list_uploads():
    files = os.listdir(app.config['UPLOADS_DEFAULT_DEST'])
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
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
            metadata = process_image(os.path.join(app.config['UPLOADS_DEFAULT_DEST'], filename))
            metadata_json = json.dumps(metadata, cls=NumpyEncoder)

            # Append upload information to previous_uploads list
            previous_uploads.append({'image_url': url_for('uploaded_file', filename=filename), 'metadata': metadata_json})

            return {'metadata': metadata_json, 'image_url': url_for('uploaded_file', filename=filename)}

    return {'error': 'No file uploaded'}, 400

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
