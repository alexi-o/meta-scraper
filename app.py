from flask import Flask, render_template, request, redirect, url_for, flash
from flask_uploads import UploadSet, configure_uploads, IMAGES
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOADS_DEFAULT_DEST'] = 'uploads'  # Folder to store uploaded images
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

# Load a pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to process uploaded image and generate metadata
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        if photo:
            filename = photos.save(photo, folder='photos')  # Save to 'photos' subfolder
            metadata = process_image(os.path.join(app.config['UPLOADS_DEFAULT_DEST'], 'photos', filename))
            flash('Metadata generated successfully!')
            return render_template('index.html', metadata=metadata, image_url=url_for('static', filename=f'uploads/photos/{filename}'))
    
    return render_template('index.html', metadata=None, image_url=None)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True)
