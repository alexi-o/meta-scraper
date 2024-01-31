# Flask Image Upload and Metadata Generation App

This is a Flask web application that allows users to upload images, generate image metadata, and display the uploaded image along with its metadata. The application utilizes the MobileNetV2 model from TensorFlow for image processing and metadata generation.

## Features

- Upload images and receive metadata about the content.
- Display the uploaded image along with its metadata.
- Utilizes the MobileNetV2 model for image classification.
- Provides an API endpoint for uploading images and receiving metadata in JSON format.

## Setup

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/flask-image-app.git
   ```
2. Navigate to the project directory:
    ```bash
   cd flask-image-app
   ```
3. Create a virtual environment (optional but recommended):
    ```bash
   python -m venv venv
   ```
4. Activate the virtual environment (replace venv with the appropriate name if you used a different name):
    ```bash
   source venv/bin/activate   # On macOS and Linux
    venv\Scripts\activate      # On Windows
   ```
5. Install the required Python packages using the requirements.txt file:
    ```bash
   pip install -r requirements.txt
   ```
6. Run the Flask app:
    ```bash
   python app.py
   ```
7. Open your web browser and navigate to http://localhost:5000 to access the web interface.

## API Endpoint
You can also use the API endpoint to upload an image and receive metadata in JSON format. Send a POST request to the following URL:
```bash
   http://localhost:5000/upload
   ```
Ensure that you include a file named "photo" in your POST request.

## Example Response
Here's an example response from the API:
```bash
{
    "metadata": "[[\"n02099601\", \"golden_retriever\", 0.15427792072296143], [\"n02113799\", \"standard_poodle\", 0.15024328231811523], [\"n02098105\", \"soft-coated_wheaten_terrier\", 0.1460559070110321], [\"n02091635\", \"otterhound\", 0.05535516142845154], [\"n02113712\", \"miniature_poodle\", 0.0528804175555706]]",
    "image_url": "/uploads/dog_5.jpeg"
}
```

## Dependencies
You can find the list of required Python dependencies in the requirements.txt file. To install them, use the following command:

```bash
pip install -r requirements.txt
```
