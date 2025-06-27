import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch # Import torch to potentially move model to GPU if available

app = Flask(__name__)

# --- Configuration ---
# Allowed image extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Temporary upload folder (images will be processed and deleted)
# Ensure this directory exists relative to app.py
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Max file size for uploads (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# --- AI Model Loading ---
# Determine the device to run the model on (GPU if available, else CPU)
# device = "cuda" if torch.cuda.is_available() else "cpu" # For GPU, uncomment this line
device = "cpu" # For simplicity and to ensure it runs on all laptops

# Load the BLIP processor and model
# This will download the model weights the first time it runs.
# It might take a moment.
print(f"Loading BLIP model on {device}...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
print("BLIP model loaded successfully!")

# --- Helper Function for Allowed Files ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Route for Image Captioning ---
@app.route('/caption', methods=['POST'])
def generate_caption():
    # Check if a file part is in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400

    file = request.files['image']

    # Check if no file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    # Process the file if it's allowed
    if file and allowed_file(file.filename):
        # Securely generate a filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) # Save the uploaded file temporarily

        try:
            # Open the image using PIL (Pillow)
            raw_image = Image.open(filepath).convert('RGB')

            # Prepare the image for the model
            inputs = processor(raw_image, return_tensors="pt").to(device)

            # Generate caption (unconditional captioning)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            return jsonify({'caption': caption}), 200

        except Exception as e:
            # Handle any errors during image processing or captioning
            print(f"Error during caption generation: {e}")
            return jsonify({'error': 'Failed to generate caption', 'details': str(e)}), 500
        finally:
            # Ensure the temporary file is deleted
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'File type not allowed or no file provided'}), 400

# --- Basic Route (keep the old one or remove, but it's good for testing server status) ---
@app.route('/')
def index():
    return jsonify(message="Image Captioning Backend is running!")

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Ensure it runs on port 5000