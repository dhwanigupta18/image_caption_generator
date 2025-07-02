# app.py
from flask import Flask, render_template, request, redirect, url_for
from caption_generator import ImageCaptioningModel # <--- ADD THIS LINE
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory # If you chose this for serving uploads

app = Flask(__name__)

# Configuration for uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize models globally to avoid reloading on each request
# This might take a moment when the app starts for the first time
print("Initializing BLIP model...")
blip_model = ImageCaptioningModel()
print("BLIP model initialized.")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    original_caption = None
    image_url = None
    error = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            error = 'No image file part'
            return render_template('index.html', error=error)

        file = request.files['image']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            error = 'No selected image file'
            return render_template('index.html', error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('uploaded_file', filename=filename) # URL for displaying image

            # Get optional prompt and GPT purpose
            prompt = request.form.get('prompt')
            gpt_purpose = request.form.get('gpt_purpose', 'general description')

            # 1. Generate caption using BLIP
            original_caption = blip_model.generate_caption(filepath, prompt)
            caption = original_caption # Initialize caption with BLIP's output


            # Clean up the uploaded image after processing (optional, for production you might store them)
            # os.remove(filepath) # Uncomment to remove after use, but image_url won't work then

        else:
            error = 'Invalid file type. Allowed types: png, jpg, jpeg, gif.'

    return render_template('index.html', caption=caption, original_caption=original_caption, image_url=image_url, error=error)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename)) # Serve from static folder
if __name__ == '__main__':
    app.run(debug=True) 