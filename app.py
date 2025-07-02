# app.py (modify imports and initialization)

from flask import Flask, render_template, request, redirect, url_for
from caption_generator import ImageCaptioningModel, WatsonXEnhancer # <--- CHANGE THIS IMPORT
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

print("Initializing BLIP model...")
blip_model = ImageCaptioningModel()
print("BLIP model initialized.")

print("Initializing Watsonx.ai Enhancer...")
# Choose a model ID. Replace with the actual model you want to use from watsonx.ai
# Common IBM models: "ibm/granite-13b-instruct-v2"
# Or a third-party model you have access to: "meta-llama/llama-2-7b-chat", "mistralai/mistral-7b-instruct-v0.2"
watsonx_enhancer = WatsonXEnhancer(model_id="ibm/granite-13b-instruct-v2") # <--- CHANGE THIS LINE
print("Watsonx.ai Enhancer initialized.")


@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_url = None
    error = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            error = 'No file part'
            return render_template('index.html', caption=caption, image_url=image_url, error=error)

        file = request.files['file']

        # If user submits an empty form without selecting a file
        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', caption=caption, image_url=image_url, error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('uploaded_file', filename=filename)

            try:
                # 1. Generate initial caption with BLIP
                original_caption = blip_model.generate_caption(filepath)
                caption = original_caption

                # Get the chosen purpose for refinement from the form
                gpt_purpose = request.form.get('caption_purpose', 'general description')

                # 2. Refine caption using Watsonx.ai (if configured)
                if watsonx_enhancer.model: # Check if watsonx.ai model is initialized
                    refined_caption = watsonx_enhancer.refine_caption(original_caption, gpt_purpose)
                    if refined_caption and "Error" not in refined_caption:
                        caption = refined_caption
                    else:
                        error = f"Watsonx.ai refinement failed: {refined_caption}. Displaying original BLIP caption."
                else:
                    error = "IBM Watsonx.ai not configured. Displaying original BLIP caption."

            except Exception as e:
                error = f"Error generating or refining caption: {e}"
                caption = "Failed to generate caption."

            # Clean up the uploaded file after processing if not needed for display later
            # (Or manage storage based on your needs)
            # os.remove(filepath)

    return render_template('index.html', caption=caption, image_url=image_url, error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)