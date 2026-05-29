from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Define a path for uploads, if needed later
# This path is relative to where app.py is run.
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize pipeline once on startup
pipeline = PredictPipeline()

@app.route('/')
def index():
    """Renders the main index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and calls the prediction pipeline."""
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            # Execute prediction
            result = pipeline.predict(file_path)
            return render_template('index.html', result=result, image_path=file_path)
        except Exception as e:
            return render_template('index.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows for automatic reloading and provides a debugger