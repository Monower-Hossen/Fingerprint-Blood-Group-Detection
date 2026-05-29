# Fingerprint-BloodGroup-Detection

A Deep Learning-based web application designed to predict human blood groups from fingerprint images using a Convolutional Neural Network (CNN).

## 📌 Overview
This project leverages computer vision and deep learning to identify patterns in fingerprints that correlate with blood groups. It provides a full-stack solution with a PyTorch backend and a Flask web interface.

## 🚀 Features
- **Deep Learning Backend**: Uses a custom CNN architecture implemented in PyTorch.
- **Web Interface**: Simple and intuitive UI for uploading fingerprint images and viewing results.
- **Prediction Pipeline**: Automated image preprocessing (resizing, grayscale conversion, and normalization) before inference.
- **Logging**: Detailed logging system to track training progress and prediction requests.
- **Modular Design**: Separated components for training, prediction, and utility functions.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Deep Learning**: PyTorch, Torchvision
- **Web Framework**: Flask
- **Image Processing**: PIL (Pillow)
- **Configuration**: YAML
- **Logging**: Standard Python Logging

## 📁 Project Structure
```text
├── app.py                      # Flask web application entry point
├── src/
│   ├── components/
│   │   └── model_trainer.py    # CNN architecture and training logic
│   ├── pipeline/
│   │   ├── predict_pipeline.py # Inference logic for the web app
│   │   └── train_pipeline.py   # Training execution pipeline
│   ├── entity/                 # Configuration and entity definitions
│   ├── utils/                  # Common utility functions
│   └── logger.py               # Custom logging configuration
├── static/
│   └── uploads/                # Temporary storage for uploaded images
├── templates/
│   └── index.html              # Main web interface
├── config/
│   └── config.yaml             # Project configuration (paths, hyperparameters)
└── requirements.txt            # Project dependencies
```

## 🔧 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Monower-Hossen/Fingerprint-Blood-Group-Detection
   cd Fingerprint-Blood-Group-Detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Model
To train the model on your dataset, configure the paths in `config/config.yaml` and run the training pipeline:
```bash
python src/pipeline/train_pipeline.py
```

### Running the Web App
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Upload a fingerprint image to get the predicted blood group and confidence score.

## 📝 Model Architecture
The project uses a `SimpleCNN` consisting of:
- Three Convolutional layers with ReLU activation and Max Pooling.
- A flattening layer.
- Fully connected layers with Dropout for regularization.
- Output layer matching the number of blood group classes.