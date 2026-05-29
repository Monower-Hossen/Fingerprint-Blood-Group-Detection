# Fingerprint-Based Blood Group Detection

A professional Deep Learning-based web application designed to predict human blood groups (A+, A-, B+, B-, AB+, AB-, O+, O-) from fingerprint images using a Convolutional Neural Network (CNN). This project implements a modular MLOps-ready structure for scalability and reproducibility.

## 📌 Overview
This project leverages computer vision and deep learning to identify patterns in fingerprints that correlate with blood groups. It provides a full-stack solution with a PyTorch backend and a Flask web interface.

## 📊 Dataset Information
The project is designed to work with the **Fingerprint Dataset for Blood Group Classification**.
- **Source**: Kaggle (`rohitpravinlohar/fingerprint-dataset-for-blood-group-classification`)
- **Input**: Grayscale fingerprint images.
- **Output**: One of 8 blood group categories.

## 🏗 Project Workflow
1. **Data Ingestion**: Automatically downloads and unzips the dataset from Kaggle.
2. **Data Transformation**: Resizes images to $128 \times 128$, converts them to grayscale, and normalizes the pixel values.
3. **Model Training**: 
    - Uses a custom `SimpleCNN` architecture.
    - Implements Early Saving (saves the best model based on validation accuracy).
    - Logs training metrics (loss/accuracy) per epoch.
4. **Model Evaluation**: Validates the model on a separate validation set to ensure generalization.
5. **Deployment/Prediction**: A Flask-based interface allows users to upload an image and get a real-time prediction with a confidence score.

## 🚀 Features
- **Deep Learning Backend**: Uses a custom CNN architecture implemented in PyTorch.
- **Web Interface**: Simple and intuitive UI for uploading fingerprint images and viewing results.
- **Config-Driven**: All hyperparameters and paths are managed via `config/config.yaml`.
- **Prediction Pipeline**: Automated image preprocessing (resizing, grayscale conversion, and normalization) before inference.
- **Logging**: Detailed logging system to track training progress and prediction requests.
- **Modular Design**: Separated components for training, prediction, and utility functions.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Deep Learning**: PyTorch, Torchvision
- **Web Framework**: Flask
- **Image Processing**: PIL (Pillow)
- **Configuration**: YAML
- **Packaging**: Setuptools (`setup.py`)
- **Logging**: Standard Python Logging

## 📁 Project Structure
```text
├── app.py                      # Flask web application entry point
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Handles dataset downloading
│   │   ├── data_transformation.py # Preprocessing logic
│   │   └── model_trainer.py    # CNN architecture and training loop
│   ├── pipeline/
│   │   ├── predict_pipeline.py # Inference logic for the web app
│   │   └── train_pipeline.py   # Training execution pipeline
│   ├── entity/                 # Configuration and entity definitions
│   ├── constants/              # Hardcoded paths and values
│   ├── utils/                  # Common utility functions
│   └── logger.py               # Custom logging configuration
├── static/
│   └── uploads/                # Temporary storage for uploaded images
├── templates/
│   └── index.html              # Main web interface
├── config/
│   ├── schema.yaml             # Configuration validation schema
│   └── config.yaml             # Project configuration (paths, hyperparameters)
└── requirements.txt            # Project dependencies
```

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Kaggle Account (for dataset access)
- NVIDIA GPU (Optional, for faster training)

### Steps
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

3. **Kaggle API Setup**:
   Place your `kaggle.json` file in `~/.kaggle/` (or `C:\Users\<User>\.kaggle\`) to enable automatic data ingestion.

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Configuration
Modify `config/config.yaml` to adjust hyperparameters:
```yaml
model_trainer:
  epochs: 10
  learning_rate: 0.001
  device: "cuda" # or "cpu"
```

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

## 🧠 Model Architecture
The `SimpleCNN` is designed for high-resolution fingerprint features:
- **Feature Extractor**: 
    - 3x Conv2D layers (32, 64, 128 filters).
    - ReLU activations and 2x2 MaxPool layers.
- A flattening layer.
- **Classifier**:
    - Fully connected layer (512 nodes).
    - Dropout (0.5) for overfitting prevention.
    - Final linear layer for 8-class classification.

## 🛠 Troubleshooting
- **Model not loading**: Ensure you have run the training pipeline first or that `model.pth` exists in `artifacts/model_trainer/`.
- **Device Error**: If you don't have a GPU, ensure `device` is set to `"cpu"` in the config.

## 📜 License
This project is licensed under the MIT License.