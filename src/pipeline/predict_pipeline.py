import os
import json
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.utils.common import read_yaml
from src.components.model_trainer import SimpleCNN
from src.logger import logging

class PredictPipeline:
    """Pipeline to handle fingerprint image loading and blood group prediction."""
    def __init__(self):
        try:
            self.config = read_yaml("config/config.yaml")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"PredictPipeline initialized using device: {self.device}")

            # 1. Define transformations (Ensure it matches DataTransformation component)
            image_size = tuple(self.config.data_transformation.image_size)
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            # 2. Load classes from JSON
            class_path = Path(self.config.model_trainer.class_names_path)
            if class_path.exists():
                with open(class_path, "r") as f:
                    classes = json.load(f)
                # Ensure we handle list or dict formats from JSON
                if isinstance(classes, list):
                    self.class_map = {i: label for i, label in enumerate(classes)}
                else:
                    self.class_map = {int(k): v for k, v in classes.items()}
                logging.info(f"Loaded {len(self.class_map)} class labels.")
            else:
                # Fallback mapping if file is missing
                logging.warning(f"Class names file not found at {class_path}. Using default mapping.")
                self.class_map = {0: 'A+', 1: 'A-', 2: 'AB+', 3: 'AB-', 4: 'B+', 5: 'B-', 6: 'O+', 7: 'O-'}
            
            # Pre-load the model once during initialization
            self.model = self._initialize_model()
        
        except Exception as e:
            logging.error(f"Critical error during PredictPipeline initialization: {e}")
            self.config = None
            self.class_map = {}
            self.model = None

    def _initialize_model(self):
        """Initializes model architecture and loads trained weights."""
        try:
            if self.config is None: return None
            
            model_path = Path(self.config.model_trainer.trained_model_path)
            if not model_path.exists():
                logging.error(f"Model file missing! Expected at: {model_path.absolute()}")
                return None

            state_dict = torch.load(model_path, map_location=self.device)
            # Detect classes from checkpoint to avoid size mismatch errors
            # Flexible detection of num_classes based on the last weight layer
            last_layer_key = list(state_dict.keys())[-1] # Usually weight of the last layer
            if 'weight' in last_layer_key:
                num_classes = state_dict[last_layer_key].shape[0]
            else:
                num_classes = len(self.class_map)
            
            if num_classes <= 1:
                logging.warning(f"Warning: Model loaded with only {num_classes} class(es). Check your training data path.")
            
            model = SimpleCNN(num_classes).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            logging.info(f"Model loaded successfully with {num_classes} classes.")
            return model
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
            return None

    def predict(self, image_path: str):
        """
        Predicts the blood group from a fingerprint image.
        
        Args:
            image_path (str): Path to the fingerprint image file.
            
        Returns:
            str: Predicted blood group label.
        """
        try:
            logging.info(f"Starting prediction for image: {image_path}")
            
            if self.model is None:
                return "Error: Model could not be loaded. Please check the model path."
            
            # 2. Load and transform image
            image = Image.open(image_path)
            image = self.transform(image).unsqueeze(0).to(self.device)

            # 3. Perform inference using the pre-loaded model
            with torch.no_grad():
                logits = self.model(image)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                prediction_idx = predicted.item()
                # Convert to percentage
                prob_score = confidence.item() * 100
                
            blood_group = self.class_map.get(prediction_idx, "Unknown")
            result_str = f"{blood_group} (Confidence: {prob_score:.2f}%)"
            logging.info(f"Prediction result: {result_str}")
            
            return result_str

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise e