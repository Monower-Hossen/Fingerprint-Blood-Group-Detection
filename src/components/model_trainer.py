import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig

class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for Fingerprint Classification."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Assuming input size of 128x128. 
        # After three 2x2 MaxPool layers, 128x128 becomes 16x16.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, train_loader, val_loader, classes):
        """
        Trains the CNN model and saves the best performing weights.
        """
        try:
            logging.info(f"Initializing training on device: {self.config.device}")
            num_classes = len(classes)
            model = SimpleCNN(num_classes).to(self.config.device)

            # Save class labels mapping immediately so predictor can use it
            os.makedirs(os.path.dirname(self.config.class_names_path), exist_ok=True)
            with open(self.config.class_names_path, "w") as f:
                json.dump(classes, f)
            logging.info(f"Class names saved to {self.config.class_names_path}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            best_accuracy = 0.0

            for epoch in range(self.config.epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(self.config.device), labels.to(self.config.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()

                # Validation Step
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.config.device), labels.to(self.config.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                logging.info(f"Epoch [{epoch+1}/{self.config.epochs}] - Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.2f}%")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
                    torch.save(model.state_dict(), self.config.trained_model_path)
                    logging.info(f"New best model saved at {self.config.trained_model_path}")

            return best_accuracy

        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer: {str(e)}")
            raise e