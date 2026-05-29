import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from src.entity.config_entity import DataTransformationConfig
from src.logger import logging

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_data_loaders(self):
        try:
            logging.info("Loading dataset and applying transformations...")
            
            if not os.path.exists(self.config.data_path):
                logging.error(f"Data directory missing: {self.config.data_path}. Please verify the folder name after extraction.")
                raise FileNotFoundError(f"The system cannot find the path: {self.config.data_path}")

            # Assuming the dataset is structured in folders by class name
            full_dataset = datasets.ImageFolder(
                root=self.config.data_path,
                transform=self.transform
            )
            
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            logging.info(f"Loaded {len(full_dataset)} images. Classes: {full_dataset.classes}")
            
            return train_loader, val_loader, full_dataset.classes
            
        except Exception as e:
            logging.error(f"Error in Data Transformation: {str(e)}")
            raise e