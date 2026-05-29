import torch
from src.utils.common import read_yaml
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from pathlib import Path

class TrainPipeline:
    def __init__(self):
        self.config = read_yaml("config/config.yaml")

    def run(self):
        # 1. Data Ingestion
        ingestion_cfg = self.config.data_ingestion
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(ingestion_cfg.root_dir),
            kaggle_dataset_slug=ingestion_cfg.kaggle_dataset_slug,
            unzip_dir=Path(ingestion_cfg.unzip_dir),
            zip_file_name=ingestion_cfg.zip_file_name
        )
        ingestion = DataIngestion(data_ingestion_config)
        ingestion.download_and_extract_data()

        # 2. Data Transformation
        trans_cfg = self.config.data_transformation
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(trans_cfg.root_dir),
            data_path=Path(trans_cfg.data_path),
            batch_size=trans_cfg.batch_size,
            image_size=tuple(trans_cfg.image_size)
        )
        transformation = DataTransformation(data_transformation_config)
        train_loader, val_loader, classes = transformation.get_data_loaders()

        # 3. Model Training
        train_cfg = self.config.model_trainer
        
        device = train_cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(train_cfg.root_dir),
            trained_model_path=Path(train_cfg.trained_model_path),
            class_names_path=Path(train_cfg.get("class_names_path", "models/classes.json")),
            epochs=train_cfg.epochs,
            learning_rate=train_cfg.learning_rate,
            device=device
        )
        trainer = ModelTrainer(model_trainer_config)
        trainer.initiate_model_trainer(train_loader, val_loader, classes)

if __name__ == "__main__":
    try:
        TrainPipeline().run()
    except Exception as e:
        logging.exception(e)