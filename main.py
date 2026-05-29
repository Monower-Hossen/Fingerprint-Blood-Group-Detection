import torch
from pathlib import Path
from src.utils.common import read_yaml
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def run_pipeline():
    try:
        config = read_yaml("config/config.yaml")

        # Step 1: Ingest Data
        ingestion_cfg = config.data_ingestion
        ingestion_config = DataIngestionConfig(
            root_dir=Path(ingestion_cfg.root_dir),
            kaggle_dataset_slug=ingestion_cfg.kaggle_dataset_slug,
            unzip_dir=Path(ingestion_cfg.unzip_dir),
            zip_file_name=ingestion_cfg.zip_file_name
        )
        ingestion = DataIngestion(ingestion_config)
        ingestion.download_and_extract_data()

        # Step 2: Preprocess and Transform Data
        trans_cfg = config.data_transformation
        transformation_config = DataTransformationConfig(
            root_dir=Path(trans_cfg.root_dir),
            data_path=Path(trans_cfg.data_path),
            batch_size=trans_cfg.batch_size,
            image_size=tuple(trans_cfg.image_size)
        )
        transformation = DataTransformation(transformation_config)
        train_loader, val_loader, classes = transformation.get_data_loaders()
        
        # Step 3: Train Model
        train_cfg = config.model_trainer

        # Safely determine the execution device
        device = train_cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA is requested but not available. Falling back to 'cpu'.")
            device = "cpu"

        trainer_config = ModelTrainerConfig(
            root_dir=Path(train_cfg.root_dir),
            trained_model_path=Path(train_cfg.trained_model_path),
            class_names_path=Path(train_cfg.get("class_names_path", "models/classes.json")),
            epochs=train_cfg.epochs,
            learning_rate=train_cfg.learning_rate,
            device=device
        )
        trainer = ModelTrainer(trainer_config)
        final_accuracy = trainer.initiate_model_trainer(train_loader, val_loader, classes)

        logging.info(f"=== Pipeline Completed Successfully. Final Accuracy: {final_accuracy:.2f}% ===")
        
    except Exception as e:
        logging.error(f"Pipeline execution halted: {str(e)}")

if __name__ == "__main__":
    run_pipeline()