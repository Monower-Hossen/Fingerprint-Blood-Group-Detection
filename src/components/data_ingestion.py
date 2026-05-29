import os
import zipfile
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.utils.common import read_yaml
from src.entity.config_entity import DataIngestionConfig
 
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_and_extract_data(self):
        """
        Downloads the fingerprint dataset via Kaggle API 
        and extracts it directly into the pipeline directories.
        """
        try:
            # Create directories if they do not exist
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            
            zip_file_path = self.config.root_dir / self.config.zip_file_name

            # 1. Download via Kaggle CLI programmatically
            if not os.path.exists(zip_file_path):
                logging.info(f"Fetching dataset: {self.config.kaggle_dataset_slug}...")
                
                # Command executes: kaggle datasets download -d <slug> -p <path>
                os.system(f"kaggle datasets download -d {self.config.kaggle_dataset_slug} -p {self.config.root_dir}")
                
                # Kaggle saves it with the dataset slug name, let's locate and reference it
                downloaded_file = self.config.root_dir / "fingerprint-dataset-for-blood-group-classification.zip"
                if os.path.exists(downloaded_file):
                    os.rename(downloaded_file, zip_file_path)
            else:
                logging.info("Zip file already exists. Skipping download.")

            # 2. Extract files
            logging.info("Extracting fingerprint images into raw artifacts...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)
                
            logging.info(f"Data ingestion complete! Extracted to: {self.config.unzip_dir}")
            
        except Exception as e:
            logging.error(f"Error during data ingestion stage: {str(e)}")
            raise e

if __name__ == "__main__":
    # Test execution
    try:
        config = read_yaml("config/config.yaml")
        ingestion_cfg = config.data_ingestion
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(ingestion_cfg.root_dir),
            kaggle_dataset_slug=ingestion_cfg.kaggle_dataset_slug,
            unzip_dir=Path(ingestion_cfg.unzip_dir),
            zip_file_name=ingestion_cfg.zip_file_name
        )
        ingestion = DataIngestion(data_ingestion_config)
        ingestion.download_and_extract_data()
    except Exception as e:
        logging.error(e)