import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/main.yaml",
    "config/config.yaml",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/artifacts/.gitkeep",
    "docs/.gitkeep",
    "models/.gitkeep",
    "notebooks/01_eda_and_preprocessing.ipynb",
    "notebooks/02_model_training_experiment.ipynb",
    "src/__init__.py",
    "src/logger.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "app.py",
    "templates/index.html",
    "static/css/style.css",
    "requirements.txt",
    "setup.py",
    "README.md",
    ".gitignore"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")