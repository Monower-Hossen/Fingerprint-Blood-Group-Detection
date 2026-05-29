import os
import yaml
import logging
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns its content wrapped in a ConfigBox.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content or {})
    except FileNotFoundError:
        raise ValueError(f"YAML file not found at: {path_to_yaml}")
    except BoxValueError:
        raise ValueError(f"YAML file: {path_to_yaml} is empty")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while reading {path_to_yaml}: {e}") from e