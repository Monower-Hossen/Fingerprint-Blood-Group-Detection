import os
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements.txt file and returns a list of dependencies.
    It filters out the '-e .' flag if present.
    """
    requirements = []
    try:
        # Use absolute path relative to this file
        abs_path = os.path.join(os.path.dirname(__file__), file_path)
        with open(abs_path, encoding="utf-8") as file_obj:
            requirements = file_obj.readlines()
            # Remove whitespace and newline characters
            requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

            # Remove '-e .' if it is used to trigger setup.py implicitly
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Installing empty dependency list.")
        
    return requirements

setup(
    name="Fingerprint-Blood-Group-Detection",
    version="0.0.1",
    author="Monower Hossen",
    author_email="monower.cse@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description="An end-to-end deep learning pipeline for fingerprint-based blood group detection.",
)