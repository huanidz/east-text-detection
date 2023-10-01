import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

project_name = "EAST"

# List of template folders and files
list_of_files = [
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/utils/__init__.py",
    f"src/utils/common.py",
    f"src/logging/__init__.py",
    f"src/config/__init__.py",
    f"src/visualizing/__init__.py",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
]

# Creating folders and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory - {filedir} for the file {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file - {filepath}")
    else:
        logging.info(f"{filename} is already exist.")