import json
import os

def save_json(data, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w+") as f:
        json.dump(data, f, indent=4)
