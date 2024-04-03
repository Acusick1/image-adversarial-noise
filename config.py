import json
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, FilePath


class Settings(BaseSettings):

    project_path: DirectoryPath = Path().resolve()
    data_path: DirectoryPath = project_path / "data"
    images_path: DirectoryPath = data_path / "sample_images"
    in1k_labels_path: FilePath = data_path / "in1k_labels.json"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = Settings()


def load_in1k_labels():
    
    with open(settings.in1k_labels_path, "r") as f:
        data = json.load(f)

        # Converting json string keys to int
        return {int(k): v for k, v in data.items()}