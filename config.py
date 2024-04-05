import json
import torch
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import DirectoryPath, FilePath


class Settings(BaseSettings):

    project_path: DirectoryPath = Path().resolve()
    data_path: DirectoryPath = project_path / "data"
    images_path: DirectoryPath = data_path / "sample_images"
    output_path: Path = project_path / "outputs"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Move to own in1k dataclass
    in1k_labels_path: FilePath = data_path / "in1k_labels.json"
    in1k_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    in1k_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_path.mkdir(exist_ok=True)


settings = Settings()


def load_in1k_labels():
    
    with open(settings.in1k_labels_path, "r") as f:
        data = json.load(f)

        # Converting json string keys to int
        return {int(k): v for k, v in data.items()}