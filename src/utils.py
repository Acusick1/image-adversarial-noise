import torch
from PIL import Image
from torchvision.transforms import v2


def load_image(image_path: str):

    return Image.open(image_path)


def get_standard_transforms(
        resize_to: int = 256,
        centre_crop_to: int = 224,
        input_mean: tuple[float, float, float] = (0.485, 0.456, 0.406), 
        input_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        ):


    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(resize_to),
        v2.CenterCrop(centre_crop_to),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=input_mean, std=input_std),
    ])
