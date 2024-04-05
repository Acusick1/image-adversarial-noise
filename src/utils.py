import torch
from torchvision.transforms import v2
from config import settings


def get_standard_transforms(
        resize_to: int = 256,
        centre_crop_to: int = 224,
        input_mean: tuple[float, float, float] = settings.in1k_mean, 
        input_std: tuple[float, float, float] = settings.in1k_std,
        ):

    # Typical to resize to 256 first for 224 models, otherwise set the same
    # TODO: refactor, there is no need for CentreCrop at all in this case
    if centre_crop_to > resize_to:
        resize_to = centre_crop_to

    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(resize_to),
        v2.CenterCrop(centre_crop_to),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=input_mean, std=input_std),
    ])
