import torch
from torchvision.transforms import v2, Compose
from config import settings


def get_standard_transforms(
        resize_to: int = 256,
        centre_crop_to: int = 224,
        input_mean: tuple[float, float, float] = settings.in1k_mean, 
        input_std: tuple[float, float, float] = settings.in1k_std,
        ) -> Compose:
    
    """
    Get standard preprocessing transforms for images input to an image classifier model.
    
    Parameters:
    - resize_to (int): The size to which the image is resized before cropping (default = 256).
    - centre_crop_to (int): The size of the square crop applied to the center of the image (default = 224).
    - input_mean: The mean for each channel used for normalization (default = ImageNet mean).
    - input_std: The standard deviation for each channel used for normalization(default = ImageNet std).

    Returns:
    - torchvision Compose transform object
    """

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
