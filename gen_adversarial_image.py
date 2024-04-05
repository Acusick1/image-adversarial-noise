import argparse
import random
import timm
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from typing import Optional
from src.adversarial import pgd_attack
from src.utils import get_standard_transforms
from config import settings, load_in1k_labels

batch_mean = torch.tensor(settings.in1k_mean).view(1, -1, 1, 1).to(settings.device)
batch_std = torch.tensor(settings.in1k_std).view(1, -1, 1, 1).to(settings.device)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser("Generate wrong-class images with adversarial noise!")
    parser.add_argument("image_path", type=Path, help="path to image to be used.")
    
    # TODO: validate within [0, 1000] range
    parser.add_argument("--target_class", default=None, type=int, help="imagenet1k class to target, default is a random class")
    parser.add_argument("--model_name", default="resnet18.a1_in1k", type=str, help="pretrained in1k model to be fooled (from timm model library).")

    # TODO: validate within [0, 1]
    parser.add_argument("--epsilon", default=0.05, type=float, help="maximum allowable perturbation of input image")
    parser.add_argument("--alpha", default=0.01, type=float, help="step size for optimization")
    parser.add_argument("--max_steps", default=10, type=int, help="maximum number of PGD iterations")
    parser.add_argument("--prob_threshold", default=0., type=float, help="probability threshold to achieve on target class")

    args = parser.parse_args()
    return args

def run(
        image_path: Path, 
        target_class: Optional[int] = None, 
        model_name: str = "resnet18.a1_in1k", 
        epsilon: float = 0.05, 
        alpha: float = 0.01, 
        max_steps: int = 10, 
        prob_threshold: float = 0.
    ):

    if target_class is None:
        target_class = random.randint(a=0, b=1000)

    model = timm.create_model(model_name, pretrained=True)
    model.to(settings.device)
    model.eval()
    assert model.num_classes == 1000

    classes = load_in1k_labels()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = get_standard_transforms(centre_crop_to=data_config["input_size"][1])
    
    input_image = Image.open(image_path)
    input_image = transforms(input_image).unsqueeze(0)
    
    orig_output = model(input_image)
    orig_prob, orig_class_idx = torch.max(F.softmax(orig_output, dim=1), 1)
    
    print(
        f"Input image: {image_path.name}",
        f"Using model: {model_name}",
        f"Original prediction: '{classes[orig_class_idx.item()]}' (probability: {orig_prob.item():.2f})",
        f"Targetting class: '{classes[target_class]}' (ID: {target_class})\n",
        sep="\n"
    )

    perturbed_image, class_idx, prob, success = pgd_attack(model, input_image, target_class, epsilon, alpha, max_steps, prob_threshold)
    print(f"\nPerturbed prediction: '{classes[class_idx]}' (probability: {prob:.2f})")
    
    # Unnormalize and scale
    perturbed_image = perturbed_image * batch_std + batch_mean
    perturbed_image = perturbed_image - perturbed_image.amin() / (perturbed_image.amax() - perturbed_image.amin())
    
    # Saving output image
    success_suffix = "SUCCESS" if success else "FAILED"
    save_name = f"{image_path.stem}_to_class{target_class}_{success_suffix}"
    save_path = (settings.output_path / save_name).with_suffix(".JPEG")
    to_pil_image(perturbed_image.squeeze()).save(save_path, "JPEG")
    print(f"Perturbed image saved to {save_path.relative_to(settings.project_path.parent)}")


if __name__ == "__main__":

    args = parse_args()
    run(
        args.image_path,
        args.target_class,
        args.model_name,
        args.epsilon,
        args.alpha,
        args.max_steps,
        args.prob_threshold
    )