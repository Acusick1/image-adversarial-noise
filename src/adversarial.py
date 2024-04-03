import torch
import torch.nn.functional as F


def fgsm(model, original_image, target_class: int, epsilon: float):
    
    original_image.requires_grad = True

    # Forward pass, compute loss
    output = model(original_image)
    loss = F.cross_entropy(output, torch.tensor([target_class]))

    # Backward pass and apply perturbation in direction
    model.zero_grad()
    loss.backward()

    # Perturb image and clip to range
    perturbed_image = original_image + epsilon * original_image.grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image