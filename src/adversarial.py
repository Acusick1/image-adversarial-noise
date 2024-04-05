import torch
import torch.nn.functional as F
from tqdm import tqdm


def fgsm(model, original_image: torch.Tensor, target_class: int, epsilon: float) -> torch.Tensor:
    
    """
    Applies the Fast Gradient Sign Method (FGSM) adversarial attack to generate a perturbed image that aims to be classified as a specified target class by the model.

    Parameters:
    - model: The model to attack.
    - original_image (torch.Tensor): The original image tensor to perturb.
    - target_class (int): The desired target class index that the model should predict for the perturbed image.
    - epsilon (float): The maximum allowable perturbation to the image.

    Returns:
    - torch.Tensor: The perturbed image tensor.
    """

    original_image.requires_grad = True

    # Forward pass, compute loss
    output = model(original_image)
    loss = F.cross_entropy(output, torch.tensor([target_class]))

    # Backward pass and apply perturbation in direction
    model.zero_grad()
    loss.backward()

    # Perturb image and clip to range
    perturbed_image = original_image + epsilon * -original_image.grad.sign()

    return perturbed_image


def pgd_attack(model, original_image, target_class: int, epsilon: float, alpha: float, max_iter: int, prob_threshold: float = 0.):
    
    """
    Runs a Projected Gradient Descent (PGD) adversarial attack to generate a perturbed image that aims to be classified as a specified target class by the model.

    Parameters:
    - model: The model to be attacked.
    - original_image (torch.Tensor): The input image tensor that will be perturbed.
    - target_class (int): The desired target class index that the model should predict for the perturbed image.
    - epsilon (float): The maximum allowable perturbation to the image.
    - alpha (float): The step size for each iteration.
    - max_iter (int): The maximum number of iterations to run.
    - prob_threshold (float, optional): The probability threshold for early stopping (default = 0 i.e. target class predicted). 

    Returns:
    - torch.Tensor: The perturbed image tensor after the attack.
    - int: The predicted class of the perturbed image by the model.
    - float: The probability of the predicted class.
    - bool: A flag indicating whether the attack was successful in achieving the target class prediction with the desired probability.
    """

    # Initialize perturbed image with random noise within the allowed epsilon range
    perturbed_image = original_image + 0.001 * torch.randn_like(original_image).uniform_(-epsilon, epsilon)

    print("Running PGD attack ...")
    pbar = tqdm(range(max_iter))

    for _ in pbar:
        
        perturbed_image = perturbed_image.clone().detach().requires_grad_(True) 

        # Forward pass
        output = model(perturbed_image)
        loss = F.cross_entropy(output, torch.tensor([target_class]))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        pbar.set_postfix({"loss": loss.item()})
        
        prob, class_idx = torch.max(F.softmax(output, dim=1), 1)
        
        # If we are already predicting the target class and desired probability, avoid making further changes to the image!
        success = class_idx == target_class and prob > prob_threshold

        if success:
            break

        # Apply perturbation using the sign of the gradient (negative for minimizing loss)
        with torch.no_grad():
            perturbation = alpha * -perturbed_image.grad.sign()
            perturbed_image = perturbed_image + perturbation

            # Project the perturbed image back into the epsilon-ball of the original image
            perturbation = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
            perturbed_image = original_image + perturbation

    if success:
        print("Attack successful!")
    else:
        print("Attack failed!")
    
    return perturbed_image, class_idx.item(), prob.item(), success
