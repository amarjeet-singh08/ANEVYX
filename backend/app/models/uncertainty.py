import torch
import torchvision.transforms.functional as TF
import numpy as np

def generate_tta_variants(image_tensor: torch.Tensor, num_variants: int = 4) -> list[torch.Tensor]:
    """
    Generates mild Test-Time Augmentations for uncertainty estimation.
    Perturbations: small brightness, contrast, and translation changes.
    """
    variants = [image_tensor] # original is always included
    
    # 1. Brightness
    variants.append(TF.adjust_brightness(image_tensor, 1.05))
    variants.append(TF.adjust_brightness(image_tensor, 0.95))
    
    # 2. Contrast
    variants.append(TF.adjust_contrast(image_tensor, 1.05))
    variants.append(TF.adjust_contrast(image_tensor, 0.95))
    
    # We take up to `num_variants`
    return variants[:num_variants]

def calculate_uncertainty(scores: list[float]) -> dict:
    """
    Calculate uncertainty based on the variance of scores across TTA variants.
    """
    if not scores:
        return {"level": "unknown", "variance": 0.0}
        
    variance = float(np.var(scores))
    mean = float(np.mean(scores))
    
    # Thresholds would ideally be calibrated
    if variance > 0.05:
        level = "highly_uncertain"
    elif variance > 0.01:
        level = "moderately_uncertain"
    else:
        level = "stable"
        
    return {
        "level": level,
        "variance": variance,
        "mean_score": mean
    }
