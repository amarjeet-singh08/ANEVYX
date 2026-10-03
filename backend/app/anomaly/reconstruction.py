import torch
import torch.nn.functional as F
import numpy as np

def calculate_mse_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Basic MSE Loss for the baseline autoencoder."""
    return F.mse_loss(recon, target)

def calculate_reconstruction_heatmap(target: np.ndarray, recon: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate pixel-wise squared error and normalize it to [0, 1] for visualization.
    If a mask is provided, error outside the mask is set to zero.
    """
    # Ensure float32
    target = target.astype(np.float32)
    recon = recon.astype(np.float32)
    
    # Calculate squared error
    error = (target - recon) ** 2
    
    if mask is not None:
        error = error * mask.astype(np.float32)
        
    # Normalize heatmap
    if error.max() > 0:
        heatmap = error / error.max()
    else:
        heatmap = error
        
    return heatmap
