import numpy as np
import cv2
import torch
import torch.nn.functional as F

def generate_localization_map(
    target: np.ndarray, 
    recon: np.ndarray, 
    feat_target: torch.Tensor, 
    feat_recon: torch.Tensor, 
    mask: np.ndarray = None,
    alpha=0.5,
    normalizers: dict = None
) -> np.ndarray:
    """
    Combines reconstruction residual and feature-space residual into a spatial heatmap.
    Uses the exact MSE logic from scoring.py and scales by the same normalizers.
    """
    # 1. Pixel residual (MSE)
    pixel_res = np.square(target - recon)
    if normalizers and 'mse_max' in normalizers:
        pixel_res = pixel_res / max(float(normalizers['mse_max']), 1e-8)
    else:
        pixel_res = pixel_res / (pixel_res.max() + 1e-8)
    
    # 2. Feature residual (MSE upsampled)
    with torch.no_grad():
        f_diff = torch.square(feat_target - feat_recon)
        f_map = f_diff.mean(dim=1, keepdim=True) # average across channels
        f_map_upscaled = F.interpolate(f_map, size=target.shape, mode='bilinear', align_corners=False)
        f_map_np = f_map_upscaled.squeeze().cpu().numpy()
        
    if normalizers and 'perc_max' in normalizers:
        f_map_np = f_map_np / max(float(normalizers['perc_max']), 1e-8)
    else:
        f_map_np = f_map_np / (f_map_np.max() + 1e-8)
        
    # 3. Combine
    heatmap = (alpha * pixel_res) + ((1 - alpha) * f_map_np)
    
    # Clip to [0, 1] so colormap doesn't wrap around
    heatmap = np.clip(heatmap, 0, 1)
    
    # 4. Mask and smooth
    if mask is not None:
        heatmap = heatmap * mask
        
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    return heatmap
