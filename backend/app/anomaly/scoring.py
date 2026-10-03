import torch
import torch.nn.functional as F
import numpy as np

def s_pixel(target: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Pixel-wise Mean Squared Error."""
    err = (target - recon) ** 2
    if mask is not None:
        err = err * mask
        return err.sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3]) + 1e-8)
    return err.mean(dim=[1,2,3])

def create_window(window_size: int, channel: int):
    # 1D Gaussian
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean(dim=[1,2,3])
    return ssim_map

def s_structural(target: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """True SSIM implementation."""
    _, channel, _, _ = target.size()
    window = create_window(11, channel).to(target.device)
    
    ssim_map = _ssim(target, recon, window, 11, channel, size_average=False)
    
    err = 1.0 - ssim_map  # distance
    if mask is not None:
        err = err * mask
        return err.sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3]) + 1e-8)
    return err.mean(dim=[1,2,3])

def s_perceptual(features_target: torch.Tensor, features_recon: torch.Tensor) -> torch.Tensor:
    """Difference in deep feature space between target and reconstruction."""
    return F.mse_loss(features_target, features_recon, reduction='none').mean(dim=[1,2,3])

def s_latent(latent_target: torch.Tensor, latent_recon: torch.Tensor) -> torch.Tensor:
    """Distance in the normality bottleneck latent space."""
    return F.mse_loss(latent_target, latent_recon, reduction='none').mean(dim=[1,2,3])

def compute_hybrid_score(
    target: torch.Tensor, 
    recon: torch.Tensor, 
    feat_target: torch.Tensor, 
    feat_recon: torch.Tensor,
    latent_target: torch.Tensor,
    latent_recon: torch.Tensor,
    mem_dist: torch.Tensor,
    mask: torch.Tensor = None,
    weights: dict = None,
    normalizers: dict = None
) -> dict:
    """Combine all scores into a calibrated hybrid score."""
    if weights is None:
        weights = {"pixel": 1.0, "structural": 1.0, "perceptual": 1.0, "latent": 0.5, "memory": 1.0}
        
    p_score = s_pixel(target, recon, mask)
    
    # Calculate MAE
    mae_err = torch.abs(target - recon)
    if mask is not None:
        mae_err = mae_err * mask
        mae_score = mae_err.sum(dim=[1,2,3]) / (mask.sum(dim=[1,2,3]) + 1e-8)
    else:
        mae_score = mae_err.mean(dim=[1,2,3])
        
    # s_structural returns (1 - SSIM), i.e. the structural error/distance
    struct_err = s_structural(target, recon, mask)
    # SSIM similarity = 1 - structural_error
    ssim_val = 1.0 - struct_err
    
    perc_score = s_perceptual(feat_target, feat_recon)
    lat_score = s_latent(latent_target, latent_recon)
    
    # Handle unfitted memory bank
    if mem_dist is None:
        mem_val = 0.0
        weights["memory"] = 0.0
    else:
        mem_val = mem_dist.item() if mem_dist.numel() == 1 else mem_dist
    
    # Normalize scores (divide by calibration max) to ensure they are relative and strictly non-negative
    n_components = {
        "pixel": (p_score / normalizers.get('mse_max', 1.0)).item() if normalizers else p_score.item(),
        "structural": (struct_err / normalizers.get('struct_err_max', 1.0)).item() if normalizers else struct_err.item(),
        "perceptual": (perc_score / normalizers.get('perc_max', 1.0)).item() if normalizers else perc_score.item(),
        "latent": (lat_score / normalizers.get('lat_max', 1.0)).item() if normalizers else lat_score.item(),
        "memory": float(mem_val / normalizers.get('mem_max', 1.0)) if normalizers and 'mem_max' in normalizers else float(mem_val)
    }
    
    hybrid = (
        weights["pixel"] * n_components["pixel"] +
        weights["structural"] * n_components["structural"] +
        weights["perceptual"] * n_components["perceptual"] +
        weights["latent"] * n_components["latent"] +
        weights["memory"] * n_components["memory"]
    )
    
    return {
        "hybrid": hybrid,
        "pixel": p_score.item() if isinstance(p_score, torch.Tensor) else p_score,
        "mae": mae_score.item() if isinstance(mae_score, torch.Tensor) else mae_score,
        "ssim": ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val,
        "structural": struct_err.item() if isinstance(struct_err, torch.Tensor) else struct_err,
        "perceptual": perc_score.item() if isinstance(perc_score, torch.Tensor) else perc_score,
        "latent": lat_score.item() if isinstance(lat_score, torch.Tensor) else lat_score,
        "memory": mem_val.item() if isinstance(mem_val, torch.Tensor) else mem_val,
        "weights": weights,
        "normalized_components": n_components
    }
