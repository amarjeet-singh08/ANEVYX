import torch
import torch.nn as nn

class MaskedNormativeDecoder(nn.Module):
    """
    Decodes the structured latent representation back into normal anatomy.
    Designed to reconstruct expected anatomy even when input regions are masked.
    """
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        
        # Upsampling blocks
        self.decoder = nn.Sequential(
            # Input: [B, in_channels, H, W]
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            
            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, target_size=None):
        """
        x: [B, in_channels, H, W] from the normality bottleneck
        target_size: (H, W) of the original image to guarantee Rule 4.
        """
        out = self.decoder(x)
        if target_size is not None and out.shape[-2:] != target_size:
            import torch.nn.functional as F
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out
