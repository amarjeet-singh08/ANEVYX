import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        # For 256x256 input, the spatial size here is 16x16
        # 256 -> 128 -> 64 -> 32 -> 16
        self.fc = nn.Linear(256 * 16 * 16, latent_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 16, 16)
        x = self.deconv(x)
        return x

class BaselineAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, latent_dim)
        self.decoder = CNNDecoder(in_channels, latent_dim)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

from backend.app.models.encoder import ResNetEncoder
from backend.app.models.bottleneck import NormalityBottleneck
from backend.app.models.decoder import MaskedNormativeDecoder

class AdvancedAutoencoder(nn.Module):
    """
    The full advanced architecture for ANEVYX.
    Image -> Foundation Encoder -> Normality Bottleneck -> Normative Decoder -> Reconstruction
    """
    def __init__(self, latent_dim=256, use_attention=True):
        super().__init__()
        
        # Rule 3 / Point 8: Use one real pretrained implementation first.
        self.encoder = ResNetEncoder(pretrained=True)
        encoder_out_channels = 512 # resnet18 output channels
            
        self.bottleneck = NormalityBottleneck(encoder_out_channels, latent_dim, use_attention)
        self.decoder = MaskedNormativeDecoder(in_channels=latent_dim, out_channels=1)
        
    def forward(self, x, mask=None):
        """
        x: [B, C, H, W]
        mask: optional [B, 1, H, W] masking patches during training
        """
        if mask is not None:
            # Apply mask to input (simulate occlusion)
            x = x * mask
            
        # 1. Foundation Representation
        features = self.encoder.feature_maps(x)
        
        # 2. Normality Bottleneck
        latent = self.bottleneck(features)
        
        # 3. Normative Reconstruction
        # Ensure output size exactly matches input size (Rule 4)
        target_size = (x.shape[2], x.shape[3])
        reconstruction = self.decoder(latent, target_size=target_size)
        
        return reconstruction, latent, features

