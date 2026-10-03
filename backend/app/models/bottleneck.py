import torch
import torch.nn as nn

class NormalityBottleneck(nn.Module):
    """
    A configurable bottleneck that represents the expected structure of normal anatomy.
    This replaces a simple dense layer with a more structured latent space.
    """
    def __init__(self, in_features, latent_dim=256, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.latent_dim = latent_dim
        
        # Spatial reduction
        self.conv_reduce = nn.Conv2d(in_features, latent_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1) # MC Dropout
        
        if use_attention:
            # Self-attention over the spatial features to learn structural relationships
            self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(latent_dim)
            self.ff = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 4),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(latent_dim * 4, latent_dim)
            )
            self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: [B, C, H, W] from the foundation encoder
        """
        B, C, H, W = x.shape
        # Reduce dimension
        x = self.conv_reduce(x) # [B, latent_dim, H, W]
        x = self.dropout(x)
        
        if self.use_attention:
            # Flatten spatial dimensions for attention
            x_flat = x.view(B, self.latent_dim, -1).permute(0, 2, 1) # [B, H*W, latent_dim]
            
            # Self attention
            attn_out, _ = self.attention(x_flat, x_flat, x_flat)
            x_flat = self.norm1(x_flat + attn_out)
            
            # Feed forward
            ff_out = self.ff(x_flat)
            x_flat = self.norm2(x_flat + ff_out)
            
            # Reshape back to spatial
            x = x_flat.permute(0, 2, 1).view(B, self.latent_dim, H, W)
            
        return x
