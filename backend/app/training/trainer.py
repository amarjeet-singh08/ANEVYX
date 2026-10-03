import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.app.models.autoencoder import AdvancedAutoencoder
from backend.app.anomaly.scoring import s_pixel, s_structural, s_perceptual, s_latent

class ModelTrainer:
    """
    Manages the training loop for the ANEVYX Advanced Autoencoder.
    Handles masked reconstruction strategy and checkpointing.
    """
    def __init__(self, model: AdvancedAutoencoder, device: str = "cpu", lr: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
    def _generate_patch_mask(self, batch_size: int, size: int = 256, patch_size: int = 32, mask_ratio: float = 0.25):
        """Generates a random patch mask for the normative decoder training strategy."""
        num_patches = (size // patch_size) ** 2
        num_mask = int(mask_ratio * num_patches)
        
        mask = torch.ones(batch_size, 1, size, size)
        
        for b in range(batch_size):
            # Randomly select patches to mask (set to 0)
            mask_idx = torch.randperm(num_patches)[:num_mask]
            for idx in mask_idx:
                row = (idx // (size // patch_size)) * patch_size
                col = (idx % (size // patch_size)) * patch_size
                mask[b, 0, row:row+patch_size, col:col+patch_size] = 0.0
                
        return mask.to(self.device)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Skip broken images (dummy -1 labels)
            valid_idx = labels >= 0
            if not valid_idx.any():
                continue
                
            images = images[valid_idx].to(self.device)
            
            # Generate occlusion masks to force the network to learn anatomy, not just copy pixels
            input_masks = self._generate_patch_mask(images.size(0), size=images.size(2))
            
            self.optimizer.zero_grad()
            
            # Forward pass (model takes the masked image)
            recons, latents, features = self.model(images, mask=input_masks)
            
            # Target features from unmasked image (for perceptual loss)
            with torch.no_grad():
                target_features = self.model.encoder.feature_maps(images)
                
            # Rule 7: Explicit loss on the masked region. 
            # input_masks is 0 where masked, 1 where visible. 
            # We want to strongly evaluate the missing region.
            eval_mask = 1.0 - input_masks
            
            # Losses (focused on masked region)
            loss_pixel = s_pixel(images, recons, mask=eval_mask).mean()
            loss_struct = s_structural(images, recons, mask=eval_mask).mean()
            # Perceptual loss can remain global, as feature maps are heavily reduced
            loss_perc = s_perceptual(target_features, features).mean()
            
            # Total hybrid loss
            total_loss = loss_pixel + 0.5 * loss_struct + 0.1 * loss_perc
            
            total_loss.backward()
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            
        return {"loss": epoch_loss / max(1, len(dataloader))}

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
