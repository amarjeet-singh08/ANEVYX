import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime

from backend.app.models.autoencoder import AdvancedAutoencoder
from backend.app.imaging.preprocessing import preprocess_pipeline
from backend.app.anomaly.scoring import compute_hybrid_score
from backend.app.anomaly.ood import OODDetector
from backend.app.models.memory_bank import NormalityMemoryBank

class RealCXRDataset(Dataset):
    def __init__(self, directory: str, target_size: int = 256):
        self.directory = directory
        self.target_size = target_size
        self.image_paths = []
        
        if os.path.exists(directory):
            for fname in os.listdir(directory):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    self.image_paths.append(os.path.join(directory, fname))
                    
        if len(self.image_paths) == 0:
            print(f"WARNING: No images found in {directory}. Skipping dummy generation as per strict rules.")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # ANEVYX STRICT RULE: Do not replace missing with black images. If read fails, raise error.
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image at {path}. Corrupt or missing file.")
            
        # ANEVYX STRICT RULE: Use EXACTLY the same preprocessing
        processed = preprocess_pipeline(img, target_size=self.target_size)
        return torch.from_numpy(processed).float().unsqueeze(0)

def train_pipeline(data_dir: str = "data", epochs: int = 50, batch_size: int = 8):
    train_dir = os.path.join(data_dir, "train", "normal")
    val_dir = os.path.join(data_dir, "val", "normal")
    calib_dir = os.path.join(data_dir, "calibration")
    
    train_ds = RealCXRDataset(train_dir)
    val_ds = RealCXRDataset(val_dir)
    calib_ds = RealCXRDataset(calib_dir)
    
    if len(train_ds) == 0:
        raise RuntimeError(f"No training data found in {train_dir}. Cannot train real model.")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)
    
    model = AdvancedAutoencoder()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    print(f"Starting training on {len(train_ds)} real images...")
    
    # 1. Train Autoencoder
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon, latent, feat = model(batch)
            loss = torch.mean((batch - recon) ** 2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f}")
        
    # 2. Build Memory Bank and OOD Matrices from Training Set
    print("Building memory bank and OOD distributions from training data...")
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for batch in train_loader:
            embed = model.encoder.embedding(batch)
            all_embeds.append(embed)
    
    all_embeds = torch.cat(all_embeds, dim=0)
    
    memory_bank = NormalityMemoryBank(embed_dim=512)
    memory_bank.add_features(all_embeds)
    
    ood_detector = OODDetector()
    ood_detector.fit(all_embeds)
    
    # 3. Calibrate Anomaly Threshold using SEPARATE CALIBRATION SET
    print(f"Calibrating anomaly threshold on {len(calib_ds)} separate calibration images...")
    if len(calib_ds) == 0:
        raise RuntimeError("No calibration data found. Cannot calibrate thresholds safely.")
        
    calib_scores = []
    with torch.no_grad():
        for batch in calib_loader:
            recon, orig_latent, orig_feat = model(batch)
            mem_dist = memory_bank.compute_distance(model.encoder.embedding(batch))
            _, recon_latent, recon_feat = model(recon)
            
            mask = torch.ones_like(batch)
            scores = compute_hybrid_score(batch, recon, orig_feat, recon_feat, orig_latent, recon_latent, mem_dist, mask)
            calib_scores.append(scores["hybrid"])
            
    calib_scores = np.array(calib_scores)
    # 95th percentile of normal calibration error becomes the threshold
    threshold = float(np.percentile(calib_scores, 95))
    print(f"Calculated 95th percentile Calibration Threshold: {threshold:.4f}")
    
    # 4. Create Unified Artifact
    ckpt = {
        'model_state_dict': model.state_dict(),
        'memory_bank_features': memory_bank.bank,
        'ood_mean': ood_detector.global_mean,
        'ood_cov_inv': ood_detector.global_cov_inv,
        'calibration_threshold': threshold,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_version': '1.0.0',
            'model_version': '1.0.0',
            'dataset_info': {
                'train_size': len(train_ds),
                'val_size': len(val_ds),
                'calib_size': len(calib_ds)
            }
        }
    }
    
    out_dir = os.path.join("backend", "app", "models", "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "best_model.pth")
    torch.save(ckpt, out_path)
    print(f"SUCCESS: Unified model artifact saved to {out_path}")

if __name__ == "__main__":
    train_pipeline()
