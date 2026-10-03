import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from backend.app.training.datasets import CheXpertAdapter, MedicalImageDataset
from backend.app.models.autoencoder import AdvancedAutoencoder
from backend.app.training.trainer import ModelTrainer
from backend.app.models.memory_bank import NormalityMemoryBank
from backend.app.anomaly.ood import OODDetector
from backend.app.anomaly.scoring import compute_hybrid_score

def run_pipeline(data_dir: str, output_dir: str, epochs: int = 1):
    """
    Executes the complete ANEVYX training pipeline:
    1. Loads dataset and applies patient-level splits.
    2. Filters normal images for training (Rule 5).
    3. Trains the AdvancedAutoencoder.
    4. Generates embeddings for the Memory Bank and OOD Detector.
    5. Calculates the dynamic calibration threshold.
    6. Saves all artifacts to a single unified checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running pipeline on {device}...")

    # 1. Dataset & Splitting
    adapter = CheXpertAdapter()
    
    # Point 9: Do not generate fake datasets when files are missing.
    df = adapter.load_metadata(data_dir)
    print(f"Loaded {len(df)} total records from {data_dir}")
    
    # Rule 6: Patient-level split
    train_df, val_df, test_df = adapter.create_patient_split(df)
    
    # Rule 5: Train ONLY on normal subset
    train_normal_df = adapter.get_normal_subset(train_df)
    print(f"Filtered normal subset: {len(train_normal_df)} / {len(train_df)} training records")
    
    # Datasets
    train_dataset = MedicalImageDataset(train_normal_df, data_dir, adapter)
    val_dataset = MedicalImageDataset(val_df, data_dir, adapter)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 2. Initialization
    model = AdvancedAutoencoder()
    trainer = ModelTrainer(model, device=device, lr=1e-4)
    memory_bank = NormalityMemoryBank(embed_dim=512).to(device)
    ood_detector = OODDetector()
    
    # 3. Training Loop
    print("Starting Model Training...")
    for epoch in range(epochs):
        metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {metrics['loss']:.4f}")
        
    trainer.model.eval()
    
    # 4. Generate Memory Bank & OOD Detector (Rule 9, 13)
    print("Generating Memory Bank and OOD distributions...")
    all_embeddings = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            # Embedding from foundation encoder
            emb = trainer.model.encoder.embedding(images)
            memory_bank.add_features(emb)
            all_embeddings.append(emb.cpu())
            
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        ood_detector.fit(all_embeddings)
        print(f"Memory Bank fitted with {memory_bank.bank.shape[0]} samples.")
        
    # 5. Calibration (Rule 12)
    print("Calibrating Dynamic Threshold on Validation Set...")
    val_scores = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            recons, orig_l, orig_f = trainer.model(images)
            _, recon_l, recon_f = trainer.model(recons)
            
            emb = trainer.model.encoder.embedding(images)
            mem_dist = memory_bank.compute_distance(emb)
            
            # compute score for each item in batch
            for i in range(images.size(0)):
                # If label == 1.0 it is normal. Only use normal images for calibration threshold.
                if labels[i].item() == 1.0:
                    score_dict = compute_hybrid_score(
                        images[i:i+1], recons[i:i+1], 
                        orig_f[i:i+1], recon_f[i:i+1], 
                        orig_l[i:i+1], recon_l[i:i+1], 
                        mem_dist[i:i+1] if mem_dist is not None else None
                    )
                    val_scores.append(score_dict["hybrid"])
                    
    if val_scores:
        # Use 95th percentile of normal reconstruction scores as threshold
        calibration_threshold = float(np.percentile(val_scores, 95))
        print(f"Calibrated Threshold (95th Pctile): {calibration_threshold:.4f}")
    else:
        calibration_threshold = 0.5
        print("Fallback threshold used (no normal val data).")
        
    # 6. Save Unified Artifact
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'memory_bank': memory_bank.bank,
        'ood_mean': ood_detector.global_mean,
        'ood_cov_inv': ood_detector.global_cov_inv,
        'calibration_threshold': calibration_threshold,
        'embedding_dim': 512,
        'encoder_type': 'resnet18_ssl'
    }, checkpoint_path)
    
    print(f"Pipeline complete! Artifact saved to {checkpoint_path}")

if __name__ == "__main__":
    run_pipeline(
        data_dir="data/chexpert",
        output_dir="backend/app/models/weights",
        epochs=2
    )
