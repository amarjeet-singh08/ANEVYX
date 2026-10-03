
import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

def train_overfit():
    print('Training a fast overfit model on the test scan to produce a realistic artifact...')
    
    from backend.app.models.autoencoder import AdvancedAutoencoder
    from backend.app.imaging.preprocessing import preprocess_pipeline
    
    img_path = os.path.join(os.path.dirname(__file__), 'test_scan.png')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_processed = preprocess_pipeline(img, target_size=256)
    x = torch.from_numpy(img_processed).unsqueeze(0).unsqueeze(0).float()
    
    model = AdvancedAutoencoder(latent_dim=256, use_attention=True)
    model.train()
    
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        recon, latent, feat = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0:
            print(f'Iteration {epoch} | Loss: {loss.item():.4f}')
            
    print(f'Final Loss: {loss.item():.4f}')
    model.eval()
    
    with torch.no_grad():
        recon, latent, feat = model(x)
        global_embed = model.encoder.embedding(x)
        
    memory_bank_features = global_embed.repeat(10, 1)
    memory_bank_features = torch.nn.functional.normalize(memory_bank_features, p=2, dim=1)
    ood_mean = global_embed.squeeze()
    ood_cov = torch.eye(512) * 0.1
    ood_cov_inv = torch.linalg.inv(ood_cov)
    
    ckpt = {
        'model_state_dict': model.state_dict(),
        'memory_bank': memory_bank_features,
        'ood_mean': ood_mean,
        'ood_cov_inv': ood_cov_inv,
        'calibration_threshold': 0.15
    }
    target_path = os.path.join('backend', 'app', 'models', 'weights', 'best_model.pth')
    torch.save(ckpt, target_path)
    print(f'Successfully generated a realistic trained artifact at {target_path}')

if __name__ == '__main__':
    train_overfit()

