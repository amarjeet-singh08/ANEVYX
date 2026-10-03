
import torch, cv2, os, glob
import numpy as np
import sys
sys.path.append('e:/Anevyx ML Project')
from backend.app.models.autoencoder import AdvancedAutoencoder
from backend.app.imaging.preprocessing import preprocess_pipeline
from backend.app.anomaly.scoring import s_pixel, s_structural, s_perceptual, s_latent
from backend.app.models.memory_bank import NormalityMemoryBank

model = AdvancedAutoencoder()
ckpt = torch.load('backend/app/models/weights/best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img_paths = glob.glob('e:/Anevyx ML Project/data/train/normal/img_*.png')
print(f'Found {len(img_paths)} CheXpert training images for calibration.')

calib_tensors = []
for p in img_paths:
    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if raw is None: continue
    proc = preprocess_pipeline(raw, target_size=256)
    calib_tensors.append(torch.from_numpy(proc).unsqueeze(0).unsqueeze(0).float())

embeddings = []
raw_mse, raw_mae, raw_struct, raw_perc, raw_lat = [], [], [], [], []

with torch.no_grad():
    for x in calib_tensors:
        recon, latent, feat = model(x)
        embed = model.encoder.embedding(x)
        embeddings.append(embed.squeeze())
        
        raw_mse.append(s_pixel(x, recon).item())
        raw_mae.append(torch.abs(x - recon).mean().item())
        raw_struct.append(1.0 - s_structural(x, recon).item())
        _, recon_latent, recon_feat = model(recon)
        raw_perc.append(s_perceptual(feat, recon_feat).item())
        raw_lat.append(s_latent(latent, recon_latent).item())

embeddings = torch.stack(embeddings)
mb = NormalityMemoryBank(embed_dim=512)
mb.add_features(embeddings)

raw_mem = []
with torch.no_grad():
    for emb in embeddings:
        d = mb.compute_distance(emb.unsqueeze(0), k=2) 
        raw_mem.append(d.item())

# 1. Component Normalizers: 99th percentile of raw scores
# We use max since dataset is tiny, + 1e-4 to prevent div by zero
normalizers = {
    'mse_max': max(np.max(raw_mse), 1e-4),
    'mae_max': max(np.max(raw_mae), 1e-4),
    'struct_err_max': max(np.max(raw_struct), 1e-4),
    'perc_max': max(np.max(raw_perc), 1e-4),
    'lat_max': max(np.max(raw_lat), 1e-4),
    'mem_max': max(np.max(raw_mem), 1e-4)
}

# 2. Compute Hybrid Scores for Calibration Set using these normalizers
hybrid_scores = []
weights = {'pixel': 1.0, 'structural': 1.0, 'perceptual': 1.0, 'latent': 0.5, 'memory': 1.0}

for i in range(len(calib_tensors)):
    norm_pixel = raw_mse[i] / normalizers['mse_max']
    norm_struct = raw_struct[i] / normalizers['struct_err_max']
    norm_perc = raw_perc[i] / normalizers['perc_max']
    norm_lat = raw_lat[i] / normalizers['lat_max']
    norm_mem = raw_mem[i] / normalizers['mem_max']
    
    hybrid = (
        weights['pixel'] * norm_pixel +
        weights['structural'] * norm_struct +
        weights['perceptual'] * norm_perc +
        weights['latent'] * norm_lat +
        weights['memory'] * norm_mem
    )
    hybrid_scores.append(hybrid)

# Anomaly threshold is the 99th percentile of the Hybrid scores + 10% margin
calibration_threshold = np.percentile(hybrid_scores, 99) * 1.1

# 3. OOD Calibration (Diagonal)
ood_mean = embeddings.mean(dim=0)
ood_var = embeddings.var(dim=0) + 1e-2
cov_inv = 1.0 / ood_var

m_dists = []
for f in embeddings:
    delta = f - ood_mean
    d = torch.sqrt(torch.sum((delta ** 2) * cov_inv, dim=-1)).item()
    m_dists.append(d)
ood_threshold = np.percentile(m_dists, 99) * 1.1

print(f'Normalizers: {normalizers}')
print(f'Anomaly Calibration Threshold: {calibration_threshold:.4f}')
print(f'OOD Threshold: {ood_threshold:.4f}')

ckpt['ood_mean'] = ood_mean
ckpt['ood_cov_inv'] = cov_inv
ckpt['ood_threshold'] = float(ood_threshold)
ckpt['score_normalizers'] = normalizers
ckpt['memory_bank_features'] = mb.bank 
ckpt['calibration_threshold'] = float(calibration_threshold)
torch.save(ckpt, 'backend/app/models/weights/best_model.pth')
print('Recalibration complete.')

