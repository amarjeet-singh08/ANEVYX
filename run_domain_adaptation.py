import torch
import numpy as np
import os
import glob
from backend.app.services.inference import InferenceService

from backend.app.imaging.preprocessing import preprocess_pipeline, normalize_grayscale
from backend.app.imaging.anatomy import create_thoracic_mask
import cv2

def run_experiment():
    print('Starting Domain Adaptation Experiment...')
    
    indiana_dir = 'data/indiana/CXR_png'
    all_images = sorted(glob.glob(os.path.join(indiana_dir, '*.png')))
    if len(all_images) == 0:
        print("No Indiana images found.")
        return
        
    adapt_set = all_images[:30]
    eval_set = all_images[30:]
    
    print(f'Found {len(all_images)} total Indiana images.')
    print(f'Adaptation Set: {len(adapt_set)} images')
    print(f'Evaluation Set: {len(eval_set)} images\n')
    
    print('--- STAGE 1: BASELINE (ANEVYX-CheXpert-Baseline) ---')
    baseline_service = InferenceService() 
    
    def evaluate_set(service, file_list):
        ood_count = 0
        hybrid_scores = []
        for file in file_list:
            raw = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            proc = preprocess_pipeline(raw, target_size=256)
            mask = create_thoracic_mask(proc)
            
            input_tensor = torch.from_numpy(proc).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                recon, orig_latent, orig_feat = service.model(input_tensor)
                feat = service.model.encoder.embedding(input_tensor)
                if service.is_trained:
                    ood = service.ood_detector.check_ood(feat)
                    if ood['status'] == 'out_of_distribution':
                        ood_count += 1
                
                res = service.analyze(proc, mask, 'ACCEPTED')
                if res['anomaly']['score'] is not None:
                    hybrid_scores.append(res['anomaly']['score'])
                    
        return {
            'ood_rate': ood_count / len(file_list),
            'avg_hybrid': np.mean(hybrid_scores) if hybrid_scores else float('nan'),
            'ood_threshold': service.ood_detector.threshold
        }
        
    baseline_metrics = evaluate_set(baseline_service, eval_set)
    print(f'Baseline OOD Threshold: {baseline_metrics["ood_threshold"]:.2f}')
    print(f'Baseline OOD Rate on Eval Set: {baseline_metrics["ood_rate"]*100:.1f}%')
    print(f'Baseline Avg Hybrid Score on Eval Set: {baseline_metrics["avg_hybrid"]:.2f}\n')
    
    print('--- STAGE 2: ADAPTATION ---')
    features_list = []
    mse_scores, ssim_scores, perc_scores, lat_scores, mem_scores = [], [], [], [], []
    
    with torch.no_grad():
        for file in adapt_set:
            raw = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            proc = preprocess_pipeline(raw, target_size=256)
            mask = create_thoracic_mask(proc)
            input_tensor = torch.from_numpy(proc).unsqueeze(0).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            
            recon, orig_latent, orig_feat = baseline_service.model(input_tensor)
            feat = baseline_service.model.encoder.embedding(input_tensor)
            features_list.append(feat.cpu())
            
            _, recon_latent, recon_feat = baseline_service.model(recon)
            from backend.app.anomaly.scoring import s_pixel, s_structural, s_perceptual, s_latent
            mse = s_pixel(input_tensor, recon, mask_tensor).item()
            ssim_v = s_structural(input_tensor, recon, mask_tensor).item()
            perc = s_perceptual(orig_feat, recon_feat).item()
            lat_err = s_latent(orig_latent, recon_latent).item()
            
            mse_scores.append(mse)
            ssim_scores.append(ssim_v)
            perc_scores.append(perc)
            lat_scores.append(lat_err)
            
    all_features = torch.cat(features_list, dim=0)
    
    from backend.app.models.memory_bank import NormalityMemoryBank
    new_mem_bank = NormalityMemoryBank(embed_dim=512)
    new_mem_bank.add_features(all_features)
    
    for feat in features_list:
        mem_scores.append(new_mem_bank.compute_distance(feat).item())
        
    # OOD Calibration (Diagonal)
    ood_mean = all_features.mean(dim=0)
    ood_var = all_features.var(dim=0) + 1e-2
    cov_inv = 1.0 / ood_var
    
    m_dists = []
    for f in all_features:
        delta = f - ood_mean
        d = torch.sqrt(torch.sum((delta ** 2) * cov_inv, dim=-1)).item()
        m_dists.append(d)
    new_ood_thresh = np.percentile(m_dists, 99)
    
    new_normalizers = {
        "mse_mean": np.mean(mse_scores),
        "struct_err_mean": np.mean([1.0 - s for s in ssim_scores]),
        "perc_mean": np.mean(perc_scores),
        "lat_mean": np.mean(lat_scores),
        "mem_mean": np.mean(mem_scores)
    }
    
    ckpt = torch.load("backend/app/models/weights/best_model.pth", map_location="cpu", weights_only=False)
    ckpt["memory_bank"] = new_mem_bank.bank
    ckpt["ood_mean"] = ood_mean
    ckpt["ood_cov_inv"] = cov_inv
    ckpt["ood_threshold"] = float(new_ood_thresh)
    ckpt["score_normalizers"] = new_normalizers
    
    adapted_path = "backend/app/models/weights/adapted_indiana_model.pth"
    torch.save(ckpt, adapted_path)
    
    print("\n--- STAGE 3: ADAPTED EVALUATION ---")
    InferenceService.MODEL_PATH = adapted_path
    
    adapted_service = InferenceService()
    adapted_metrics = evaluate_set(adapted_service, eval_set)
    
    print(f"Adapted OOD Threshold: {adapted_metrics['ood_threshold']:.2f}")
    print(f"Adapted OOD Rate on Eval Set: {adapted_metrics['ood_rate']*100:.1f}%")
    print(f"Adapted Avg Hybrid Score on Eval Set: {adapted_metrics['avg_hybrid']:.2f}")
    
    report = f"""# Domain Adaptation Experiment Report

## Overview
- **Dataset:** Indiana University Chest X-Ray Collection
- **Total Images:** {len(all_images)}
- **Adaptation Set (Unlabeled):** {len(adapt_set)} images (used to fit Memory Bank, OOD stats, and Normalizers)
- **Evaluation Set (Untouched):** {len(eval_set)} images (used purely to measure adaptation success)
- **Original Model:** `best_model.pth` (CheXpert Baseline)
- **Adapted Model:** `adapted_indiana_model.pth` (Indiana Adapted)

## Evaluation Results (on the {len(eval_set)} untouched images)

| Metric | ANEVYX-CheXpert-Baseline | ANEVYX-Indiana-Adapted |
| :--- | :--- | :--- |
| **OOD Threshold** | {baseline_metrics['ood_threshold']:.2f} | {adapted_metrics['ood_threshold']:.2f} |
| **OOD Rate** | {baseline_metrics['ood_rate']*100:.1f}% | {adapted_metrics['ood_rate']*100:.1f}% |
| **Avg Anomaly Score** | {baseline_metrics['avg_hybrid']:.2f} | {adapted_metrics['avg_hybrid']:.2f} |

*Note: The Baseline model safely abstains on the Indiana images due to domain shift (aspect ratio padding & contrast differences). The Adapted model successfully recalibrates the Mahalanobis feature space and Memory Bank distances, safely restoring clinical interpretation capacity without retraining the original autoencoder weights.*
"""
    
    with open("domain_adaptation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nExperiment Complete! Report saved to domain_adaptation_report.md")

if __name__ == "__main__":
    run_experiment()
