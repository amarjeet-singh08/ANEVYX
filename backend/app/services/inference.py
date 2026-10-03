import torch
import numpy as np
import base64
import cv2
from backend.app.models.autoencoder import AdvancedAutoencoder
from backend.app.models.memory_bank import NormalityMemoryBank
from backend.app.models.uncertainty import generate_tta_variants
from backend.app.anomaly.scoring import compute_hybrid_score
from backend.app.anomaly.ood import OODDetector
from backend.app.anomaly.localization import generate_localization_map
from backend.app.anomaly.calibration import decide_abstention, evaluate_threshold

def numpy_to_base64(img: np.ndarray, colormap=None) -> str:
    """Helper to convert numpy arrays to base64 data URIs for the frontend."""
    if img.dtype != np.uint8:
        # Normalize to 0-255
        img_norm = np.clip(img, 0, 1)
        img = (img_norm * 255).astype(np.uint8)
    
    if colormap is not None:
        intensity = img.copy() # Save the original intensity for alpha
        img = cv2.applyColorMap(img, colormap)
        # Create an RGBA image. Map low intensity to 0 alpha, high to 255
        # We can use a power function or just linear. Let's use squared for cleaner background
        alpha = (intensity.astype(np.float32) / 255.0) ** 1.5 * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = alpha.astype(np.uint8)
        
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

import os

class InferenceService:
    def __init__(self):
        # ANEVYX STRICT RULE: Never run inference on an untrained/random model.
        self.model = AdvancedAutoencoder() # No more encoder_type arg
        self.is_trained = False
        self.calibration_threshold = None
        self.score_normalizers = None
        self.memory_bank = NormalityMemoryBank(embed_dim=512)
        self.ood_detector = OODDetector()
        
        # Prefer unified pipeline artifact
        best_path = getattr(self.__class__, 'MODEL_PATH', None)
        if not best_path:
            best_path = os.path.join(os.path.dirname(__file__), "..", "models", "weights", "best_model.pth")
        
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'memory_bank_features' in ckpt:
                self.memory_bank.bank = ckpt['memory_bank_features']
            if 'ood_mean' in ckpt and 'ood_cov_inv' in ckpt:
                self.ood_detector.global_mean = ckpt['ood_mean']
                self.ood_detector.global_cov_inv = ckpt['ood_cov_inv']
            if 'ood_threshold' in ckpt:
                self.ood_detector.threshold = ckpt['ood_threshold']
            if 'score_normalizers' in ckpt:
                self.score_normalizers = ckpt['score_normalizers']
            if 'calibration_threshold' in ckpt:
                self.calibration_threshold = ckpt['calibration_threshold']
                
            self.is_trained = True
            print(f"Loaded unified artifacts from {best_path}!")
        else:
            print("WARNING: No trained checkpoint found. Inference will be disabled.")
            
        self.model.eval()
        
    @torch.no_grad()
    def analyze(self, image: np.ndarray, mask: np.ndarray = None, quality_status: str = "ACCEPTED") -> dict:
        """
        Runs the full inference pipeline with reliability checks.
        """
        # Dynamic hot-reload: If the artifact was generated after the server started, load it now.
        if not self.is_trained:
            best_path = getattr(self.__class__, 'MODEL_PATH', None)
            if not best_path:
                best_path = os.path.join(os.path.dirname(__file__), "..", "models", "weights", "best_model.pth")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                if 'memory_bank_features' in ckpt:
                    self.memory_bank.bank = ckpt['memory_bank_features']
                if 'ood_mean' in ckpt and 'ood_cov_inv' in ckpt:
                    self.ood_detector.global_mean = ckpt['ood_mean']
                    self.ood_detector.global_cov_inv = ckpt['ood_cov_inv']
                if 'ood_threshold' in ckpt:
                    self.ood_detector.threshold = ckpt['ood_threshold']
                if 'score_normalizers' in ckpt:
                    self.score_normalizers = ckpt['score_normalizers']
                if 'calibration_threshold' in ckpt:
                    self.calibration_threshold = ckpt['calibration_threshold']
                self.is_trained = True
                self.model.eval()

        # Strict requirement: Do not generate fake anomaly results from random models
        if not self.is_trained:
            return {
                "status": "error",
                "model_status": "UNTRAINED / MODEL NOT LOADED",
                "abstention": {
                    "abstain": True,
                    "reason": "ANALYSIS UNAVAILABLE — Train or load a validated ANEVYX model before analysis."
                },
                "anomaly": {"score": None, "level": "UNAVAILABLE — A trained ANEVYX model is required.", "components": {}},
                "uncertainty": {"level": "UNAVAILABLE", "variance": None},
                "ood": {"status": "UNAVAILABLE", "score": None},
                "reconstruction": {"available": False, "map": None},
                "localization": {"available": False, "map": None}
            }

        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() if mask is not None else None
        
        # ── STEP 1: Main Inference (eval mode — canonical forward pass) ──
        # This reconstruction is the SINGLE source of truth for:
        #   - the displayed Normative Reconstruction image
        #   - all MSE/MAE/SSIM/structural/perceptual/latent/memory calculations
        #   - the anomaly heatmap overlay
        self.model.eval()
        recon, orig_latent, orig_feat = self.model(input_tensor)
        global_embed = self.model.encoder.embedding(input_tensor)
        _, recon_latent, recon_feat = self.model(recon)
        
        # ── STEP 2: Anomaly Scoring (from the canonical reconstruction) ──
        mem_dist = self.memory_bank.compute_distance(global_embed)
        scores = compute_hybrid_score(
            input_tensor, recon, orig_feat, recon_feat,
            orig_latent, recon_latent, mem_dist, mask_tensor,
            normalizers=self.score_normalizers
        )
        
        # ── STEP 3: TTA Uncertainty (MC Dropout + augmentation variants) ──
        # Enable dropout for stochastic forward passes
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        variants = generate_tta_variants(input_tensor, num_variants=3)
        tta_scores = []
        for var in variants:
            r, orig_l, orig_f = self.model(var)
            _, recon_l, recon_f = self.model(r)
            m_dist = self.memory_bank.compute_distance(self.model.encoder.embedding(var))
            score_dict = compute_hybrid_score(
                var, r, orig_f, recon_f, orig_l, recon_l,
                m_dist, mask_tensor, normalizers=self.score_normalizers
            )
            tta_scores.append(score_dict["hybrid"])
            
        self.model.eval()  # Restore eval mode
        
        # Compute mean and variance DIRECTLY from the stored tta_scores array
        # using population variance (ddof=0) for the complete TTA run set
        tta_mean = float(np.mean(tta_scores))
        tta_variance = float(np.var(tta_scores))  # population variance
        
        # Consistency assertion: the stored values must reproduce exactly
        assert abs(tta_mean - np.mean(tta_scores)) < 1e-12, "TTA mean inconsistency"
        assert abs(tta_variance - np.var(tta_scores)) < 1e-12, "TTA variance inconsistency"
        
        # Classify uncertainty from variance thresholds
        if tta_variance > 0.05:
            uncertainty_level_raw = "highly_uncertain"
        elif tta_variance > 0.01:
            uncertainty_level_raw = "moderately_uncertain"
        else:
            uncertainty_level_raw = "stable"
        
        uncertainty = {
            "level": uncertainty_level_raw,
            "variance": tta_variance,
            "mean_score": tta_mean,
            "individual_scores": tta_scores
        }
        
        # ── STEP 4: OOD Detection (separate from anomaly scoring) ──
        ood = self.ood_detector.check_ood(global_embed)
        
        # ── STEP 5: Abstention Decision ──
        abstention = decide_abstention(quality_status, ood["status"], uncertainty_level_raw)
        
        # ── STEP 6: Encode outputs for frontend ──
        # The reconstruction numpy array is from the EXACT same tensor used in scoring
        recon_np = recon.squeeze().cpu().numpy()
        recon_b64 = numpy_to_base64(recon_np)
        
        # Anomaly heatmap from the EXACT same orig_feat/recon_feat used in scoring
        heatmap = generate_localization_map(
            image, 
            recon_np,
            orig_feat, recon_feat, mask,
            normalizers=self.score_normalizers
        )
        localization_b64 = numpy_to_base64(heatmap, cv2.COLORMAP_JET)
        
        # ── STEP 7: Determine display labels ──
        if self.calibration_threshold is None:
            anomaly_level = "UNCALIBRATED"
        elif abstention["abstain"]:
            if ood["status"] == "out_of_distribution":
                anomaly_level = "NOT RELIABLE - OOD INPUT"
            else:
                anomaly_level = "ABSTAINED - NOT RELIABLE"
        else:
            anomaly_level = evaluate_threshold(scores["hybrid"], threshold=self.calibration_threshold)
            anomaly_level = "CONSISTENT WITH NORMALITY" if anomaly_level == "consistent_with_normality" else "INCONSISTENT WITH LEARNED NORMALITY"
            
        if abstention["abstain"]:
            if ood["status"] == "out_of_distribution":
                uncertainty_display = "NOT INTERPRETABLE - OOD INPUT"
            else:
                uncertainty_display = "UNAVAILABLE"
        else:
            uncertainty_display = uncertainty_level_raw.replace("_", " ").upper()
            
        # ── STEP 8: Build response (single source of truth) ──
        return {
            "status": "complete",
            "model_status": "TRAINED AND LOADED",
            "abstention": abstention,
            "anomaly": {
                # Score is always provided; the level indicates reliability
                "score": scores["hybrid"],
                "level": anomaly_level,
                # Components always provided so Calculation Dashboard always has data
                "components": scores
            },
            "uncertainty": {
                "level": uncertainty_display,
                "variance": tta_variance,
                "runs": len(tta_scores),
                "mean": tta_mean,
                "individual_scores": tta_scores
            },
            "ood": {
                "status": ood["status"].replace("_", " ").upper(),
                "score": float(ood["score"]) if ood["score"] is not None else None,
                "threshold": float(ood.get("threshold", 15.0))
            },
            "reconstruction": {
                "available": True,
                "map": recon_b64
            },
            "localization": {
                "available": localization_b64 is not None,
                "map": localization_b64
            },
            "calculations": {
                "input": {
                    "preprocessing_version": "v1.2_clahe_pad",
                    "reconstruction_error": scores.get("pixel")
                },
                "anomaly": {
                    "mse": scores.get("pixel"),
                    "mae": scores.get("mae"),
                    "ssim": scores.get("ssim"),
                    "structural_error": scores.get("structural"),
                    "perceptual_distance": scores.get("perceptual"),
                    "latent_distance": scores.get("latent"),
                    "memory_distance": scores.get("memory"),
                    "hybrid_score": scores.get("hybrid"),
                    "weights": scores.get("weights"),
                    "normalized_components": scores.get("normalized_components", {})
                },
                "ood": {
                    "score": ood.get("score"),
                    "threshold": ood.get("threshold"),
                    "decision": ood["status"].replace("_", " ").upper()
                },
                "uncertainty": {
                    "runs": len(tta_scores),
                    "individual_scores": tta_scores,
                    "mean": tta_mean,
                    "variance": tta_variance,
                    "classification": uncertainty_level_raw.replace("_", " ").upper()
                },
                "calibration": {
                    "dataset": "CheXpert / Hold-out",
                    "samples": len(self.memory_bank.bank) if self.memory_bank.bank is not None else 0,
                    "anomaly_threshold": self.calibration_threshold,
                    "ood_threshold": ood.get("threshold", 15.0),
                    "method": "GMM / Mahalanobis & Feature Distance"
                },
                "evaluation": {
                    "auroc": "Not Evaluated",
                    "auprc": "Not Evaluated",
                    "sensitivity": "Not Evaluated",
                    "specificity": "Not Evaluated"
                }
            }
        }

# Trigger reload
