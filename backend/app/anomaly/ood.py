import torch

class OODDetector:
    """
    Detects if an input image falls far outside the training distribution, 
    independent of whether it is 'anomalous' (diseased).
    """
    def __init__(self, threshold: float = 15.0):
        # Mahalanobis distance threshold or similar
        self.threshold = threshold
        self.global_mean = None
        self.global_cov_inv = None

    def fit(self, features: torch.Tensor):
        """Fit the distribution of training features."""
        self.global_mean = features.mean(dim=0)
        # Simplified: using diagonal covariance (variance) for stability in high dims
        var = features.var(dim=0) + 1e-2
        self.global_cov_inv = 1.0 / var

    def check_ood(self, feature: torch.Tensor) -> dict:
        """
        Check if a single feature vector is OOD.
        Returns status and score.
        """
        if self.global_mean is None:
            # Rule 13: If OOD is not fitted, strictly report unavailable.
            return {"status": "unavailable", "distance": None}
            
        # Simplified Mahalanobis distance
        diff = feature - self.global_mean
        dist = torch.sqrt(torch.sum((diff ** 2) * self.global_cov_inv, dim=-1)).mean()
        
        score = dist.item()
        
        # Calculate OOD Index (score / threshold) so threshold is visually 1.0
        ood_index = score / self.threshold if self.threshold > 0 else score
        
        if score > self.threshold:
            status = "out_of_distribution"
        elif score > self.threshold * 0.7:
            status = "review_recommended"
        else:
            status = "within_validated_range"
            
        return {
            "status": status,
            "score": ood_index,
            "raw_score": score,
            "threshold": 1.0
        }
