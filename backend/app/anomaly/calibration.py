def decide_abstention(quality_status: str, ood_status: str, uncertainty_level: str) -> dict:
    """
    Decides whether the model should abstain from providing an anomaly decision
    based on quality, OOD, and uncertainty.
    """
    if quality_status == "REJECTED":
        return {
            "abstain": True,
            "reason": "Image quality rejected. Human review required."
        }
        
    if ood_status == "out_of_distribution":
        return {
            "abstain": True,
            "reason": "ABSTAINED - INPUT IS OUTSIDE THE VALIDATED DISTRIBUTION. HUMAN REVIEW REQUIRED."
        }
        
    if uncertainty_level == "highly_uncertain":
        return {
            "abstain": True,
            "reason": "Model prediction is highly unstable. Human review required."
        }
        
    return {
        "abstain": False,
        "reason": None
    }

def evaluate_threshold(score: float, threshold: float = 0.5) -> str:
    """
    Returns the severity level based on a calibrated threshold.
    """
    if score > threshold * 1.5:
        return "high_deviation"
    elif score > threshold:
        return "suspicious_deviation"
    else:
        return "consistent_with_normality"
