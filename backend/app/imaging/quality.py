import numpy as np
import cv2

class QualityStatus:
    ACCEPTED = "ACCEPTED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"

def check_image_quality(image: np.ndarray) -> dict:
    """Assess the quality of the image for obvious flaws before ML processing."""
    reasons = []
    status = QualityStatus.ACCEPTED
    
    if image.size == 0:
        return {"status": QualityStatus.REJECTED, "reasons": ["Empty image"]}
        
    h, w = image.shape
    
    # 1. Resolution Check
    if min(h, w) < 256:
        status = QualityStatus.REJECTED
        reasons.append("Resolution too low (minimum 256x256)")
        
    # 2. Aspect Ratio Check (Chest X-Rays are generally roughly square/portrait)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        status = QualityStatus.REJECTED
        reasons.append("Extreme aspect ratio")

    # 3. Exposure / Contrast Check
    # Assuming image is normalized float [0, 1]
    img_mean = np.mean(image)
    if img_mean < 0.05:
        if status != QualityStatus.REJECTED: status = QualityStatus.REVIEW
        reasons.append("Severe underexposure / very dark")
    elif img_mean > 0.95:
        if status != QualityStatus.REJECTED: status = QualityStatus.REVIEW
        reasons.append("Severe overexposure / very bright")
        
    # 4. Blur detection (Variance of Laplacian)
    img_uint8 = (image * 255).astype(np.uint8)
    blur_score = cv2.Laplacian(img_uint8, cv2.CV_64F).var()
    if blur_score < 10.0:  # Threshold needs tuning on real data
        if status != QualityStatus.REJECTED: status = QualityStatus.REVIEW
        reasons.append(f"Potential blur detected (score {blur_score:.1f})")

    return {
        "status": status,
        "reasons": reasons,
        "metrics": {
            "mean_intensity": float(img_mean),
            "blur_score": float(blur_score),
            "dimensions": [h, w]
        }
    }
