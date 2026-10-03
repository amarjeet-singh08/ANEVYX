import numpy as np
import cv2

def create_thoracic_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a valid anatomy mask to ignore background and corners.
    Uses simple Otsu thresholding and morphological operations via cv2.
    Assumes image is preprocessed (0-1 float, 512x512).
    """
    img_8u = (image * 255).astype(np.uint8)
    
    # 1. Basic threshold to separate body from background
    try:
        _, binary = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        _, binary = cv2.threshold(img_8u, 50, 255, cv2.THRESH_BINARY)
        
    # 2. Morphological closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 3. Find largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    if num_labels <= 1:
        return np.ones_like(image, dtype=np.float32)
        
    # The first component is usually background, find the largest non-background
    largest_label = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i
            
    mask = (labels == largest_label).astype(np.float32)
    
    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

