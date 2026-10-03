import numpy as np
import cv2

def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Scale pixel intensities to [0, 1] as float32."""
    img_float = image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val > min_val:
        img_float = (img_float - min_val) / (max_val - min_val)
    else:
        img_float = np.zeros_like(img_float)
    return img_float

def equalize_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    # Requires uint16 or uint8. Assuming input is float [0, 1] from normalize_grayscale
    img_uint16 = (image * 65535).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_uint16)
    return normalize_grayscale(img_eq)

def resize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Resize image to target square dimensions, preserving aspect ratio by padding."""
    h, w = image.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to make it exactly target_size x target_size
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    padded = np.pad(
        resized, 
        ((pad_h, target_size - new_h - pad_h), (pad_w, target_size - new_w - pad_w)),
        mode='constant',
        constant_values=0
    )
    return padded

def preprocess_pipeline(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Full preprocessing pipeline for CXR."""
    img = normalize_grayscale(image)
    img = equalize_contrast(img)
    img = resize_image(img, target_size)
    return img
