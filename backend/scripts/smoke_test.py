import os
import sys
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.imaging.preprocessing import preprocess_pipeline
from backend.app.imaging.quality import check_image_quality
from backend.app.imaging.anatomy import create_thoracic_mask
from backend.app.models.autoencoder import BaselineAutoencoder
from backend.app.anomaly.reconstruction import calculate_reconstruction_heatmap

def run_smoke_test():
    print("Starting Phase 3 Smoke Test...")
    
    # 1. Create a dummy test image if not exists
    test_img_path = 'test_scan.png'
    if not os.path.exists(test_img_path):
        img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        Image.fromarray(img).save(test_img_path)
        print(f"Created dummy image {test_img_path}")
        
    # 2. Preprocessing
    print("Running preprocessing pipeline...")
    raw_img = np.array(Image.open(test_img_path).convert("L"))
    processed_img = preprocess_pipeline(raw_img, target_size=256)
    
    quality = check_image_quality(processed_img)
    print(f"Quality check: {quality['status']}")
    
    mask = create_thoracic_mask(processed_img)
    print(f"Mask generated with sum: {mask.sum()}")
    
    # 3. Model Inference
    print("Initializing Baseline Autoencoder...")
    model = BaselineAutoencoder(in_channels=1, latent_dim=128)
    model.eval()
    
    # Prepare tensor: (B, C, H, W)
    input_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).float()
    
    print("Running inference...")
    with torch.no_grad():
        recon_tensor, latent = model(input_tensor)
        
    recon_img = recon_tensor.squeeze().numpy()
    
    # 4. Anomaly Heatmap
    print("Calculating anomaly heatmap...")
    heatmap = calculate_reconstruction_heatmap(processed_img, recon_img, mask)
    
    anomaly_score = float(heatmap.mean())
    print(f"Mean anomaly score in valid region: {anomaly_score:.4f}")
    
    # 5. Validate shapes
    assert processed_img.shape == (256, 256), "Input shape mismatch"
    assert recon_img.shape == (256, 256), "Reconstruction shape mismatch"
    assert heatmap.shape == (256, 256), "Heatmap shape mismatch"
    
    print("\n[PASS] SMOKE TEST PASSED: Pipeline is working end-to-end.")
    
if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as e:
        print(f"\n[FAIL] SMOKE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
