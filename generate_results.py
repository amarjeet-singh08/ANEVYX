import os
import sys
import asyncio
import io
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from backend.app.services.inference import InferenceService
from backend.app.imaging.preprocessing import preprocess_pipeline
from PIL import Image
import base64
import json

async def run_evaluation():
    print("Loading inference service...")
    InferenceService.MODEL_PATH = "backend/app/models/weights/best_model.pth"
    service = InferenceService()
    
    # Setup results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Images to test
    desktop_dir = Path(r"C:\Users\ANKUR THAKUR\OneDrive\Desktop")
    
    test_images = {
        "Indiana_Normal_1": "data/test/normal/982_IM-2470-1001.png",
        "Indiana_Abnormal_1": "data/test/abnormal/990_IM-2476-1001.png",
        "Unknown_Desktop_1": str(desktop_dir / "IMG_3459.JPG"),
        "Unknown_Desktop_2": str(desktop_dir / "IMG_6425.PNG"),
        "Unknown_Desktop_3": str(desktop_dir / "IMG_7229.PNG"),
        "Unknown_Desktop_4": str(desktop_dir / "WhatsApp Image 2026-09-29 at 12.47.54 PM.jpeg")
    }
    
    report_lines = [
        "# ANEVYX Inference Results",
        "",
        "This report contains the evaluation results of the ANEVYX domain-adapted model on both the Indiana dataset and unknown chest X-ray images downloaded from the web.",
        "",
        "| Image | OOD Status | Anomaly Level | Hybrid Score | Uncertainty |",
        "|-------|------------|---------------|--------------|-------------|"
    ]
    
    for name, img_path in test_images.items():
        if not os.path.exists(img_path):
            print(f"Skipping {name}, file not found: {img_path}")
            continue
            
        print(f"Analyzing {name}...")
        # Load and read image bytes
        with open(img_path, "rb") as f:
            img_bytes = f.read()
            
        import numpy as np
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
        tensor_img = preprocess_pipeline(np.array(pil_img), target_size=256)
        
        res = service.analyze(tensor_img)
        print(f"Res keys: {res.keys()}")
        print(f"Res OOD: {res.get('ood')}")
        
        # Extract data
        ood_status = res["ood"]["status"]
        ood_score = res["ood"]["score"]
        anom_level = res["anomaly"]["level"]
        hybrid = res["anomaly"]["score"]
        uncert = res["uncertainty"]["level"]
        
        # Save base64 images for the report
        def save_b64(b64_str, filename):
            if not b64_str: return
            header, encoded = b64_str.split(",", 1)
            data = base64.b64decode(encoded)
            with open(results_dir / filename, "wb") as f:
                f.write(data)
        
        if res["reconstruction"]["available"]:
            save_b64(res["reconstruction"]["map"], f"{name}_recon.png")
        if res["localization"]["available"]:
            save_b64(res["localization"]["map"], f"{name}_heatmap.png")
            
        if hybrid is None:
            hybrid_str = "N/A"
        else:
            hybrid_str = f"{hybrid:.4f}"
            
        ood_str = f"{ood_score:.2f}x" if ood_score is not None else "N/A"
            
        report_lines.append(f"| {name} | {ood_status} ({ood_str}) | {anom_level} | {hybrid_str} | {uncert} |")
            
    with open("RESULTS.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print("Done! Results saved to RESULTS.md and images saved to results/")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
