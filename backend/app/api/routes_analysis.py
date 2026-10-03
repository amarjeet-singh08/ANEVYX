from fastapi import APIRouter, UploadFile, File, HTTPException
import traceback
from backend.app.imaging.dicom import extract_image_array
from backend.app.imaging.preprocessing import preprocess_pipeline, normalize_grayscale
from backend.app.imaging.quality import check_image_quality
from backend.app.imaging.anatomy import create_thoracic_mask
from backend.app.services.inference import InferenceService
from backend.app.services.llm_report import ClinicalReportService
import asyncio

router = APIRouter()
inference_service = InferenceService()
llm_service = ClinicalReportService()

@router.post("/analyze")
async def analyze_scan(file: UploadFile = File(...)):
    """
    Phase 2 Integration Test: 
    Accepts an image, extracts array, runs quality check, preprocesses, and masks.
    Returns status json.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    try:
        file_bytes = await file.read()
        
        # Point 14: Security limit file size (10MB max)
        if len(file_bytes) > 10 * 1024 * 1024:
            raise ValueError("File exceeds maximum allowed size of 10MB")
            
        # 1. Extraction
        raw_pixels = extract_image_array(file_bytes)
        
        # 2. Quality Gate (on normalized but unpadded image)
        norm_img = normalize_grayscale(raw_pixels)
        quality = check_image_quality(norm_img)
        
        # 3. Preprocessing
        processed_img = preprocess_pipeline(raw_pixels, target_size=256) # 256 for demo speed
        
        # 4. Anatomy Mask
        mask = create_thoracic_mask(processed_img)
        
        # 5. Full Inference Pipeline
        inference_result = inference_service.analyze(processed_img, mask, quality["status"])
        
        # 6. AI Clinical Report Generation (Non-blocking)
        clinical_report = await asyncio.to_thread(llm_service.generate_summary, inference_result)
        inference_result["clinical_report"] = clinical_report
        
        from backend.app.services.inference import numpy_to_base64
        processed_b64 = numpy_to_base64(processed_img)
        
        return {
            "status": "success",
            "quality": quality,
            "pipeline": {
                "original_shape": list(raw_pixels.shape),
                "processed_shape": list(processed_img.shape),
                "mask_sum": float(mask.sum()),
                "processed_b64": processed_b64
            },
            "inference": inference_result,
            "message": "Analysis complete"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during analysis")
