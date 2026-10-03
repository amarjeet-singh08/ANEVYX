from fastapi import APIRouter

router = APIRouter()

@router.get("/system-health")
async def system_health():
    return {
        "status": "online",
        "processing_mode": "remote" if False else "local",  # TODO hook to settings
        "message": "ANEVYX core operational"
    }

@router.get("/model-health")
async def model_health():
    return {
        "status": "unknown",
        "message": "No model loaded"
    }
