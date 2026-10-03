from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_analysis import router as analysis_router
from backend.app.api.routes_research import router as research_router

app = FastAPI(
    title="ANEVYX",
    description="Imaging Intelligence for Unseen Anomalies",
    version="1.0.0"
)

# Set up CORS
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(research_router, prefix="/api/v1/research")

@app.get("/")
def read_root():
    return {"message": "Welcome to ANEVYX API"}
