import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = os.getenv("APP_ENV", "development")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", 8000))
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    data_dir: str = os.getenv("DATA_DIR", "./data")
    allow_remote_processing: bool = os.getenv("ALLOW_REMOTE_PROCESSING", "false").lower() == "true"
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    openrouter_api_key: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
