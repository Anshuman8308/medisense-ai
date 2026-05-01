"""
MediSense AI v2 — Centralised Configuration
Reads from environment variables (with safe defaults for development).
Copy .env.example → .env and fill in your values.
"""
import os
from pathlib import Path

# Project root = two levels above this file (backend/config.py → project root)
ROOT    = Path(__file__).resolve().parent.parent
ML_DIR  = ROOT / "ml" / "models"
DATA_DIR = ROOT / "ml" / "data"

class Config:
    # Flask
    SECRET_KEY   = os.getenv("SECRET_KEY", "medisense-dev-secret-change-in-prod")
    DEBUG        = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    PORT         = int(os.getenv("PORT", 5000))

    # CORS — comma-separated list of allowed origins
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

    # Paths
    ML_DIR       = ML_DIR
    DATA_DIR     = DATA_DIR

    # Google Maps (optional — doctor recommendation)
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

    # Claude API (optional — enriched chat)
    ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

    # MongoDB (optional — prediction history, user accounts)
    MONGO_URI    = os.getenv("MONGO_URI", "mongodb://localhost:27017/medisense")

    # Rate limiting
    RATE_LIMIT   = os.getenv("RATE_LIMIT", "60 per minute")

    # PDF report
    PDF_OUTPUT_DIR = ROOT / "reports"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


ENV_MAP = {
    "development": DevelopmentConfig,
    "production":  ProductionConfig,
}

def get_config():
    env = os.getenv("FLASK_ENV", "development")
    return ENV_MAP.get(env, DevelopmentConfig)
