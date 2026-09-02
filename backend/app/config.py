"""Environment-driven configuration. Kept dependency-free (plain os.environ)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

ICU_RISK_THRESHOLD = float(os.environ.get("ICU_RISK_THRESHOLD", "0.5"))
TELEMETRY_RISK_THRESHOLD = float(os.environ.get("TELEMETRY_RISK_THRESHOLD", "0.25"))
