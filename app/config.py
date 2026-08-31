"""Environment-driven configuration. Kept dependency-free (plain os.environ)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("AHOP_DATA_DIR", str(BASE_DIR / "data")))
MODELS_DIR = Path(os.environ.get("AHOP_MODELS_DIR", str(BASE_DIR / "models")))
REPORTS_DIR = Path(os.environ.get("AHOP_REPORTS_DIR", str(BASE_DIR / "reports")))

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR / 'ahop.db'}")

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

ICU_RISK_THRESHOLD = float(os.environ.get("ICU_RISK_THRESHOLD", "0.5"))
TELEMETRY_RISK_THRESHOLD = float(os.environ.get("TELEMETRY_RISK_THRESHOLD", "0.25"))
SOLVER_TIME_BUDGET_SEC = float(os.environ.get("SOLVER_TIME_BUDGET_SEC", "2.0"))
WAITLIST_LOOKBACK_HOURS = int(os.environ.get("WAITLIST_LOOKBACK_HOURS", "48"))
