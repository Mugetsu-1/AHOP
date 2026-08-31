"""ICU-escalation risk scoring against the trained XGBoost model.

Feature order must match src/ml/forecasting_and_risk.py exactly.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import xgboost as xgb

from ..config import (
    ICU_RISK_THRESHOLD,
    MODELS_DIR,
    TELEMETRY_RISK_THRESHOLD,
)

FEATURE_COLS = [
    "age",
    "gender_enc",
    "esi_level",
    "chief_complaint_enc",
    "comorbidity_index",
    "heart_rate",
    "sys_bp",
    "dia_bp",
    "spo2",
    "temp_c",
    "lactate",
    "ed_wait_time_min",
    "is_isolation_required",
    "arrival_hour",
    "day_of_week",
    "is_surge_arrival",
]

_CHIEF_COMPLAINT_KEYWORDS = {
    "Cardiovascular": ["chest", "cardiac", "palpitation", "heart", "angina"],
    "Respiratory": ["breath", "respir", "asthma", "cough", "wheeze", "pneumonia"],
    "Trauma": ["trauma", "fall", "injury", "fracture", "laceration", "burn", "accident"],
    "Gastrointestinal": ["abdominal", "abdomen", "nausea", "vomit", "diarrhea", "gi", "stomach"],
}


def _categorize_chief_complaint(text: str) -> str:
    lowered = (text or "").lower()
    for category, keywords in _CHIEF_COMPLAINT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "General"


_ENC: dict | None = None
_BOOSTER: xgb.Booster | None = None


def _encoders() -> dict:
    global _ENC
    if _ENC is None:
        with open(MODELS_DIR / "encoders.json", encoding="utf-8") as fh:
            _ENC = json.load(fh)
    return _ENC


def _booster() -> xgb.Booster:
    global _BOOSTER
    if _BOOSTER is None:
        _BOOSTER = xgb.Booster()
        _BOOSTER.load_model(str(MODELS_DIR / "xgboost_icu.json"))
    return _BOOSTER


def _encode_categorical(value, classes) -> int:
    try:
        return int(classes.index(value))
    except ValueError:
        return 0


def score_frame(df) -> np.ndarray:
    """Batch-score a DataFrame that already contains every FEATURE_COLS column."""
    enc = _encoders()
    df = df.copy()
    gender_map = {cls: i for i, cls in enumerate(enc["gender"]["classes"])}
    cc_map = {cls: i for i, cls in enumerate(enc["chief_complaint"]["classes"])}
    df["gender_enc"] = df["gender"].map(gender_map).fillna(0).astype(int)
    df["chief_complaint_enc"] = (
        df["chief_complaint_category"].map(cc_map).fillna(0).astype(int)
    )
    dmat = xgb.DMatrix(df[FEATURE_COLS], feature_names=FEATURE_COLS)
    return _booster().predict(dmat)


def predict_icu_risk(request) -> dict:
    """Score a single TriageAssessRequest and return the API response payload."""
    now = datetime.now()
    arrival_hour = request.arrival_hour if request.arrival_hour is not None else now.hour
    day_of_week = request.day_of_week if request.day_of_week is not None else now.weekday()

    for field in ("heart_rate", "sys_bp", "dia_bp", "spo2", "temp_c"):
        if getattr(request, field) is None:
            raise ValueError(f"missing vital sign: {field}")

    enc = _encoders()
    gender_enc = _encode_categorical(request.gender, enc["gender"]["classes"])
    cc = request.chief_complaint_category or _categorize_chief_complaint(
        request.chief_complaint
    )
    cc_enc = _encode_categorical(cc, enc["chief_complaint"]["classes"])

    features = [
        request.age,
        gender_enc,
        request.esi_level,
        cc_enc,
        request.comorbidity_index,
        request.heart_rate,
        request.sys_bp,
        request.dia_bp,
        request.spo2,
        request.temp_c,
        request.lactate if request.lactate is not None else 0.0,
        request.ed_wait_time_min if request.ed_wait_time_min is not None else 0,
        int(request.is_isolation_required),
        arrival_hour,
        day_of_week,
        int(request.is_surge_arrival),
    ]
    prob = float(
        _booster().predict(
            xgb.DMatrix(np.array([features]), feature_names=FEATURE_COLS)
        )[0]
    )

    importance = _booster().get_score(importance_type="gain")
    total = sum(importance.values()) or 1.0
    impacts = []
    for feat, gain in sorted(importance.items(), key=lambda kv: kv[1], reverse=True):
        if feat.startswith("f") and feat[1:].isdigit():
            feature_name = FEATURE_COLS[int(feat[1:])]
        elif feat in FEATURE_COLS:
            feature_name = feat
        else:
            feature_name = feat
        impacts.append({"feature": feature_name, "impact": round(gain / total, 4)})

    if prob >= ICU_RISK_THRESHOLD:
        category, unit = "HIGH_RISK", "ICU"
    elif prob >= TELEMETRY_RISK_THRESHOLD:
        category, unit = "MEDIUM_RISK", "TELEMETRY"
    else:
        category, unit = "LOW_RISK", "GENERAL"

    return {
        "icu_escalation_probability": round(prob, 4),
        "risk_category": category,
        "recommended_unit": unit,
        "shap_factors": impacts[:5],
    }
