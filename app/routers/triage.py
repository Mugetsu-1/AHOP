"""Triage assessment endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Patient, TriageEvent
from ..schemas import TriageAssessRequest, TriageAssessResponse
from ..services.icu_risk import predict_icu_risk

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/assess", response_model=TriageAssessResponse)
def assess_triage(payload: TriageAssessRequest, db: Session = Depends(get_db)):
    try:
        result = predict_icu_risk(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    patient_id = payload.patient_id or str(uuid.uuid4())
    now = datetime.utcnow()

    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if patient is None:
        patient = Patient(
            patient_id=patient_id,
            age=payload.age,
            gender=payload.gender,
            is_isolation_required=payload.is_isolation_required,
        )
        db.add(patient)
        db.flush()

    db.add(
        TriageEvent(
            patient_id=patient_id,
            esi_level=payload.esi_level,
            chief_complaint=payload.chief_complaint,
            heart_rate=payload.heart_rate,
            sys_bp=payload.sys_bp,
            dia_bp=payload.dia_bp,
            spo2=payload.spo2,
            temp_c=payload.temp_c,
            icu_escalation_prob=result["icu_escalation_probability"],
            recorded_at=now,
        )
    )
    db.commit()

    return TriageAssessResponse(
        patient_id=patient_id,
        icu_escalation_probability=result["icu_escalation_probability"],
        risk_category=result["risk_category"],
        recommended_unit=result["recommended_unit"],
        shap_factors=result["shap_factors"],
    )
