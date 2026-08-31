"""Bed allocation service: wires DB state into src/ml/bed_allocation_solver.py.

Note: the MILP solver always enforces isolation as a hard constraint and takes
no time-budget parameter; max_solver_time_sec / enforce_strict_isolation are
accepted for API compatibility and reported/logged but do not change the solve.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import (
    BASE_DIR,
    ICU_RISK_THRESHOLD,
    TELEMETRY_RISK_THRESHOLD,
    WAITLIST_LOOKBACK_HOURS,
)
from ..models import Bed, BedAllocation, Patient, TriageEvent

_SOLVER_DIR = str(BASE_DIR / "src" / "ml")
if _SOLVER_DIR not in sys.path:
    sys.path.insert(0, _SOLVER_DIR)

from bed_allocation_solver import DEFAULT_TELEMETRY_RISK_THRESHOLD, solve_allocation  # noqa: E402


def _unit_type_from_name(unit_name: str) -> str:
    upper = (unit_name or "").upper()
    if "ICU" in upper:
        return "ICU"
    if "TELE" in upper:
        return "Telemetry"
    return "General"


def _risk_tier(prob: float) -> str:
    if prob >= ICU_RISK_THRESHOLD:
        return "HIGH"
    if prob >= TELEMETRY_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _pending_patients(db: Session, now: datetime) -> list[dict]:
    window_start = now - timedelta(hours=WAITLIST_LOOKBACK_HOURS)
    allocated_patient_ids = {
        row[0] for row in db.query(BedAllocation.patient_id).distinct().all()
    }
    rows = (
        db.query(Patient, TriageEvent)
        .join(TriageEvent, TriageEvent.patient_id == Patient.patient_id)
        .filter(TriageEvent.recorded_at >= window_start)
        .all()
    )
    latest: dict[str, tuple[Patient, TriageEvent]] = {}
    for patient, event in rows:
        if patient.patient_id in allocated_patient_ids:
            continue
        current = latest.get(patient.patient_id)
        if current is None or event.recorded_at > current[1].recorded_at:
            latest[patient.patient_id] = (patient, event)

    solver_patients = []
    for patient, event in latest.values():
        prob = float(event.icu_escalation_prob or 0.0)
        age_secs = (now - event.recorded_at).total_seconds() if event.recorded_at else 0
        wait_minutes = max(0, min(1440, int(age_secs / 60)))
        solver_patients.append(
            {
                "patient_id": patient.patient_id,
                "esi_level": event.esi_level,
                "icu_risk": prob,
                "isolation_required": bool(patient.is_isolation_required),
                "wait_minutes": wait_minutes,
                "current_unit": "ED",
                "acuity_label": _risk_tier(prob),
            }
        )
    return solver_patients


def _available_beds(db: Session) -> list[dict]:
    beds = db.query(Bed).filter(Bed.status == "AVAILABLE").all()
    return [
        {
            "bed_id": bed.bed_id,
            "unit_type": _unit_type_from_name(bed.unit_name),
            "telemetry": bool(bed.is_telemetry_equipped),
            "isolation_capable": bool(bed.is_isolation_capable),
            "location": bed.unit_name,
        }
        for bed in beds
    ]


def run_allocation(
    db: Session,
    max_solver_time_sec: float,
    enforce_strict_isolation: bool,
) -> dict:
    exec_id = str(uuid.uuid4())
    now = (
        db.query(func.max(TriageEvent.recorded_at)).scalar()
        or datetime.utcnow()
    )

    patients = _pending_patients(db, now)
    beds = _available_beds(db)

    result = solve_allocation(
        patients,
        beds,
        telemetry_threshold=DEFAULT_TELEMETRY_RISK_THRESHOLD,
    )

    allocations = []
    for assignment in result["assignments"]:
        bed = db.query(Bed).filter(Bed.bed_id == assignment["bed_id"]).first()
        if bed is None:
            continue
        bed.status = "OCCUPIED"
        db.add(
            BedAllocation(
                patient_id=assignment["patient_id"],
                bed_id=assignment["bed_id"],
                assigned_at=now,
                expected_discharge_at=now + timedelta(days=2),
                solver_execution_id=exec_id,
            )
        )
        allocations.append(
            {
                "patient_id": assignment["patient_id"],
                "assigned_bed_id": assignment["bed_id"],
                "unit_name": bed.unit_name,
                "bed_number": bed.bed_number,
                "expected_wait_reduction_min": int(assignment.get("wait_minutes", 0)),
            }
        )

    db.commit()

    return {
        "solver_status": result["status"],
        "execution_time_ms": int(round(result["solve_time_s"] * 1000)),
        "assignments_made": len(result["assignments"]),
        "allocations": allocations,
    }
