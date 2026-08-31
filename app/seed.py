"""Populate the SQLite DB with the clinical CSV + deterministic bed inventory.

Usage:
    python -m app.seed            # seed if empty
    python -m app.seed --reset    # drop + recreate tables, then seed
"""
from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

from .config import DATA_DIR
from .database import Base, SessionLocal, engine
from .models import Bed, Patient, TriageEvent
from .services.icu_risk import score_frame

CLINICAL_CSV = DATA_DIR / "patient_clinical_records.csv"
CHUNK = 5000

BED_PLAN = [
    ("ICU_NORTH", 60, "ICU"),
    ("ICU_SOUTH", 40, "ICU"),
    ("TELEMETRY_WEST", 100, "TELE"),
    ("TELEMETRY_EAST", 100, "TELE"),
    ("GENERAL_1", 120, "GEN"),
    ("GENERAL_2", 120, "GEN"),
    ("GENERAL_3", 130, "GEN"),
    ("GENERAL_4", 130, "GEN"),
]


def seed_beds(db) -> int:
    if db.query(Bed).count() > 0:
        return 0
    beds = []
    for unit_name, count, prefix in BED_PLAN:
        for idx in range(1, count + 1):
            beds.append(
                Bed(
                    unit_name=unit_name,
                    bed_number=f"{prefix}-{idx:03d}",
                    is_telemetry_equipped=(prefix in ("ICU", "TELE")),
                    is_isolation_capable=(idx % 5 < 2),
                )
            )
    db.add_all(beds)
    db.commit()
    return len(beds)


def seed_clinical(db) -> int:
    if db.query(Patient).count() > 0:
        return 0
    df = pd.read_csv(CLINICAL_CSV)
    probs = score_frame(df)

    n_patients = 0
    n_events = 0
    for start in range(0, len(df), CHUNK):
        chunk = df.iloc[start : start + CHUNK]
        patients = []
        events = []
        for offset, (_, row) in enumerate(chunk.iterrows()):
            pid = str(row["patient_id"])
            recorded_at = datetime.fromisoformat(str(row["arrival_datetime_utc"]))
            patients.append(
                Patient(
                    patient_id=pid,
                    mrn=str(row["mrn"]),
                    age=int(row["age"]),
                    gender=str(row["gender"]),
                    is_isolation_required=bool(row["is_isolation_required"]),
                )
            )
            events.append(
                TriageEvent(
                    patient_id=pid,
                    esi_level=int(row["esi_level"]),
                    chief_complaint=str(row["chief_complaint_category"]),
                    heart_rate=int(row["heart_rate"]),
                    sys_bp=int(row["sys_bp"]),
                    dia_bp=int(row["dia_bp"]),
                    spo2=float(row["spo2"]),
                    temp_c=float(row["temp_c"]),
                    icu_escalation_prob=round(float(probs[start + offset]), 4),
                    recorded_at=recorded_at,
                )
            )
        db.bulk_save_objects(patients)
        db.bulk_save_objects(events)
        db.commit()
        n_patients += len(patients)
        n_events += len(events)
    print(f"triage events seeded: {n_events}")
    return n_patients


def main() -> None:
    reset = "--reset" in sys.argv
    if reset:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        bed_count = seed_beds(db)
        patient_count = seed_clinical(db)
        print(f"beds seeded: {bed_count}")
        print(f"patients seeded: {patient_count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
