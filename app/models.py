"""ORM models mirroring the spec DDL (SQLite-compatible)."""
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mrn: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    is_isolation_required: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class TriageEvent(Base):
    __tablename__ = "triage_events"
    __table_args__ = (PrimaryKeyConstraint("event_id", "recorded_at"),)

    event_id: Mapped[str] = mapped_column(String(36), default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.patient_id"), nullable=False
    )
    esi_level: Mapped[int] = mapped_column(Integer, nullable=False)
    chief_complaint: Mapped[str] = mapped_column(String(255), nullable=False)
    heart_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sys_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dia_bp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    temp_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    icu_escalation_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class Bed(Base):
    __tablename__ = "beds"

    bed_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    unit_name: Mapped[str] = mapped_column(String(64), nullable=False)
    bed_number: Mapped[str] = mapped_column(String(16), nullable=False)
    is_telemetry_equipped: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    is_isolation_capable: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="AVAILABLE", nullable=False)


class BedAllocation(Base):
    __tablename__ = "bed_allocations"

    allocation_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("patients.patient_id"), nullable=False
    )
    bed_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("beds.bed_id"), nullable=False
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    expected_discharge_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    actual_discharge_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    solver_execution_id: Mapped[str] = mapped_column(String(64), nullable=False)
