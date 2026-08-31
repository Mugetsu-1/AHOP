"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Vitals(BaseModel):
    heart_rate: int
    sys_bp: int
    dia_bp: int
    spo2: float
    temp_c: float


class TriageAssessRequest(BaseModel):
    """Superset schema: accepts nested `vitals` and/or flat vitals fields."""

    patient_id: str | None = None
    age: int = Field(ge=0, le=120)
    gender: str = "M"
    esi_level: int = Field(ge=1, le=5)
    chief_complaint: str = ""
    chief_complaint_category: str | None = None
    comorbidity_index: int = Field(default=0, ge=0)
    heart_rate: int | None = None
    sys_bp: int | None = None
    dia_bp: int | None = None
    spo2: float | None = None
    temp_c: float | None = None
    lactate: float | None = None
    ed_wait_time_min: int | None = None
    is_isolation_required: bool = False
    arrival_hour: int | None = None
    day_of_week: int | None = None
    is_surge_arrival: int = 0
    vitals: Vitals | None = None

    @model_validator(mode="before")
    @classmethod
    def merge_vitals(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        vitals = data.pop("vitals", None)
        if isinstance(vitals, dict):
            for key, value in vitals.items():
                data.setdefault(key, value)
        return data


class ShapFactor(BaseModel):
    feature: str
    impact: float


class TriageAssessResponse(BaseModel):
    patient_id: str
    icu_escalation_probability: float
    risk_category: str
    recommended_unit: str
    shap_factors: list[ShapFactor]


class AllocationOptimizeRequest(BaseModel):
    max_solver_time_sec: float = Field(default=2.0, gt=0, le=30)
    enforce_strict_isolation: bool = True


class AllocationItem(BaseModel):
    patient_id: str
    assigned_bed_id: str
    unit_name: str
    bed_number: str
    expected_wait_reduction_min: int


class AllocationOptimizeResponse(BaseModel):
    solver_status: str
    execution_time_ms: int
    assignments_made: int
    allocations: list[AllocationItem]


class BedOccupancyUnit(BaseModel):
    unit_name: str
    total: int
    occupied: int
    available: int


class BedOccupancy(BaseModel):
    total_beds: int
    occupied_beds: int
    available_beds: int
    occupancy_pct: float
    by_unit: list[BedOccupancyUnit]


class ArrivalSeriesPoint(BaseModel):
    timestamp: str
    value: float


class ArrivalForecast(BaseModel):
    actual: list[ArrivalSeriesPoint]
    predicted: list[ArrivalSeriesPoint]


class DashboardMetricsResponse(BaseModel):
    bed_occupancy: BedOccupancy
    arrival_forecast: ArrivalForecast
    last_updated_utc: str
