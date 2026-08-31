"""Dashboard metrics endpoint."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Bed
from ..schemas import (
    ArrivalForecast,
    ArrivalSeriesPoint,
    BedOccupancy,
    BedOccupancyUnit,
    DashboardMetricsResponse,
)
from ..services.forecast import hourly_forecast

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/metrics", response_model=DashboardMetricsResponse)
def metrics(db: Session = Depends(get_db)):
    total = db.query(func.count(Bed.bed_id)).scalar() or 0
    occupied = db.query(func.count(Bed.bed_id)).filter(Bed.status == "OCCUPIED").scalar() or 0
    available = total - occupied

    rows = (
        db.query(Bed.unit_name, func.count(Bed.bed_id), Bed.status)
        .group_by(Bed.unit_name, Bed.status)
        .all()
    )
    by_unit: dict[str, dict[str, int]] = {}
    for unit_name, count, status in rows:
        entry = by_unit.setdefault(unit_name, {"total": 0, "occupied": 0, "available": 0})
        entry["total"] += count
        if status == "OCCUPIED":
            entry["occupied"] += count
        else:
            entry["available"] += count

    actual, predicted = hourly_forecast()

    return DashboardMetricsResponse(
        bed_occupancy=BedOccupancy(
            total_beds=total,
            occupied_beds=occupied,
            available_beds=available,
            occupancy_pct=round((occupied / total * 100) if total else 0.0, 1),
            by_unit=[
                BedOccupancyUnit(unit_name=name, **counts)
                for name, counts in sorted(by_unit.items())
            ],
        ),
        arrival_forecast=ArrivalForecast(
            actual=[ArrivalSeriesPoint(**point) for point in actual],
            predicted=[ArrivalSeriesPoint(**point) for point in predicted],
        ),
        last_updated_utc=datetime.now(timezone.utc).isoformat(),
    )
