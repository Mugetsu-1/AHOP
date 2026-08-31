"""Bed-allocation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    AllocationOptimizeRequest,
    AllocationOptimizeResponse,
    AllocationItem,
)
from ..services.allocation import run_allocation

router = APIRouter(prefix="/allocation", tags=["allocation"])


@router.post("/optimize", response_model=AllocationOptimizeResponse)
def optimize(payload: AllocationOptimizeRequest, db: Session = Depends(get_db)):
    try:
        result = run_allocation(
            db,
            max_solver_time_sec=payload.max_solver_time_sec,
            enforce_strict_isolation=payload.enforce_strict_isolation,
        )
    except Exception as exc:  # solver/DB failures surface as 500s with detail
        raise HTTPException(status_code=500, detail=str(exc))

    return AllocationOptimizeResponse(
        solver_status=result["solver_status"],
        execution_time_ms=result["execution_time_ms"],
        assignments_made=result["assignments_made"],
        allocations=[AllocationItem(**item) for item in result["allocations"]],
    )
