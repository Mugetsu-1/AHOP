"""Smoke tests for the AHOP API (runs against the seeded SQLite DB)."""
import os

os.environ.setdefault("AHOP_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
os.environ.setdefault("AHOP_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_triage_assess():
    payload = {
        "age": 64,
        "gender": "M",
        "esi_level": 2,
        "chief_complaint": "chest pain",
        "comorbidity_index": 1,
        "vitals": {
            "heart_rate": 104,
            "sys_bp": 142,
            "dia_bp": 68,
            "spo2": 97.0,
            "temp_c": 37.5,
        },
        "lactate": 4.01,
        "ed_wait_time_min": 178,
        "is_isolation_required": False,
        "arrival_hour": 20,
        "day_of_week": 6,
        "is_surge_arrival": 0,
    }
    resp = client.post("/api/v1/triage/assess", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "patient_id" in body
    assert "icu_escalation_probability" in body
    assert "risk_category" in body
    assert "recommended_unit" in body
    assert "shap_factors" in body
    assert 0.0 <= body["icu_escalation_probability"] <= 1.0


def test_triage_assess_missing_vitals_422():
    payload = {
        "age": 40,
        "esi_level": 3,
        "chief_complaint": "general weakness",
        "vitals": {
            "heart_rate": 90,
            "sys_bp": 120,
            "dia_bp": 80,
            "spo2": 98.0,
            "temp_c": 36.8,
        },
    }
    resp = client.post("/api/v1/triage/assess", json=payload)
    assert resp.status_code == 200

    bad = {"age": 40, "esi_level": 3, "chief_complaint": "general weakness"}
    resp = client.post("/api/v1/triage/assess", json=bad)
    assert resp.status_code == 422


def test_dashboard_metrics():
    resp = client.get("/api/v1/dashboard/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["bed_occupancy"]["total_beds"] == 800
    assert len(body["bed_occupancy"]["by_unit"]) == 8
    assert len(body["arrival_forecast"]["actual"]) == 24
    assert len(body["arrival_forecast"]["predicted"]) == 24


def test_allocation_optimize():
    resp = client.post(
        "/api/v1/allocation/optimize",
        json={"max_solver_time_sec": 2.0, "enforce_strict_isolation": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "solver_status" in body
    assert "assignments_made" in body
    assert body["assignments_made"] >= 0
    assert "allocations" in body
