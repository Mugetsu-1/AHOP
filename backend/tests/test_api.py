"""Smoke tests for the AHOP realtime-only API."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_realtime_status():
    resp = client.get("/api/v1/realtime/status")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "sim_min",
        "sim_iso",
        "speed",
        "paused",
        "running",
        "queue_length",
        "admitted",
        "available_beds",
        "total_beds",
        "events_sent",
        "allocations_made",
    ):
        assert key in body
    assert body["total_beds"] == 800


def test_control_speed():
    resp = client.post(
        "/api/v1/realtime/control", json={"action": "speed", "speed": 2.0}
    )
    assert resp.status_code == 200
    assert resp.json()["speed"] == 2.0


def test_control_speed_missing_400():
    resp = client.post("/api/v1/realtime/control", json={"action": "speed"})
    assert resp.status_code == 400


def test_control_invalid_action_422():
    resp = client.post("/api/v1/realtime/control", json={"action": "bogus"})
    assert resp.status_code == 422


def test_control_pause_resume():
    resp = client.post("/api/v1/realtime/control", json={"action": "pause"})
    assert resp.status_code == 200
    assert resp.json()["paused"] is True
    resp = client.post("/api/v1/realtime/control", json={"action": "resume"})
    assert resp.status_code == 200
    assert resp.json()["paused"] is False
