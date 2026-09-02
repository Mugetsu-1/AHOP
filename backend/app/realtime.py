"""Live replay hub: bridges the MIMIC telemetry replay engine to WebSocket clients.

Keeps an in-memory mirror of the ED (triage queue + admitted patients + bed
inventory) that advances with the replayed timeline. Live bed allocation reuses
the same MILP solver as the REST allocation path (bed_allocation_solver),
applied to the hub's in-memory queue instead of the DB.

The replay task is started lazily on the first WebSocket connection (or an
explicit control call), so importing this module has no side effects and tests
never spawn a background task.
"""
from __future__ import annotations

import asyncio
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from fastapi import WebSocket

from backend.app.config import ICU_RISK_THRESHOLD, TELEMETRY_RISK_THRESHOLD
from backend.ml.bed_allocation_solver import (
    DEFAULT_TELEMETRY_RISK_THRESHOLD,
    solve_allocation,
)
from backend.streamer.live_telemetry_replay import LiveTelemetryReplay

# Display anchor for the rebased sim clock (sim minute 0 == 2000-01-01T00:00Z).
REPLAY_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# Demo ICU-risk heuristic. The production XGBoost model needs 16 features that
# are not present in the MIMIC-IV-ED demo rows, so the hub derives a live proxy
# from ESI level + escalation flag.
ESI_BASE_RISK = {1: 0.85, 2: 0.55, 3: 0.30, 4: 0.12, 5: 0.05}
ESI_ESCALATION_BUMP = 0.15
FORECAST_REFRESH_TICKS = 60

# Minimum ED wait (sim-minutes) before a queued patient is eligible for bed
# allocation. At 100x speed (~100 sim-min per real second), 3 sim-minutes holds
# the patient in the queue for ~1 real second so the UI shows the delay.
QUEUE_HOLD_MINUTES = 3.0

# Wire-level event names for the WS {type, payload} envelope. The replay engine
# keeps its internal "arrival"/"telemetry"/"discharge" types; the hub maps them
# to the public contract here.
EVENT_WIRE_TYPE = {
    "arrival": "PATIENT_ARRIVED",
    "telemetry": "telemetry",
    "discharge": "PATIENT_DISCHARGED",
}

# Static in-memory bed plan (was previously seeded into SQLite). 800 beds.
BED_PLAN: list[tuple[str, int, str]] = [
    ("ICU_NORTH", 60, "ICU"),
    ("ICU_SOUTH", 40, "ICU"),
    ("TELEMETRY_WEST", 100, "TELE"),
    ("TELEMETRY_EAST", 100, "TELE"),
    ("GENERAL_1", 120, "GEN"),
    ("GENERAL_2", 120, "GEN"),
    ("GENERAL_3", 130, "GEN"),
    ("GENERAL_4", 130, "GEN"),
]


def _build_beds() -> dict[str, dict[str, Any]]:
    beds: dict[str, dict[str, Any]] = {}
    for unit_name, count, prefix in BED_PLAN:
        for idx in range(1, count + 1):
            bed_id = f"{unit_name}:{idx:03d}"
            beds[bed_id] = {
                "bed_id": bed_id,
                "unit_name": unit_name,
                "bed_number": f"{prefix}-{idx:03d}",
                "is_telemetry_equipped": prefix in ("ICU", "TELE"),
                "is_isolation_capable": idx % 5 < 2,
                "status": "AVAILABLE",
            }
    return beds


def _unit_type(unit_name: str) -> str:
    upper = (unit_name or "").upper()
    if "ICU" in upper:
        return "ICU"
    if "TELE" in upper:
        return "Telemetry"
    return "General"


@dataclass
class LivePatient:
    patient_id: str
    mrn: str
    esi_level: int
    gender: str
    chief_complaint: str
    disposition: str
    icu_escalation_flag: bool
    icu_risk: float
    isolation_required: bool
    arrival_min: float
    discharge_min: float
    vitals: dict[str, float] = field(default_factory=dict)
    wait_minutes: int = 0
    admitted: bool = False
    bed_id: str | None = None
    unit_name: str | None = None
    bed_number: str | None = None


class LiveReplayHub:
    """In-memory ED state driven by the telemetry replay engine."""

    def __init__(self) -> None:
        self.replay = LiveTelemetryReplay()
        self.patients: dict[str, LivePatient] = {}
        self.queue: list[str] = []
        self.allocations_made = 0
        self.paused = False
        self.running = False
        self.events_sent = 0

        self.clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self._tick_count = 0
        self._forecast_cache: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

        self._beds_loaded = False
        self.beds: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ beds
    def _ensure_beds(self) -> None:
        if self._beds_loaded:
            return
        self.beds = _build_beds()
        self._beds_loaded = True

    # ---------------------------------------------------------------- control
    def start(self) -> None:
        if self.running and self._task and not self._task.done():
            return
        self.running = True
        self.paused = False
        self._task = asyncio.get_running_loop().create_task(self._run_loop())

    def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def reset(self) -> None:
        self.replay.reset()
        self.patients.clear()
        self.queue.clear()
        self.allocations_made = 0
        self.paused = False
        self._forecast_cache = None
        for bed in self.beds.values():
            bed["status"] = "AVAILABLE"
        await self.broadcast({"type": "clock", "payload": self._clock_payload()})
        await self.broadcast(self.snapshot())

    # ------------------------------------------------------------------ loop
    async def _run_loop(self) -> None:
        self._ensure_beds()
        while self.running:
            await asyncio.sleep(self.replay.tick_seconds)
            if self.paused:
                continue
            self._tick_count += 1
            try:
                events = self.replay.step()
            except Exception:
                events = []
            await self._process_events(events)
            await self._live_allocate()
            await self.broadcast_queue()
            await self.broadcast({"type": "clock", "payload": self._clock_payload()})
            if self._tick_count % FORECAST_REFRESH_TICKS == 0:
                self._forecast_cache = None

    async def _process_events(self, events: list[dict]) -> None:
        if not events:
            return
        for ev in events:
            kind = ev["type"]
            if kind == "arrival":
                self._handle_arrival(ev)
            elif kind == "telemetry":
                pid = ev["patient_id"]
                if pid in self.patients:
                    self.patients[pid].vitals = ev["vitals"]
            elif kind == "discharge":
                p = self._handle_discharge(ev)
                if p is not None:
                    ev = {
                        **ev,
                        "admitted": p.admitted,
                        "bed_id": p.bed_id,
                        "unit_name": p.unit_name,
                        "bed_number": p.bed_number,
                    }
            wire_type = EVENT_WIRE_TYPE.get(kind, kind)
            await self.broadcast({"type": wire_type, "payload": ev})

    # ----------------------------------------------------------------- events
    def _handle_arrival(self, ev: dict) -> None:
        pid = ev["patient_id"]
        esi = int(ev["esi_level"])
        escalation = bool(ev["icu_escalation_flag"])
        risk = min(ESI_BASE_RISK.get(esi, 0.3) + (ESI_ESCALATION_BUMP if escalation else 0.0), 0.95)
        isolation = zlib.crc32(pid.encode("utf-8")) % 10 == 0
        self.patients[pid] = LivePatient(
            patient_id=pid,
            mrn=ev["mrn"],
            esi_level=esi,
            gender=ev["gender"],
            chief_complaint=ev["chief_complaint"],
            disposition=ev["disposition"],
            icu_escalation_flag=escalation,
            icu_risk=round(risk, 4),
            isolation_required=isolation,
            arrival_min=float(ev["arrival_min"]),
            discharge_min=float(ev["discharge_min"]),
            vitals=ev["vitals"],
        )
        self.queue.append(pid)

    def _handle_discharge(self, ev: dict) -> LivePatient | None:
        pid = ev["patient_id"]
        p = self.patients.pop(pid, None)
        if p is None:
            return None
        if p.admitted and p.bed_id:
            bed = self.beds.get(p.bed_id)
            if bed is not None:
                bed["status"] = "AVAILABLE"
        if pid in self.queue:
            self.queue.remove(pid)
        return p

    def _refresh_waits(self) -> None:
        clock = self.replay.sim_clock_min
        for pid in self.queue:
            p = self.patients[pid]
            p.wait_minutes = max(0, int(clock - p.arrival_min))

    # -------------------------------------------------------------- allocation
    def _risk_tier(self, prob: float) -> str:
        if prob >= ICU_RISK_THRESHOLD:
            return "HIGH"
        if prob >= TELEMETRY_RISK_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    async def _live_allocate(self) -> None:
        self._ensure_beds()
        self._refresh_waits()
        clock = self.replay.sim_clock_min
        queued = [
            self.patients[pid]
            for pid in self.queue
            if clock - self.patients[pid].arrival_min >= QUEUE_HOLD_MINUTES
        ]
        if not queued:
            return

        solver_patients = [
            {
                "patient_id": p.patient_id,
                "esi_level": p.esi_level,
                "icu_risk": p.icu_risk,
                "isolation_required": p.isolation_required,
                "wait_minutes": p.wait_minutes,
                "current_unit": "ED",
                "acuity_label": self._risk_tier(p.icu_risk),
                "location": "0",
            }
            for p in queued
        ]
        solver_beds = [
            {
                "bed_id": b["bed_id"],
                "unit_type": _unit_type(b["unit_name"]),
                "telemetry": b["is_telemetry_equipped"],
                "isolation_capable": b["is_isolation_capable"],
                "location": b["unit_name"],
            }
            for b in self.beds.values()
            if b["status"] == "AVAILABLE"
        ]
        if not solver_beds:
            return

        try:
            result = solve_allocation(
                solver_patients,
                solver_beds,
                telemetry_threshold=DEFAULT_TELEMETRY_RISK_THRESHOLD,
            )
        except Exception:
            return

        assigned: list[str] = []
        for assignment in result["assignments"]:
            pid = assignment["patient_id"]
            if pid not in self.queue:
                continue
            p = self.patients[pid]
            self.queue.remove(pid)
            p.admitted = True
            p.bed_id = assignment["bed_id"]
            bed = self.beds.get(assignment["bed_id"])
            if bed is not None:
                bed["status"] = "OCCUPIED"
                p.unit_name = bed["unit_name"]
                p.bed_number = bed["bed_number"]
            assigned.append(pid)

        if assigned:
            self.allocations_made += len(assigned)
            for pid in assigned:
                p = self.patients[pid]
                await self.broadcast(
                    {
                        "type": "BED_ALLOCATED",
                        "payload": {
                            "patient_id": pid,
                            "bed_id": p.bed_id,
                            "unit_name": p.unit_name,
                            "bed_number": p.bed_number,
                        },
                    }
                )
            await self.broadcast(self.snapshot())

    # --------------------------------------------------------------- payloads
    def _clock_payload(self) -> dict[str, Any]:
        status = self.replay.status()
        sim_min = self.replay.sim_clock_min
        return {
            "sim_min": round(sim_min, 2),
            "sim_iso": (REPLAY_EPOCH + timedelta(minutes=sim_min)).isoformat(),
            "speed": self.replay.speed,
            "paused": self.paused,
            "running": self.running,
            "patients_in_ed": len(self.replay.active),
            "patients_seen": self.replay.patients_seen,
            "discharged": self.replay.discharged,
            "arrivals_remaining": status["arrivals_remaining"],
            "total_patients": status["total_patients"],
        }

    def _patient_view(self, p: LivePatient) -> dict[str, Any]:
        return {
            "patient_id": p.patient_id,
            "mrn": p.mrn,
            "esi_level": p.esi_level,
            "gender": p.gender,
            "chief_complaint": p.chief_complaint,
            "icu_escalation_flag": p.icu_escalation_flag,
            "icu_risk": p.icu_risk,
            "risk_tier": self._risk_tier(p.icu_risk),
            "isolation_required": p.isolation_required,
            "arrival_min": p.arrival_min,
            "discharge_min": p.discharge_min,
            "wait_minutes": p.wait_minutes,
            "admitted": p.admitted,
            "bed_id": p.bed_id,
            "unit_name": p.unit_name,
            "bed_number": p.bed_number,
            "vitals": dict(p.vitals),
        }

    def snapshot(self) -> dict[str, Any]:
        self._ensure_beds()
        self._refresh_waits()
        occupied = sum(1 for b in self.beds.values() if b["status"] == "OCCUPIED")
        total = len(self.beds)
        return {
            "type": "snapshot",
            "payload": {
                "clock": self._clock_payload(),
                "queue": [self._patient_view(self.patients[pid]) for pid in self.queue],
                "admitted": [
                    self._patient_view(p) for p in self.patients.values() if p.admitted
                ],
                "beds": [dict(b) for b in self.beds.values()],
                "bed_summary": {
                    "total": total,
                    "occupied": occupied,
                    "available": total - occupied,
                },
                "forecast": self.forecast(),
                "events_sent": self.events_sent,
                "allocations_made": self.allocations_made,
            },
        }

    def forecast(self) -> dict[str, Any]:
        if self._forecast_cache is None:
            try:
                events = self.replay.events
                if events is None or events.empty:
                    raise ValueError("no replay events")
                hours = events["arrival_min"].to_numpy() // 60
                counts = np.bincount(
                    (hours.astype(int) % 24).clip(min=0), minlength=24
                ).astype(float)
                std = float(counts.std()) if counts.size else 0.0
                actual = []
                predicted = []
                for i in range(24):
                    ts = REPLAY_EPOCH + timedelta(hours=i)
                    value = float(counts[i])
                    actual.append({"t": ts.isoformat(), "value": round(value, 4)})
                    predicted.append(
                        {
                            "t": ts.isoformat(),
                            "value": round(value, 4),
                            "lower": round(max(0.0, value - 1.96 * std), 4),
                            "upper": round(value + 1.96 * std, 4),
                        }
                    )
                self._forecast_cache = {
                    "actual": actual,
                    "predicted": predicted,
                    "residual_std": round(std, 4),
                }
            except Exception:
                self._forecast_cache = {"actual": [], "predicted": [], "residual_std": 0.0}
        return self._forecast_cache

    # ---------------------------------------------------------------- clients
    async def register(self, ws: WebSocket) -> None:
        self._ensure_beds()
        self.clients.add(ws)
        await ws.send_json({"type": "hello", "payload": self._clock_payload()})
        await ws.send_json(self.snapshot())
        self.start()

    def unregister(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self.events_sent += 1
        if not self.clients:
            return
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def broadcast_queue(self) -> None:
        queue = [self._patient_view(self.patients[pid]) for pid in self.queue]
        await self.broadcast({"type": "queue_update", "payload": {"queue": queue}})


hub = LiveReplayHub()
