"""Live telemetry replay engine for the AHOP demo.

Loads data/replay_events.csv (produced by mimic_replay_mapping.py) and replays
ED visits on the real (rebased) MIMIC timeline: an arrival fires when the sim
clock crosses its arrival_min, every admitted patient emits drifting vitals
every telemetry_interval_min, and a discharge fires at discharge_min.

The sim clock is kept in simulated minutes rebased onto the earliest ED visit
(so the demo data's true ~89-year span is preserved verbatim). The clock
advances by `speed` simulated minutes per tick; with speed=1.0 and a 1.0s tick
that is 1 sim-minute per real second. Because the source timeline is sparse,
most ticks emit nothing but a clock tick.

Run (smoke test):
    python src/streamer/live_telemetry_replay.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_CSV = BASE_DIR / "data" / "replay_events.csv"

DEFAULT_TICK_SECONDS = 1.0  # real seconds per step()
DEFAULT_SPEED = 1.0  # simulated minutes advanced per step()
DEFAULT_TELEMETRY_INTERVAL_MIN = 5.0  # sim-minutes between telemetry samples
DEFAULT_TEMPERATURE_C = 36.8

VITAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (35.0, 40.5),
    "resprate": (8.0, 40.0),
    "pain": (0.0, 10.0),
    "heartrate": (40.0, 200.0),
    "sbp": (70.0, 220.0),
    "dbp": (40.0, 130.0),
    "o2sat": (82.0, 100.0),
}

VITAL_DRIFT_STD: dict[str, float] = {
    "temperature_c": 0.1,
    "resprate": 1.0,
    "pain": 0.5,
    "heartrate": 3.0,
    "sbp": 3.0,
    "dbp": 2.0,
    "o2sat": 1.0,
}


@dataclass
class ActivePatient:
    patient_id: str
    mrn: str
    esi_level: int
    gender: str
    chief_complaint: str
    disposition: str
    icu_escalation_flag: bool
    arrival_min: float
    discharge_min: float
    vitals: dict[str, float]
    last_telemetry_min: float = 0.0


class LiveTelemetryReplay:
    def __init__(
        self,
        events_csv: Path = DEFAULT_EVENTS_CSV,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        speed: float = DEFAULT_SPEED,
        telemetry_interval_min: float = DEFAULT_TELEMETRY_INTERVAL_MIN,
        rng_seed: int | None = 42,
    ) -> None:
        self.events_csv = Path(events_csv)
        self.tick_seconds = tick_seconds
        self.telemetry_interval_min = telemetry_interval_min
        self._seed = rng_seed
        self.speed = float(speed)
        self._rng = np.random.default_rng(rng_seed)
        self.events = pd.read_csv(self.events_csv)
        self.events = self.events.sort_values("arrival_min").reset_index(drop=True)
        self.reset()

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.sim_clock_min = 0.0
        self._arrival_idx = 0
        self.active: dict[str, ActivePatient] = {}
        self.patients_seen = 0
        self.discharged = 0

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.0, float(speed))

    # -- event loop -------------------------------------------------------

    def step(self) -> list[dict[str, Any]]:
        """Advance the sim clock by `speed` sim-minutes and emit any events."""
        return self.advance_to(self.sim_clock_min + self.speed)

    def advance_to(self, clock_min: float) -> list[dict[str, Any]]:
        """Advance the sim clock to an absolute sim-minute and emit events.

        Ordering within a tick: arrivals, then discharges, then telemetry.
        """
        self.sim_clock_min = clock_min
        emitted: list[dict[str, Any]] = []

        while (
            self._arrival_idx < len(self.events)
            and self.events.iloc[self._arrival_idx]["arrival_min"] <= clock_min
        ):
            row = self.events.iloc[self._arrival_idx]
            self._arrival_idx += 1
            self.patients_seen += 1
            emitted.append(self._emit_arrival(row))

        for pid in list(self.active):
            ap = self.active[pid]
            if clock_min >= ap.discharge_min:
                emitted.append(self._emit_discharge(ap))
                del self.active[pid]
                self.discharged += 1

        for pid in list(self.active):
            ap = self.active[pid]
            if clock_min - ap.last_telemetry_min >= self.telemetry_interval_min:
                self._drift(ap)
                ap.last_telemetry_min = clock_min
                emitted.append(self._emit_telemetry(ap))

        return emitted

    # -- patient bookkeeping ----------------------------------------------

    def _base_vitals(self, row: pd.Series) -> dict[str, float]:
        temp = row["temperature_c"] if pd.notna(row["temperature_c"]) else DEFAULT_TEMPERATURE_C
        resprate = row["resprate"] if pd.notna(row["resprate"]) else 18.0
        pain = row["pain"] if pd.notna(row["pain"]) else min(10.0, float(row["esi_level"]) * 2.0)
        sbp = float(row["sbp"])
        dbp = float(row["dbp"]) if pd.notna(row["dbp"]) else round(sbp * 0.6, 1)
        return {
            "temperature_c": float(temp),
            "resprate": float(resprate),
            "pain": float(pain),
            "heartrate": float(row["heartrate"]),
            "sbp": sbp,
            "dbp": dbp,
            "o2sat": float(row["o2sat"]),
        }

    def _emit_arrival(self, row: pd.Series) -> dict[str, Any]:
        ap = ActivePatient(
            patient_id=str(row["stay_id"]),
            mrn=str(row["subject_id"]),
            esi_level=int(row["esi_level"]),
            gender=str(row["gender"]),
            chief_complaint=str(row["chief_complaint"]),
            disposition=str(row["disposition"]),
            icu_escalation_flag=bool(row["icu_escalation_flag"]),
            arrival_min=float(row["arrival_min"]),
            discharge_min=float(row["discharge_min"]),
            vitals=self._base_vitals(row),
        )
        self.active[ap.patient_id] = ap
        return {
            "type": "arrival",
            "patient_id": ap.patient_id,
            "mrn": ap.mrn,
            "esi_level": ap.esi_level,
            "gender": ap.gender,
            "chief_complaint": ap.chief_complaint,
            "disposition": ap.disposition,
            "icu_escalation_flag": ap.icu_escalation_flag,
            "arrival_min": ap.arrival_min,
            "discharge_min": ap.discharge_min,
            "sim_min": self.sim_clock_min,
            "vitals": dict(ap.vitals),
        }

    def _emit_telemetry(self, ap: ActivePatient) -> dict[str, Any]:
        return {
            "type": "telemetry",
            "patient_id": ap.patient_id,
            "sim_min": self.sim_clock_min,
            "vitals": dict(ap.vitals),
        }

    def _emit_discharge(self, ap: ActivePatient) -> dict[str, Any]:
        return {
            "type": "discharge",
            "patient_id": ap.patient_id,
            "discharge_min": ap.discharge_min,
            "sim_min": self.sim_clock_min,
            "disposition": ap.disposition,
            "icu_escalation_flag": ap.icu_escalation_flag,
        }

    def _drift(self, ap: ActivePatient) -> None:
        for key, (lo, hi) in VITAL_BOUNDS.items():
            value = ap.vitals[key]
            if value is None:
                continue
            noise = float(self._rng.normal(0.0, VITAL_DRIFT_STD[key]))
            ap.vitals[key] = round(float(np.clip(value + noise, lo, hi)), 1)

    # -- introspection -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "sim_clock_min": self.sim_clock_min,
            "speed": self.speed,
            "patients_in_ed": len(self.active),
            "patients_seen": self.patients_seen,
            "discharged": self.discharged,
            "arrivals_remaining": max(0, len(self.events) - self._arrival_idx),
            "total_patients": len(self.events),
        }


def main() -> None:
    replay = LiveTelemetryReplay()
    print(f"loaded {len(replay.events)} events from {replay.events_csv.name}")
    print(
        f"sim window: {replay.events['arrival_min'].min():.1f}.."
        f"{replay.events['discharge_min'].max():.1f} sim-min"
    )
    n_ticks = 2_000
    counts = {"arrival": 0, "discharge": 0, "telemetry": 0}
    first_arrival_tick: int | None = None
    for tick in range(n_ticks):
        for event in replay.step():
            if event["type"] == "arrival" and first_arrival_tick is None:
                first_arrival_tick = tick
            counts[event["type"]] += 1
    print(f"after {n_ticks} ticks (clock={replay.sim_clock_min:.0f} sim-min):")
    print(
        f"  arrivals={counts['arrival']} (first at tick {first_arrival_tick}) "
        f"discharges={counts['discharge']} telemetry={counts['telemetry']}"
    )
    print(f"  {replay.status()}")


if __name__ == "__main__":
    sys.exit(main())
