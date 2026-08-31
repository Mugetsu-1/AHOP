"""
Prompt 1: Synthetic Data Pipeline Engine
=========================================
Generates a realistic synthetic Emergency Department (ED) dataset over a
2-year period at hourly frequency, modeled after MetroHealth Medical Center
(~250 ED visits/day, 750-bed tertiary care hospital).

Outputs:
    data/ed_hourly_arrivals.csv          - Hourly arrival stream + surge flags
    data/patient_clinical_records.csv    - Patient-level clinical records

Authors: AHOP Prompt Suite
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
YEARS = 2
HOURS = 24 * 365 * YEARS  # 17,520 hourly timesteps
AVG_DAILY_ARRIVALS = 250

ESI_DIST = {1: 0.05, 2: 0.20, 3: 0.45, 4: 0.20, 5: 0.10}
CHIEF_COMPLAINT_CATS = ["Cardiovascular", "Respiratory", "Trauma", "Gastrointestinal", "General"]
CHIEF_COMPLAINT_P = [0.25, 0.20, 0.20, 0.15, 0.20]

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ARRIVALS_OUT = OUT_DIR / "ed_hourly_arrivals.csv"
PATIENTS_OUT = OUT_DIR / "patient_clinical_records.csv"


def _seeded_rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. Hourly arrival stream
# ---------------------------------------------------------------------------
def build_arrival_stream() -> pd.DataFrame:
    rng = _seeded_rng()
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=HOURS, freq="h", tz="UTC")

    hour = timestamps.hour.values
    dow = timestamps.dayofweek.values  # 0 = Monday
    month = timestamps.month.values
    doy = timestamps.dayofyear.values

    # --- Diurnal cycle: peak 14:00-20:00, trough 02:00-06:00 ---
    # scaled around a 24h period so that amplitude peaks in the evening.
    diurnal_factor = 1.0 + 0.45 * np.cos((hour - 17.0) / 24.0 * 2 * np.pi)
    # --- Day-of-week effect: Monday peak, weekend trough ---
    dow_factor = 1.0 + 0.12 * np.cos((dow - 0.0) / 7.0 * 2 * np.pi)
    # --- Seasonality: winter/flu season peak (Jan-Mar) ---
    season_factor = 1.0 + 0.18 * np.cos((doy - 20.0) / 365.0 * 2 * np.pi)
    # --- Slow multi-year growth trend ---
    trend_factor = 1.0 + 0.05 * np.arange(HOURS) / HOURS

    base_rate = (AVG_DAILY_ARRIVALS / 24.0) * diurnal_factor * dow_factor * season_factor * trend_factor

    # --- Random shock / surge events (multi-vehicle accidents, weather) ---
    surge_prob = 0.012  # ~ every 3.5 days on average
    is_surge = rng.random(HOURS) < surge_prob
    surge_factor = np.where(is_surge, rng.uniform(1.5, 2.5, HOURS), 1.0)

    arrivals = rng.poisson(base_rate * surge_factor).astype(int)

    surge_type = np.where(
        is_surge,
        rng.choice(["MULTI_VEHICLE_ACCIDENT", "EXTREME_WEATHER", "PUBLIC_EVENT", "POWER_OUTAGE"], size=HOURS, p=[0.4, 0.3, 0.2, 0.1]),
        "NONE",
    )

    stream = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "hour_of_day": hour,
            "day_of_week": dow,
            "month": month,
            "arrivals": arrivals,
            "is_surge": is_surge.astype(int),
            "surge_type": surge_type,
        }
    )
    return stream


# ---------------------------------------------------------------------------
# 2. Patient-level clinical records
# ---------------------------------------------------------------------------
def sample_esi(rng: np.random.Generator, n: int) -> np.ndarray:
    levels = np.array(list(ESI_DIST.keys()))
    probs = np.array(list(ESI_DIST.values()))
    return rng.choice(levels, size=n, p=probs)


def vitals_for_esi(esi: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Generate vital signs whose severity is correlated with ESI acuity."""
    n = len(esi)

    def _scale(sev_mean, sev_std, esi):
        # ESI 1 (most acute) -> strongest effect; ESI 5 -> near-normal baselines.
        return sev_mean + sev_std * (6 - esi) / 5.0

    hr_mean = _scale(72, 28, esi)
    sys_mean = _scale(128, 26, esi)
    sys_mu = np.log(sys_mean)
    dia_mu = np.log(_scale(78, 12, esi))

    vitals = pd.DataFrame(
        {
            "heart_rate": np.clip(rng.normal(hr_mean, 14), 35, 220).round(1),
            "sys_bp": np.clip(rng.lognormal(sys_mu, 0.12), 55, 260).round(1),
            "dia_bp": np.clip(rng.lognormal(dia_mu, 0.14), 30, 150).round(1),
            "spo2": np.clip(rng.normal(_scale(97.0, 5.5, esi), 2.5), 55.0, 100.0).round(1),
            "temp_c": np.clip(rng.normal(_scale(37.0, 0.9, esi), 0.5), 34.0, 41.5).round(1),
            "lactate": np.clip(rng.lognormal(np.log(_scale(1.1, 3.2, esi)), 0.35), 0.3, 18.0).round(2),
        }
    )
    return vitals


def build_patient_records(arrival_stream: pd.DataFrame) -> pd.DataFrame:
    rng = _seeded_rng()
    total = int(arrival_stream["arrivals"].sum())
    print(f"Total simulated ED arrivals over {YEARS} years: {total:,}")

    faker = Faker()
    faker.seed_instance(SEED)

    esi = sample_esi(rng, total)
    vitals = vitals_for_esi(esi, rng)
    # Assign each patient to an arrival hour (weighted by hourly counts).
    arrival_hour_idx = rng.choice(len(arrival_stream), size=total, p=arrival_stream["arrivals"] / total)
    arrival_dt = arrival_stream["timestamp_utc"].iloc[arrival_hour_idx].reset_index(drop=True)
    surge_at_arrival = arrival_stream["is_surge"].iloc[arrival_hour_idx].reset_index(drop=True)

    age = np.clip(rng.lognormal(np.log(52), 0.45, size=total), 0, 99).round(0).astype(int)
    gender = rng.choice(["M", "F"], size=total, p=[0.48, 0.52])
    comorbidity = rng.poisson(1.2, size=total)  # Charlson-like index 0..4+

    # --- Actual LOS ~ Log-Normal parameterized by ESI and comorbidity ---
    # ESI 1 (critical) has the longest LOS; ESI 5 (minor) discharges quickly.
    esi_los_mu = {1: 2.4, 2: 1.9, 3: 1.3, 4: 0.6, 5: 0.0}
    esi_los_sigma = {1: 0.55, 2: 0.6, 3: 0.7, 4: 0.8, 5: 0.9}
    los_mu = np.array([esi_los_mu[e] for e in esi]) + 0.12 * comorbidity
    los_sigma = np.array([esi_los_sigma[e] for e in esi])
    los_hours = np.clip(rng.lognormal(los_mu, los_sigma), 0.5, 480.0).round(2)

    # --- ICU escalation flag (binary) ---
    # Strongly driven by ESI, vitals severity and surge status.
    icu_base = {1: 0.85, 2: 0.45, 3: 0.15, 4: 0.03, 5: 0.005}
    icu_prob = np.array([icu_base[e] for e in esi])
    icu_prob += 0.06 * ((vitals["spo2"] < 92).astype(float))
    icu_prob += 0.08 * ((vitals["sys_bp"] < 100).astype(float))
    icu_prob += 0.05 * ((vitals["lactate"] > 4).astype(float))
    icu_prob += 0.03 * surge_at_arrival.values.astype(float)
    icu_escalation = (rng.random(total) < np.clip(icu_prob, 0, 1)).astype(int)

    # --- ED wait time (door-to-bed) under surge pressure ---
    wait_esi_base = {1: 12, 2: 55, 3: 95, 4: 130, 5: 150}
    wait_mean = np.array([wait_esi_base[e] for e in esi], dtype=float)
    wait_mean *= 1 + 0.35 * surge_at_arrival.values.astype(float)
    wait_min = np.clip(rng.lognormal(np.log(wait_mean), 0.45), 3, 600).round(0).astype(int)

    # --- Isolation requirement (e.g., contact/droplet precautions) ---
    isolation = rng.choice([0, 1], size=total, p=[0.82, 0.18])

    complaints = rng.choice(CHIEF_COMPLAINT_CATS, size=total, p=CHIEF_COMPLAINT_P)

    records = pd.DataFrame(
        {
            "patient_id": [str(uuid.uuid4()) for _ in range(total)],
            "mrn": [faker.unique.numerify("MRN#########") for _ in range(total)],
            "arrival_datetime_utc": arrival_dt.values,
            "arrival_hour": arrival_dt.dt.hour.values,
            "day_of_week": arrival_dt.dt.dayofweek.values,
            "is_surge_arrival": surge_at_arrival.values.astype(int),
            "age": age,
            "gender": gender,
            "esi_level": esi,
            "chief_complaint_category": complaints,
            "comorbidity_index": comorbidity,
            **vitals.to_dict("list"),
            "los_hours": los_hours,
            "ed_wait_time_min": wait_min,
            "is_isolation_required": isolation,
            "icu_escalation_flag": icu_escalation,
        }
    )
    return records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stream = build_arrival_stream()
    stream.to_csv(ARRIVALS_OUT, index=False)

    records = build_patient_records(stream)
    records.to_csv(PATIENTS_OUT, index=False)

    print(f"\nWrote: {ARRIVALS_OUT}")
    print(f"Wrote: {PATIENTS_OUT}")
    print(records.head(10).to_string())


if __name__ == "__main__":
    main()
