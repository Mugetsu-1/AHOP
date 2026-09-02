"""Map real MIMIC-IV-ED demo data (edstays + triage) into a replayable event set.

Reads the uncompressed demo twins (edstays_demo.csv / triage_demo.csv) or their
.gz originals, rebases every intime/outtime onto a simulated clock in minutes,
derives the ICU-escalation flag, and writes data/replay_events.csv for the live
telemetry replay engine.

The raw MIMIC-IV-ED demo spans ~89 years of simulated time with only 197 visits,
so by default the arrival timeline is compressed into a dense, continuous
operational window (default 24h) using a seeded mid-peak arrival curve. Every
clinical record (ESI, chief complaint, disposition, vitals) and each patient's
relative LOS is preserved; LOS is bounded to 1-12 sim-hours so ED beds turn over
realistically during the demo. Pass --window-hours 0 to keep the raw timeline.

Run:
    python backend/streamer/mimic_replay_mapping.py [--window-hours 24] [--rng-seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

EDSTAYS_NOMINAL = DATA_DIR / "edstays.csv.gz"
TRIAGE_NOMINAL = DATA_DIR / "triage.csv.gz"
EDSTAYS_FALLBACK = DATA_DIR / "edstays_demo.csv"
TRIAGE_FALLBACK = DATA_DIR / "triage_demo.csv"

OUTPUT_CSV = DATA_DIR / "replay_events.csv"

OUTPUT_COLUMNS = [
    "stay_id",
    "subject_id",
    "gender",
    "esi_level",
    "chief_complaint",
    "disposition",
    "icu_escalation_flag",
    "arrival_min",
    "discharge_min",
    "temperature_c",
    "resprate",
    "pain",
    "heartrate",
    "sbp",
    "dbp",
    "o2sat",
]

HIGH_RISK_DISPOSITIONS = {"ICU", "CRITICAL"}
REQUIRED_FIELDS = ["acuity", "intime", "outtime", "heartrate", "sbp", "o2sat"]

DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_RNG_SEED = 42
LOS_MIN_HOURS = 1.0
LOS_MAX_HOURS = 12.0


def _pick_csv(nominal: Path, fallback: Path) -> Path:
    return nominal if nominal.exists() else fallback


def _f_to_c(temp_f: float) -> float:
    return round((temp_f - 32) * 5 / 9, 2)


def _icu_escalation(row: pd.Series) -> bool:
    disposition = str(row.get("disposition", "")).upper()
    if disposition in HIGH_RISK_DISPOSITIONS:
        return True
    if row.get("acuity") == 1:
        return True
    o2 = row.get("o2sat")
    sbp = row.get("sbp")
    if pd.notna(o2) and pd.notna(sbp) and o2 < 90 and sbp < 95:
        return True
    return False


def load_and_map() -> pd.DataFrame:
    edstays = pd.read_csv(_pick_csv(EDSTAYS_NOMINAL, EDSTAYS_FALLBACK))
    triage = pd.read_csv(_pick_csv(TRIAGE_NOMINAL, TRIAGE_FALLBACK))

    triage = triage.rename(columns={"chiefcomplaint": "chief_complaint"})
    merged = triage.merge(
        edstays[
            ["stay_id", "gender", "disposition", "intime", "outtime"]
        ],
        on="stay_id",
        how="inner",
    )
    merged = merged.dropna(subset=REQUIRED_FIELDS).copy()

    merged["intime"] = pd.to_datetime(merged["intime"])
    merged["outtime"] = pd.to_datetime(merged["outtime"])
    merged["outtime"] = merged["outtime"].mask(
        merged["outtime"] < merged["intime"], merged["intime"]
    )

    epoch = merged["intime"].min()
    merged["arrival_min"] = (
        (merged["intime"] - epoch).dt.total_seconds() / 60.0
    ).round(1)
    merged["discharge_min"] = (
        (merged["outtime"] - epoch).dt.total_seconds() / 60.0
    ).round(1)

    merged["esi_level"] = merged["acuity"].astype(int)
    merged["icu_escalation_flag"] = merged.apply(_icu_escalation, axis=1)

    merged["temperature_c"] = merged["temperature"].apply(
        lambda v: _f_to_c(v) if pd.notna(v) and v > 50 else v
    )

    out = merged[OUTPUT_COLUMNS].sort_values("arrival_min").reset_index(drop=True)
    out["temperature_c"] = out["temperature_c"].round(1)
    out["icu_escalation_flag"] = out["icu_escalation_flag"].astype(bool)
    return out


def compress_timeline(
    df: pd.DataFrame,
    target_window_hours: float = DEFAULT_WINDOW_HOURS,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> pd.DataFrame:
    """Squeeze arrival_min into a dense window while preserving each LOS.

    Arrival offsets are drawn from a sorted beta(2, 2) curve (a mid-day peak,
    so the 24h window shows realistic ED arrival patterns) scaled to
    [0, target_window_hours * 60] sim-minutes, seeded for reproducibility.
    The first visit is pinned to sim-minute 0 so the window still begins with
    the earliest ED arrival. LOS is clipped to [LOS_MIN_HOURS, LOS_MAX_HOURS]
    so beds turn over realistically, and discharge_min = offset + LOS.

    A target_window_hours <= 0 returns the raw timeline unchanged.
    """
    if target_window_hours <= 0:
        return df

    rng = np.random.default_rng(rng_seed)
    n = len(df)
    window_min = target_window_hours * 60.0
    offsets = np.sort(rng.beta(2.0, 2.0, n)) * window_min
    offsets[0] = 0.0

    los_min = (df["discharge_min"] - df["arrival_min"]).clip(
        LOS_MIN_HOURS * 60.0, LOS_MAX_HOURS * 60.0
    )

    out = df.copy()
    out["arrival_min"] = offsets.round(1)
    out["discharge_min"] = (offsets + los_min.to_numpy()).round(1)
    return out


def generate(
    output_csv: Path = OUTPUT_CSV,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> pd.DataFrame:
    df = compress_timeline(load_and_map(), window_hours, rng_seed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map MIMIC-IV-ED demo data into replay_events.csv"
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
        help="Compress arrivals into this many sim-hours (0 = keep raw timeline). Default: %(default)s.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help="Seed for reproducible arrival compression. Default: %(default)s.",
    )
    args = parser.parse_args()

    df = generate(window_hours=args.window_hours, rng_seed=args.rng_seed)
    n = len(df)
    esi_dist = df["esi_level"].value_counts().sort_index()
    flagged = int(df["icu_escalation_flag"].sum())
    print(f"mapped {n} ED visits -> {OUTPUT_CSV.name}")
    print(f"sim window: {df['arrival_min'].min():.0f}..{df['discharge_min'].max():.0f} sim-min")
    print("esi_level distribution:")
    for lvl, count in esi_dist.items():
        print(f"  ESI {int(lvl)}: {count}")
    print(f"icu_escalation_flag: {flagged}/{n}")
    print(f"temperature_c non-null: {df['temperature_c'].notna().sum()}/{n}")


if __name__ == "__main__":
    sys.exit(main())
