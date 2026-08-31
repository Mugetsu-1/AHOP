"""Hourly ED-arrival forecasting against the trained LightGBM t+1h model.

Recursive next-24h predictions. Feature semantics mirror
build_arrival_features() in src/ml/forecasting_and_risk.py: each feature row
summarises the series through timestamp t and predicts arrivals at t+1h.
"""
from __future__ import annotations

import statistics

import lightgbm as lgb
import pandas as pd

from ..config import DATA_DIR, MODELS_DIR

MODEL_FILE = MODELS_DIR / "lightgbm_arrival_t1h.txt"
ARRIVALS_CSV = DATA_DIR / "ed_hourly_arrivals.csv"
HORIZON_HOURS = 24

_BOOSTER: lgb.Booster | None = None


def _load_arrivals() -> pd.DataFrame:
    df = pd.read_csv(ARRIVALS_CSV, parse_dates=["timestamp_utc"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def _feature_row(history: list[float], ts) -> list[float]:
    lag_1h = history[-2] if len(history) >= 2 else history[-1]
    lag_24h = history[-25] if len(history) >= 25 else lag_1h
    lag_168h = history[-169] if len(history) >= 169 else lag_1h
    roll_mean_6h = sum(history[-6:]) / min(6, len(history))
    roll_std_24h = statistics.stdev(history[-24:]) if len(history) >= 24 else 0.0
    return [
        history[-1],
        lag_1h,
        lag_24h,
        lag_168h,
        roll_mean_6h,
        roll_std_24h,
        ts.hour,
        ts.weekday(),
        ts.month,
        0,  # is_surge
    ]


def _booster() -> lgb.Booster:
    global _BOOSTER
    if _BOOSTER is None:
        _BOOSTER = lgb.Booster(model_file=str(MODEL_FILE))
    return _BOOSTER


def hourly_forecast(hours: int = HORIZON_HOURS) -> tuple[list[dict], list[dict]]:
    """Return (actuals, predictions) point lists.

    actuals    = the last `hours` observed hourly arrivals
    predicted  = recursive next-`hours` hourly arrival forecasts
    """
    df = _load_arrivals()
    history = df["arrivals"].astype(float).tolist()
    current_ts = df["timestamp_utc"].iloc[-1]

    predictions = []
    for _ in range(hours):
        features = _feature_row(history, current_ts)
        pred = float(_booster().predict([features])[0])
        predictions.append(round(max(pred, 0.0), 2))
        current_ts = current_ts + pd.Timedelta(hours=1)
        history.append(pred)

    actuals_df = df.tail(hours)
    actuals = [
        {
            "timestamp": ts.isoformat(),
            "value": float(val),
        }
        for ts, val in zip(actuals_df["timestamp_utc"], actuals_df["arrivals"])
    ]
    predicted = [
        {
            "timestamp": (df["timestamp_utc"].iloc[-1] + pd.Timedelta(hours=i + 1)).isoformat(),
            "value": val,
        }
        for i, val in enumerate(predictions)
    ]
    return actuals, predicted
