"""AHOP Prompt 2 (Data Analysis Module).

Produces:
  a) Heatmap of patient arrivals by Hour of Day vs Day of Week.
  b) Door-to-Bed queue latency distribution by ESI level.
  c) Correlation matrix between initial vital-sign abnormalities and total LOS.

Inputs:
  data/ed_hourly_arrivals.csv
  data/patient_clinical_records.csv

Outputs:
  reports/figures/arrival_heatmap.png, arrival_heatmap.html
  reports/figures/door_to_bed_latency_by_esi.png
  reports/figures/vital_los_correlation.png, vital_los_correlation.html
  Console summary of key statistics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"

ARRIVALS = DATA / "ed_hourly_arrivals.csv"
PATIENTS = DATA / "patient_clinical_records.csv"

sns.set_theme(style="whitegrid")


def _heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Hour-of-Day x Day-of-Week into a 24 x 7 matrix of mean arrivals."""
    m = df.pivot_table(
        index="hour_of_day",
        columns="day_of_week",
        values="arrivals",
        aggfunc="mean",
    )
    for c in range(7):
        if c not in m.columns:
            m[c] = np.nan
    m = m.reindex(columns=sorted(m.columns))
    return m


def arrival_heatmap(arrivals: pd.DataFrame) -> None:
    m = _heatmap_matrix(arrivals)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        m,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Mean arrivals / hour"},
    )
    ax.set_title("Patient Arrivals Heatmap: Hour of Day vs Day of Week")
    ax.set_xlabel("Day of Week (0=Monday)")
    ax.set_ylabel("Hour of Day (0-23)")
    fig.tight_layout()
    fig.savefig(FIG / "arrival_heatmap.png", dpi=150)
    plt.close(fig)

    fig_html = px.imshow(
        m.T,
        color_continuous_scale="YlOrRd",
        labels={"x": "Hour of Day", "y": "Day of Week", "color": "Mean arrivals"},
        title="Patient Arrivals Heatmap: Hour of Day vs Day of Week",
    )
    fig_html.write_html(FIG / "arrival_heatmap.html")

    peak_hour = int(m.sum(axis=1).idxmax())
    peak_dow = int(m.sum(axis=0).idxmax())
    print(f"[EDA] Peak arrival hour: {peak_hour:02d}:00  (mean {m.sum(axis=1).max():.1f} patients/h)")
    print(f"[EDA] Peak day-of-week: {peak_dow}  (mean {m.sum(axis=0).max():.1f} patients/day)")
    print(f"[EDA] Trough arrival hour: {int(m.sum(axis=1).idxmin()):02d}:00")
    print(f"[EDA] Diurnal ratio (peak/trough): {m.sum(axis=1).max() / m.sum(axis=1).min():.2f}x")


def door_to_bed_latency(patients: pd.DataFrame) -> None:
    g = patients.groupby("esi_level")["ed_wait_time_min"]
    agg = g.agg(["count", "mean", "median", "std"])
    agg["p95"] = g.apply(lambda s: s.quantile(0.95))
    agg["p99"] = g.apply(lambda s: s.quantile(0.99))
    print("\n[EDA] Door-to-Bed queue latency (minutes) by ESI level:")
    print(agg.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=patients, x="esi_level", y="ed_wait_time_min", ax=axes[0], palette="viridis")
    axes[0].set_title("Door-to-Bed Latency by ESI (boxplot)")
    axes[0].set_xlabel("ESI level")
    axes[0].set_ylabel("ED wait time (min)")
    axes[0].set_ylim(0, patients["ed_wait_time_min"].quantile(0.99))

    sns.violinplot(data=patients, x="esi_level", y="ed_wait_time_min", ax=axes[1], palette="viridis", cut=0)
    axes[1].set_title("Door-to-Bed Latency by ESI (violin)")
    axes[1].set_xlabel("ESI level")
    axes[1].set_ylabel("ED wait time (min)")
    axes[1].set_ylim(0, patients["ed_wait_time_min"].quantile(0.99))

    fig.tight_layout()
    fig.savefig(FIG / "door_to_bed_latency_by_esi.png", dpi=150)
    plt.close(fig)


def _abnormality_flags(p: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=p.index)
    flags["HR_abn"] = ((p["heart_rate"] < 60) | (p["heart_rate"] > 100)).astype(int)
    flags["SBP_low"] = (p["sys_bp"] < 90).astype(int)
    flags["DBP_low"] = (p["dia_bp"] < 60).astype(int)
    flags["SpO2_low"] = (p["spo2"] < 95).astype(int)
    flags["Temp_abn"] = ((p["temp_c"] < 36.0) | (p["temp_c"] > 38.0)).astype(int)
    flags["Lactate_high"] = (p["lactate"] > 2.0).astype(int)
    return flags


def vital_los_correlation(patients: pd.DataFrame) -> None:
    flags = _abnormality_flags(patients)
    merged = flags.copy()
    merged["total_los_hours"] = patients["los_hours"].values
    merged["ed_wait_min"] = patients["ed_wait_time_min"].values

    corr = merged.corr()
    print("\n[EDA] Correlation matrix (vital-sign abnormality flags vs total LOS):")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation: Vital-Sign Abnormalities vs Total LOS")
    fig.tight_layout()
    fig.savefig(FIG / "vital_los_correlation.png", dpi=150)
    plt.close(fig)

    fig_html = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation: Vital-Sign Abnormalities vs Total LOS",
        text_auto=True,
    )
    fig_html.write_html(FIG / "vital_los_correlation.html")

    los_corrs = corr["total_los_hours"].drop("total_los_hours").sort_values(ascending=False)
    print("\n[EDA] Ranked correlation with total LOS:")
    print(los_corrs.round(3).to_string())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    arrivals = pd.read_csv(ARRIVALS)
    patients = pd.read_csv(PATIENTS)

    arrival_heatmap(arrivals)
    door_to_bed_latency(patients)
    vital_los_correlation(patients)

    print(f"\n[EDA] Figures written to {FIG.resolve()}")


if __name__ == "__main__":
    main()
