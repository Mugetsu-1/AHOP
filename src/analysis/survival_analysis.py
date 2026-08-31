"""AHOP Prompt 2 (Data Science Survival Analysis Module).

Uses lifelines to:
  a) Fit Kaplan-Meier survival curves for Length of Stay (time-to-discharge)
     stratified by ESI level, plus log-rank tests.
  b) Build a Cox Proportional Hazards Model for discharge hazard based on
     Age, ESI level, Systolic BP, and Lactate.
  c) Print hazard-ratio tables with 95% confidence intervals and p-values.

Input : data/patient_clinical_records.csv
Output: reports/figures/km_curves_by_esi.png, km_curves_by_esi.html
        reports/survival_analysis_summary.txt (copies of console output)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import (
    logrank_test,
    multivariate_logrank_test,
    pairwise_logrank_test,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
PATIENTS = DATA / "patient_clinical_records.csv"
SUMMARY = ROOT / "reports" / "survival_analysis_summary.txt"

_tab = "  "


def _log(msg: str = "") -> None:
    print(msg)
    print(msg, file=_buffer)


def _hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Pretty hazard-ratio table with 95% CI and p-values."""
    s = cph.summary
    out = pd.DataFrame(
        {
            "coef": s["coef"],
            "exp(coef) HR": s["exp(coef)"],
            "exp(coef) 95% CI lower": s["exp(coef) lower 95%"],
            "exp(coef) 95% CI upper": s["exp(coef) upper 95%"],
            "p": s["p"],
        }
    )
    return out.round(4)


def km_by_esi(patients: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    esi_levels = sorted(patients["esi_level"].unique())
    kmf = KaplanMeierFitter()

    for esi in esi_levels:
        sub = patients[patients["esi_level"] == esi]
        kmf.fit(
            durations=sub["los_hours"],
            event_observed=np.ones(len(sub)),
            label=f"ESI {esi}",
        )
        kmf.plot_survival_function(ax=ax)

    ax.set_title("Kaplan-Meier Survival Curves for ED Length of Stay by ESI")
    ax.set_xlabel("Time since arrival (hours)")
    ax.set_ylabel("P(Still in ED)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "km_curves_by_esi.png", dpi=150)
    plt.close(fig)

    medians = {}
    for esi in esi_levels:
        kmf.fit(patients.loc[patients["esi_level"] == esi, "los_hours"])
        medians[esi] = kmf.median_survival_time_
    _log("\n[KM] Median LOS (hours) by ESI level:")
    _log(_tab + "\n".join(f"ESI {k}: {v:.2f} h" for k, v in medians.items()))

    _log("\n[KM] Pairwise log-rank tests (H0: identical survival curves):")
    pair = pairwise_logrank_test(
        patients["los_hours"],
        patients["esi_level"],
        event_observed=np.ones(len(patients)),
    )
    _log(_tab + str(pair.summary.round(4)))

    _log("\n[KM] Global log-rank test across all ESI strata:")
    global_test = multivariate_logrank_test(
        patients["los_hours"],
        patients["esi_level"],
        event_observed=np.ones(len(patients)),
    )
    _log(_tab + f"chi2={global_test.test_statistic:.3f}, "
                f"p={global_test.p_value:.3e}")


def cox_model(patients: pd.DataFrame) -> CoxPHFitter:
    cols = ["age", "esi_level", "sys_bp", "lactate"]
    df = patients[cols].copy()
    df["esi_level"] = df["esi_level"].astype(float)
    df["los_hours"] = patients["los_hours"].values
    df["event"] = np.ones(len(df))

    cph = CoxPHFitter()
    cph.fit(
        df,
        duration_col="los_hours",
        event_col="event",
        formula="age + esi_level + sys_bp + lactate",
        show_progress=False,
    )

    _log("\n[Cox PH] Model summary (baseline = no covariates):")
    _log(_tab + "Log-likelihood ratio test: "
         f"p={cph.log_likelihood_ratio_test().p_value:.3e}")
    _log(_tab + f"C-index: {cph.concordance_index_:.4f}")

    _log("\n[Cox PH] Hazard ratio table (exp(coef)) with 95% CI and p-values:")
    _log(_tab + _hr_table(cph).to_string())

    _log("\n[Cox PH] Interpretation: HR > 1 increases discharge hazard (shorter LOS);")
    _log(_tab + "HR < 1 decreases discharge hazard (longer LOS).")
    return cph


def _buffer_init() -> None:
    global _buffer
    _buffer = open(SUMMARY, "w", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    _buffer_init()

    patients = pd.read_csv(PATIENTS)
    _log(f"Patients loaded: {len(patients):,} rows")
    _log(f"Columns: {', '.join(patients.columns)}")

    km_by_esi(patients)
    cox_model(patients)

    _log(f"\n[OK] Figures + summary written under {ROOT / 'reports'}")

    _buffer.close()
    sys.stdout.write(f"\n[survival_analysis] Summary saved to {SUMMARY}\n")


if __name__ == "__main__":
    main()
