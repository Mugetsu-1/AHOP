"""Prompt 3 (ML): Multi-horizon arrival forecasting + ICU escalation risk classifier.

Model 1: LightGBM Regressor on hourly ED arrivals.
    - Targets: arrivals at t+1h, t+6h, t+24h.
    - Features: lag_1h, lag_24h, lag_168h, rolling_mean_6h, rolling_std_24h,
      plus calendar / surge covariates.
    - Metrics: MAE, RMSE, WAPE.

Model 2: XGBoost Classifier on patient clinical records.
    - Target: icu_escalation_flag (clinical deterioration within 12h proxy).
    - Metrics: ROC-AUC, PR-AUC.
    - Interpretability: SHAP (TreeExplainer) summary.

Outputs:
    reports/forecasting_metrics.txt
    reports/figures/arrival_forecast_t{1,6,24}h.png
    reports/risk_model_metrics.txt
    reports/figures/icu_risk_roc_pr.png
    reports/figures/shap_summary.png
    models/lightgbm_arrival_t{1,6,24}h.txt
    models/xgboost_icu.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

FORECAST_HORIZONS = [1, 6, 24]

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_estimators": 500,
    "random_state": 42,
    "verbosity": -1,
}


def _log(msg: str) -> None:
    print(msg)


def build_arrival_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct lag / rolling features for hourly arrival forecasting."""
    out = pd.DataFrame(index=df.index)
    out["arrivals"] = df["arrivals"]
    out["lag_1h"] = df["arrivals"].shift(1)
    out["lag_24h"] = df["arrivals"].shift(24)
    out["lag_168h"] = df["arrivals"].shift(168)
    out["rolling_mean_6h"] = df["arrivals"].rolling(6).mean()
    out["rolling_std_24h"] = df["arrivals"].rolling(24).std()
    out["hour_of_day"] = df["hour_of_day"]
    out["day_of_week"] = df["day_of_week"]
    out["month"] = df["month"]
    out["is_surge"] = df["is_surge"]
    return out


def run_forecasting(arrivals_csv: Path) -> None:
    _log("[Forecasting] Loading arrivals...")
    df = pd.read_csv(arrivals_csv, parse_dates=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
    feat = build_arrival_features(df)

    metrics = {}
    fig, axes = plt.subplots(1, len(FORECAST_HORIZONS), figsize=(18, 4.5), sharex=True)

    for i, h in enumerate(FORECAST_HORIZONS):
        _log(f"[Forecasting] Horizon t+{h}h")
        y = df["arrivals"].shift(-h)
        data = feat.copy()
        data["target"] = y
        data = data.dropna(subset=["target"])

        features = [c for c in feat.columns]
        X = data[features]
        y_ = data["target"]

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_.iloc[:split], y_.iloc[split:]

        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        wape = np.sum(np.abs(y_test - pred)) / np.sum(np.abs(y_test))

        metrics[h] = {"mae": float(mae), "rmse": float(rmse), "wape": float(wape)}
        _log(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  WAPE={wape:.3%}")

        model.booster_.save_model(str(MODELS_DIR / f"lightgbm_arrival_t{h}h.txt"))

        ax = axes[i]
        ax.plot(y_test.to_numpy()[:336], label="Actual", alpha=0.85, linewidth=1.2)
        ax.plot(pred[:336], label="Predicted", alpha=0.85, linewidth=1.2)
        ax.set_title(f"t+{h}h | MAE {mae:.2f}, WAPE {wape:.1%}")
        ax.set_xlabel("Hours (test, first 14 days)")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("LightGBM Multi-Horizon Arrival Forecast (Hourly ED Arrivals)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "arrival_forecast_all_horizons.png", dpi=150)
    plt.close(fig)

    summary_lines = ["Arrival Forecasting (LightGBM) - Test Set Metrics"]
    summary_lines.append("=" * 60)
    for h in FORECAST_HORIZONS:
        m = metrics[h]
        summary_lines.append(
            f"t+{h:>2d}h  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  WAPE={m['wape']:.2%}"
        )
    (REPORTS_DIR / "forecasting_metrics.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / "forecasting_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _log(f"[Forecasting] Summary -> {REPORTS_DIR / 'forecasting_metrics.txt'}")


def run_risk_classifier(clinical_csv: Path) -> None:
    _log("[Risk] Loading patient clinical records...")
    df = pd.read_csv(clinical_csv)

    le_gender = LabelEncoder()
    le_cc = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["chief_complaint_enc"] = le_cc.fit_transform(df["chief_complaint_category"].astype(str))

    encoders = {
        "gender": {"classes": le_gender.classes_.tolist()},
        "chief_complaint": {"classes": le_cc.classes_.tolist()},
    }
    (MODELS_DIR / "encoders.json").write_text(json.dumps(encoders, indent=2), encoding="utf-8")
    _log(f"[Risk] Encoders persisted -> {MODELS_DIR / 'encoders.json'}")

    feature_cols = [
        "age",
        "gender_enc",
        "esi_level",
        "chief_complaint_enc",
        "comorbidity_index",
        "heart_rate",
        "sys_bp",
        "dia_bp",
        "spo2",
        "temp_c",
        "lactate",
        "ed_wait_time_min",
        "is_isolation_required",
        "arrival_hour",
        "day_of_week",
        "is_surge_arrival",
    ]
    target_col = "icu_escalation_flag"

    data = df[feature_cols + [target_col]].dropna()
    X = data[feature_cols]
    y = data[target_col]

    _log(f"[Risk] Samples={len(X)}  Positive rate={y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)

    _log(f"[Risk] Test ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    model.save_model(str(MODELS_DIR / "xgboost_icu.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = _roc_curve(y_test, prob)
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax1.set_title("ROC - ICU Escalation Risk")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    precision, recall, _ = _pr_curve(y_test, prob)
    ax2.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    ax2.set_title("Precision-Recall - ICU Escalation Risk")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "icu_risk_roc_pr.png", dpi=150)
    plt.close(fig)

    _log("[Risk] Computing SHAP explanations (TreeExplainer on test subsample)...")
    try:
        import shap

        sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            sample,
            feature_names=feature_cols,
            show=False,
            max_display=15,
        )
        fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - SHAP is optional-hardening
        _log(f"[Risk] SHAP skipped: {exc}")

    (REPORTS_DIR / "risk_model_metrics.txt").write_text(
        f"ICU Escalation Risk Classifier (XGBoost) - Test Metrics\n"
        f"{'=' * 60}\n"
        f"ROC-AUC = {roc_auc:.4f}\n"
        f"PR-AUC  = {pr_auc:.4f}\n"
        f"Positive rate (test) = {y_test.mean():.2%}\n",
        encoding="utf-8",
    )
    _log(f"[Risk] Summary -> {REPORTS_DIR / 'risk_model_metrics.txt'}")


def _roc_curve(y_true, y_score):
    from sklearn.metrics import roc_curve

    return roc_curve(y_true, y_score)


def _pr_curve(y_true, y_score):
    from sklearn.metrics import precision_recall_curve

    return precision_recall_curve(y_true, y_score)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_forecasting(DATA_DIR / "ed_hourly_arrivals.csv")
    run_risk_classifier(DATA_DIR / "patient_clinical_records.csv")
    _log("Forecasting + risk model pipeline complete.")


if __name__ == "__main__":
    main()
