# AHOP — Emergency Department Bed Allocation & Forecasting System

Absolute-detail technical summary, grounded in the current codebase at `D:\AHOP`.

---

## 1. Project Overview

AHOP is a decision-support system that helps hospital emergency departments (ED) decide **which patient should be placed in which bed**, given:

- a live estimate of each patient's **ICU escalation risk** (XGBoost classifier + SHAP explanations),
- a forecast of **hourly patient arrivals** for the next 24 hours (LightGBM time-series model),
- a **bed inventory** with unit, telemetry and isolation capabilities, and
- a **Mixed-Integer Linear Program (MILP)** that assigns patients to beds while respecting hard clinical constraints and minimizing wait time, unit mismatch, and transfer distance.

The system is a full-stack app: FastAPI backend (REST + SQLite), React/Vite/Tailwind dashboard, plus a Python analytics/simulation workspace. All data is **synthetically generated** (no real PHI).

---

## 2. Architecture

```
data/  ──►  training scripts  ──►  models/  ──►  FastAPI app  ──►  React dashboard
  │            (src/ml,                    │         │
  │             src/analysis,              │         └─ SQLite (ahop.db)
  │             src/data_generation)       │
  └────────────────────────────────────────┴─► reports/ (metrics + figures)
```

- **Data layer:** `data/patient_clinical_records.csv` (189,222 rows) and `data/ed_hourly_arrivals.csv` (17,520 rows).
- **ML/analytics workspace:** `src/` — synthetic data generation, forecasting + risk model training/evaluation, survival analysis, EDA, and the MILP solver.
- **Trained artifacts:** `models/encoders.json`, `models/xgboost_icu.json`, `models/lightgbm_arrival_t1h.txt`, `lightgbm_arrival_t6h.txt`, `lightgbm_arrival_t24h.txt`.
- **Service layer:** FastAPI app in `app/` (models, schemas, routers, services), DB seed in `app/seed.py`.
- **Runtime DB:** SQLite at `D:\AHOP\ahop.db` (created on startup or by seed).
- **Presentation:** React SPA in `frontend/` talking to `/api/v1/*`.

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.14 (`cpython-314`), Windows/PowerShell |
| Backend | FastAPI + Uvicorn, Pydantic v2 (`model_validator`) |
| ORM/DB | SQLAlchemy 2, SQLite (`check_same_thread=False`) |
| Optimization | PuLP + HiGHS solver |
| ML | XGBoost, LightGBM, scikit-learn, SHAP |
| Survival | lifelines (Kaplan-Meier + Cox PH) |
| Frontend | React 19, Vite 8, Tailwind CSS 4 (`@tailwindcss/vite`), oxlint |
| Testing | pytest (5 passing smoke tests) |

---

## 4. Repository Layout

```
D:\AHOP\
├── app/
│   ├── main.py                  # FastAPI app, CORS, router mounting, startup table creation
│   ├── config.py                # env-driven settings (see §8)
│   ├── database.py              # engine, SessionLocal, Base, get_db()
│   ├── models.py                # Patient, TriageEvent, Bed, BedAllocation
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── triage.py            # POST /api/v1/triage/assess
│   │   ├── allocation.py        # POST /api/v1/allocation/optimize
│   │   └── dashboard.py         # GET /api/v1/dashboard/metrics
│   ├── services/
│   │   ├── icu_risk.py          # XGBoost risk inference + chief-complaint categorization
│   │   ├── forecast.py          # LightGBM arrival forecast (recursive t+1h model)
│   │   └── allocation.py        # MILP orchestration, risk tiers, pending-patient queries
│   └── seed.py                  # DB bootstrap (beds + 189k synthetic patients/triage events)
├── src/
│   ├── data_generation/generate_synthetic_data.py
│   ├── ml/bed_allocation_solver.py      # MILP (standalone CLI + library)
│   ├── ml/forecasting_and_risk.py       # LightGBM + XGBoost training/eval, figures
│   └── analysis/survival_analysis.py, eda_analysis.py
├── models/                      # trained artifacts (encoders.json, xgboost_icu.json, 3× lightgbm)
├── data/                        # patient_clinical_records.csv, ed_hourly_arrivals.csv
├── reports/                     # metrics JSON/TXT + figures/ (9 artifacts)
├── docs/                        # SAD_REPORT.md, IEEE_PAPER.md
├── tests/test_api.py            # 5 API smoke tests
├── frontend/                    # React dashboard (src/, vite.config.js, package.json)
├── requirements.txt, ahops.db, README.md, summary.md
```

---

## 5. Data Pipeline

### 5.1 Synthetic generator (`src/data_generation/generate_synthetic_data.py`)
- `SEED = 42`, 2 years of hourly timesteps starting `2024-01-01 00:00:00 UTC`.
- 17,520 hourly timesteps; ~250 ED visits/day on average → **189,222 patient records**.
- ESI acuity distribution: `{1: 0.05, 2: 0.20, 3: 0.45, 4: 0.20, 5: 0.10}`.
- Chief-complaint categories: Cardiovascular, Respiratory, Trauma, Gastrointestinal, General.
- Hourly arrivals, daily/weekly/monthly seasonality, surge flags (`is_surge_arrival`, `surge_type`).

### 5.2 Patient dataset (`data/patient_clinical_records.csv` — 189,222 rows × 21 cols)
```
patient_id, mrn, arrival_datetime_utc, arrival_hour, day_of_week, is_surge_arrival,
age, gender, esi_level, chief_complaint_category, comorbidity_index,
heart_rate, sys_bp, dia_bp, spo2, temp_c, lactate,
los_hours, ed_wait_time_min, is_isolation_required, icu_escalation_flag
```
Example row: `181e291a-…, MRN104332181, 2025-12-21 20:00:00, 20, 6, 0, 64, M, 4, General, 1, 102.0, 142.0, 68.0, 99.7, 37.5, 4.01, 5.02, 178, 0, 0`.

### 5.3 Arrival dataset (`data/ed_hourly_arrivals.csv` — 17,520 rows × 6 cols)
```
timestamp_utc, hour_of_day, day_of_week, month, arrivals, is_surge, surge_type
2024-01-01 00:00:00+00:00, 0, 0, 1, 11, 0, NONE
```

### 5.4 Seeding (`app/seed.py`)
- CLI: `python -m app.seed` (use `--seed --reset` to rebuild from scratch).
- Bed plan (800 beds):
  | Unit | Beds | Type |
  |---|---|---|
  | ICU_NORTH | 60 | ICU |
  | ICU_SOUTH | 40 | ICU |
  | TELEMETRY_WEST | 100 | Telemetry |
  | TELEMETRY_EAST | 100 | Telemetry |
  | GENERAL_1…4 | 120/120/130/130 | General |
- Bed IDs: `{ICU|TELE|GEN}-{NNN}`. Telemetry-equipped if prefix ICU/TELE; isolation-capable if `index % 5 < 2`.
- `seed_clinical` reads the CSV, scores each row with `score_frame()` (XGBoost), writes Patient + TriageEvent rows in chunks of 5,000.

---

## 6. Database Schema (SQLite `ahop.db`)

- **`patients`** — `patient_id` String(36) PK (uuid4), `mrn` unique nullable, `age`, `gender`, `is_isolation_required` bool default False, `created_at` (UTC).
- **`triage_events`** — composite PK `(event_id, recorded_at)`; `patient_id` FK → patients; `esi_level`, `chief_complaint` String(255) non-null, `heart_rate`, `sys_bp`, `dia_bp`, `spo2`, `temp_c`, `icu_escalation_prob`, `recorded_at`.
- **`beds`** — `bed_id`, `unit_name`, `bed_number`, `status` (`OCCUPIED`/else available), telemetry + isolation capability flags.
- **`bed_allocations`** — one row per assigned patient linking patient → bed.

---

## 7. ML / Analytics Methodology

### 7.1 ICU escalation risk (XGBoost) — `src/ml/forecasting_and_risk.py`, `app/services/icu_risk.py`
- Target: `icu_escalation_flag` — a 12-hour deterioration proxy.
- 16 features (`FEATURE_COLS`): age, gender_enc, esi_level, chief_complaint_enc, comorbidity_index, heart_rate, sys_bp, dia_bp, spo2, temp_c, lactate, ed_wait_time_min, is_isolation_required, arrival_hour, day_of_week, is_surge_arrival.
- Chief-complaint keyword categorizer (Cardiovascular: chest/cardiac/palpitation/heart/angina; Respiratory: breath/respir/asthma/cough/wheeze/pneumonia; Trauma: trauma/fall/injury/fracture/laceration/burn/accident; Gastrointestinal: abdominal/abdomen/nausea/vomit/diarrhea/gi/stomach; else "General").
- Model: `models/xgboost_icu.json` + `models/encoders.json`; explanation via SHAP TreeExplainer (`shap_summary.png`).
- **Metrics:** ROC-AUC **0.8211**, PR-AUC **0.6033**, positive rate **22.00%** (`reports/risk_model_metrics.txt`, `reports/figures/icu_risk_roc_pr.png`).

### 7.2 Arrival forecasting (LightGBM) — `app/services/forecast.py`
- Trained per horizon: `lightgbm_arrival_t1h/t6h/t24h.txt` (API uses the t+1h model recursively for the next 24h).
- Features: arrivals at lag 1h/24h/168h, rolling mean 6h, rolling std 24h, hour, weekday, month, `is_surge` (=0).
- **Metrics** (`reports/forecasting_metrics.json` / `.txt`):
  | Horizon | MAE | RMSE | WAPE |
  |---|---|---|---|
  | t+1h | 2.7931 | 3.7543 | 25.41% |
  | t+6h | 2.7842 | 3.7385 | 25.33% |
  | t+24h | 2.8006 | 3.7827 | 25.47% |
- Figure: `reports/figures/arrival_forecast_all_horizons.png`; heatmap `arrival_heatmap.png|html`.

### 7.3 Survival / LOS analysis — `src/analysis/survival_analysis.py`
- Dataset: 189,222 rows, 21 columns.
- Kaplan-Meier median LOS by ESI: ESI1 **12.82h**, ESI2 **7.69h**, ESI3 **4.23h**, ESI4 **2.10h**, ESI5 **1.16h**.
- Pairwise + global log-rank: χ² = **98111.983**, p = 0 (all strata).
- Cox PH: C-index **0.7156**; HRs — ESI 2.055 (p=0), age 1.000 (p=0.9043), sys_bp 1.0002 (p=0.1286), lactate 0.997 (p=0.1298).
- Figures: `km_curves_by_esi.png`, `door_to_bed_latency_by_esi.png`, `vital_los_correlation.png|html`.

---

## 8. Optimization — MILP Bed Allocation

### 8.1 Problem formulation (`src/ml/bed_allocation_solver.py`)
- Variables: binary `x_{i,j}` = patient `i` assigned to bed `j`.
- **Objective:** minimize `Σ (w₁·WaitCost + w₂·MismatchPenalty + w₃·TransferDistance)`
  with weights **wait 1.0 / mismatch 5.0 / distance 1.5**.
- **Constraints:**
  - each patient assigned to ≤ 1 bed: `Σ_j x_ij ≤ 1`
  - each bed used ≤ 1 time: `Σ_i x_ij ≤ 1`
  - **hard acuity floor:** `icu_risk > τ` ⇒ ICU-only bed
  - **hard isolation** for isolation-required patients
  - **soft telemetry penalty** (encourages telemetry beds for medium risk)
- Scale reduction via identical-bed capacity aggregation (~313k → ~58k variables).
- Solver: PuLP + HiGHS. CLI: `python src/ml/bed_allocation_solver.py [--inputs patients.json beds.json --output result.json]`.
- `app/services/allocation.py` wires this to the API: risk tiers (`≥0.5` HIGH, `≥0.25` MEDIUM, else LOW), unit typing (ICU/TELE→Telemetry/General), pending-patient lookback window (`WAITLIST_LOOKBACK_HOURS=48`), excluding already-allocated ids. `max_solver_time_sec`/`enforce_strict_isolation` are accepted and reported but do not change the solve (hard isolation is always enforced).

### 8.2 Current report (`reports/bed_allocation_result.json`)
- Top-level keys: `assignments`, `unassigned`, `objective`, `solve_time_s`, `status`.
- **500 assignments, status `Optimal`, solve time 0.36 s.**
- Assignment fields: `patient_id, bed_id, unit_type, icu_risk, esi_level, isolation_required, wait_minutes, telemetry, isolation_capable`.
- Samples: P0001→B0144 (General, risk 0.02, ESI4, wait 35); P0003→B0271 (ICU, risk 0.3933, ESI3, wait 147); P0004→B0015 (Telemetry, risk 0.3443, ESI3, isolation); P0006→B0104 (ICU, risk 0.6139, ESI2, isolation, wait 199); P0500→B0632 (General, risk 0.0789, ESI3, isolation, wait 254).

---

## 9. Backend API

Base URL: `http://127.0.0.1:8000` (docs at `/docs`). All endpoints under `/api/v1`.

### `GET /health`
`{"status": "ok"}`

### `POST /api/v1/triage/assess`
Body (Pydantic v2; nested `vitals` merged into flat fields via `model_validator(mode="before")`):
```json
{
  "patient_id": "P999", "age": 64, "gender": "M", "esi_level": 3,
  "chief_complaint_category": "Cardiovascular", "comorbidity_index": 1,
  "arrival_hour": 14, "day_of_week": 2, "is_surge_arrival": 0,
  "vitals": {"heart_rate": 102, "sys_bp": 142, "dia_bp": 68, "spo2": 97, "temp_c": 37.5}
}
```
Validation: age `0–120`, esi_level `1–5`, comorbidity_index `≥ 0`. Missing vitals → **422**.
Behavior: computes risk via `predict_icu_risk`; creates the Patient if unknown; records a TriageEvent with `icu_escalation_prob`; returns `{patient_id, icu_escalation_probability, risk_category, recommended_unit, shap_factors: [{feature, impact}]}`.

### `POST /api/v1/allocation/optimize`
Body: `{"max_solver_time_sec": 2.0, "enforce_strict_isolation": true}`.
Returns `{solver_status, execution_time_ms, assignments_made, allocations}`. Internal errors → **500** with detail.

### `GET /api/v1/dashboard/metrics`
Returns:
- `bed_occupancy`: `total_beds`, `occupied_beds`, `available_beds`, `occupancy_pct` (1 dp), `by_unit[{unit_name, total, occupied, available}]` (sorted).
- `arrival_forecast`: `actual[]` + `predicted[]` series of `{timestamp, value}` points.
- `last_updated_utc`: ISO 8601 UTC.

### Config (`app/config.py` — env vars)
| Env | Default |
|---|---|
| `AHOP_DATA_DIR` | `<project>/data` |
| `AHOP_MODELS_DIR` | `<project>/models` |
| `AHOP_REPORTS_DIR` | `<project>/reports` |
| `DATABASE_URL` | `sqlite:///<project>/ahop.db` |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `ICU_RISK_THRESHOLD` | 0.5 |
| `TELEMETRY_RISK_THRESHOLD` | 0.25 |
| `SOLVER_TIME_BUDGET_SEC` | 2.0 |
| `WAITLIST_LOOKBACK_HOURS` | 48 |

---

## 10. Frontend (`frontend/`)

- React 19 + Vite 8 + Tailwind CSS 4 + oxlint (`npm run lint`). Scripts: `dev`, `build`, `preview`.
- `src/main.jsx` → `App.jsx` (header + 3 panels, 60 s auto-refresh).
- `src/api.js` — `getMetrics()`, `runAllocation(maxSolverTimeSec=2.0, enforceStrictIsolation=true)`, `assessTriage(payload)`; shared fetch wrapper surfacing `res.detail`.
- Components: `BedMatrix.jsx` (occupancy grid), `InflowChart.jsx` (actual vs predicted arrivals), `TriageQueue.jsx` (waiting patients).

---

## 11. Tests

`tests/test_api.py` — sets default env to `../data` and `../models` relative to the tests dir, then:
- `test_health` — `GET /health` returns `{"status": "ok"}`.
- `test_triage_assess` — full nested-vitals payload; asserts `patient_id`, `icu_escalation_probability ∈ [0,1]`, `risk_category`, `recommended_unit`, `shap_factors`.
- `test_triage_assess_missing_vitals_422` — missing vitals → 422.
- **5 tests, all passing** (`pytest -q`).

---

## 12. Run Instructions

```powershell
# one-time
D:\AHOP\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\AHOP\.venv\Scripts\python.exe -m app.seed --reset
cd D:\AHOP\frontend; npm install

# terminal 1 — backend
D:\AHOP\.venv\Scripts\python.exe -m uvicorn app.main:app --reload   # → http://127.0.0.1:8000

# terminal 2 — frontend
cd D:\AHOP\frontend; npm run dev                                     # → http://localhost:5173
```

---

## 13. Known Limitations / Future Work

- Arrival forecast model is TFT (Temporal Fusion Transformer) discussed as SOTA future work; current production model is LightGBM (recursive t+1h).
- `max_solver_time_sec` / `enforce_strict_isolation` are accepted but not wired to solver behavior (hard isolation is always enforced); wiring them as configurable knobs is pending.
- SQLite is for local/dev; a production deployment would move to PostgreSQL (SQLAlchemy is DB-agnostic).
- All data is synthetic — the models and numbers are for demonstration/architecture validation, not clinical use.
- API forecast endpoint uses only the t+1h LightGBM model recursively, not the trained t6h/t24h checkpoints.
