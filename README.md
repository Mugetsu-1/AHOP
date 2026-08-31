# AHOP — ED Bed Allocation & Forecasting System

Decision-support system for hospital emergency departments: predicts ICU escalation risk per patient (XGBoost + SHAP), forecasts hourly arrivals (LightGBM), and assigns patients to beds via a PuLP/HiGHS MILP solver. Full-stack FastAPI + React dashboard; all data is synthetic.

See [`summary.md`](summary.md) for the complete technical deep-dive.

## Stack

Python 3.14 · FastAPI · SQLAlchemy/SQLite · PuLP + HiGHS · XGBoost · LightGBM · SHAP · lifelines · React 19 + Vite 8 + Tailwind 4 · pytest

## Quickstart

```powershell
# 1. Install
D:\AHOP\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd D:\AHOP\frontend; npm install

# 2. Seed the database (800 beds + 189k patients / triage events)
D:\AHOP\.venv\Scripts\python.exe -m app.seed --reset

# 3. Terminal 1 — backend
D:\AHOP\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
#    API: http://127.0.0.1:8000   Swagger: http://127.0.0.1:8000/docs

# 4. Terminal 2 — frontend
cd D:\AHOP\frontend; npm run dev
#    Dashboard: http://localhost:5173
```

## Tests

```powershell
pytest -q     # 5 passing API smoke tests
```

## Project Layout

```
app/            FastAPI app (routers, services, models, schemas, seed)
src/            ML workspace: data generation, forecasting/risk, solver, analyses
models/         Trained artifacts (xgboost_icu.json, 3× lightgbm, encoders.json)
data/           Synthetic CSVs (patients, hourly arrivals)
reports/        Metrics + figures (risk, forecast, survival, allocation)
frontend/       React dashboard (BedMatrix, InflowChart, TriageQueue)
tests/          API smoke tests
docs/           SAD_REPORT.md, IEEE_PAPER.md
```

## API (`/api/v1`)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/triage/assess` | ICU risk score + SHAP explanation (nested `vitals` accepted) |
| POST | `/allocation/optimize` | Run MILP bed assignment |
| GET | `/dashboard/metrics` | Occupancy + arrival forecast for the dashboard |

## Config (env vars)

`AHOP_DATA_DIR` · `AHOP_MODELS_DIR` · `AHOP_REPORTS_DIR` · `DATABASE_URL` (default `sqlite:///ahop.db`) · `CORS_ORIGINS` · `ICU_RISK_THRESHOLD` (0.5) · `TELEMETRY_RISK_THRESHOLD` (0.25) · `SOLVER_TIME_BUDGET_SEC` (2.0) · `WAITLIST_LOOKBACK_HOURS` (48)

## Key Metrics

- ICU risk model: ROC-AUC **0.8211**, PR-AUC **0.6033**
- Arrival forecast (t+1h): MAE **2.79**, WAPE **25.4%**
- Survival: KM median LOS ESI1 **12.8h** → ESI5 **1.2h**; Cox C-index **0.7156**
- Allocation: 500 assignments, **Optimal** in **0.36 s** (HiGHS)

## Disclaimer

Demo/prototype only — synthetic data, not for clinical use. Future work: TFT arrival model, PostgreSQL, wiring solver config knobs.
# AHOP
