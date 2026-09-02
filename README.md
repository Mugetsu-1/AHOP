# AHOP — ED Bed Allocation & Forecasting System

Decision-support system for hospital emergency departments: replays MIMIC-derived ED arrivals over a live WebSocket, estimates ICU escalation risk per patient, forecasts hourly arrivals, and assigns patients to beds via a PuLP/HiGHS MILP solver. Full-stack FastAPI + React dashboard; no database, all state lives in a process-level hub driven by the replay layer.

See [`summary.md`](summary.md) for the complete technical deep-dive and the WebSocket message contract.

## Stack

Python 3.14 · FastAPI + Uvicorn · PuLP + HiGHS · pandas / NumPy · React 19 + Vite 8 + Tailwind 4 · pytest

## Quickstart

```powershell
# 1. Install
D:\AHOP\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd D:\AHOP\frontend; npm install

# 2. Terminal 1 — backend
D:\AHOP\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
#    API: http://127.0.0.1:8000   Swagger: http://127.0.0.1:8000/docs

# 3. Terminal 2 — frontend
cd D:\AHOP\frontend; npm run dev
#    Dashboard: http://localhost:5173 (auto-connects to the realtime WS)
```

No database seeding required — the app is entirely replay-driven from `data/replay_events.csv` (197 events).

## Tests

```powershell
D:\AHOP\.venv\Scripts\python.exe -m pytest -q    # 6 passing API smoke tests
cd D:\AHOP\frontend; npm run lint                # oxlint — 0 warnings / 0 errors
cd D:\AHOP\frontend; npm run build               # vite production build (dist/)
```

## E2E verification

```powershell
D:\AHOP\.venv\Scripts\python.exe e2e_proxy.py --seconds 15 --min-events 4 --reset-before --speed 20
```

Connects to the realtime WebSocket, POSTs a control `reset` plus a lowered speed, and asserts the live lifecycle — `PATIENT_ARRIVED`, telemetry, `BED_ALLOCATED`, `PATIENT_DISCHARGED` — within the window.

## Project Layout

```
backend/
├── app/                  FastAPI app (main, config, schemas, realtime hub)
│   └── routers/          realtime REST + WS router (/api/v1/realtime)
├── ml/                   MILP bed allocation solver (PuLP + HiGHS)
├── streamer/             MIMIC mapping + live telemetry replay engine
└── tests/                API smoke tests
data/                     MIMIC-derived replay events (replay_events.csv — 197 events)
frontend/                 React dashboard (live WS-driven panels)
docs/                     legacy design docs
e2e_proxy.py              WebSocket E2E contract verifier
```

## API (`/api/v1/realtime`)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/realtime/status` | Current sim clock, queue, beds, counters (flat dict) |
| POST | `/realtime/control` | `start` / `pause` / `resume` / `reset` / `speed` (returns flat status) |
| WS | `/realtime/ws` | Live event stream — every message is `{type, payload}` |

WS envelope message types:

| `type` | payload |
|---|---|
| `hello` | clock payload (sent once on connect) |
| `clock` | `sim_min, sim_iso, speed, paused, running, patients_in_ed, patients_seen, discharged, arrivals_remaining, total_patients` |
| `snapshot` | full state: `clock, queue, admitted, beds, bed_summary, forecast, events_sent, allocations_made` |
| `queue_update` | `{queue: [...]}` after each event batch |
| `PATIENT_ARRIVED` | raw replay arrival event |
| `telemetry` | raw replay telemetry event (`patient_id, sim_min, vitals`) |
| `PATIENT_DISCHARGED` | raw replay discharge event |
| `BED_ALLOCATED` | `{patient_id, bed_id, unit_name, bed_number}` — one per placement |

## Config (env vars)

`CORS_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`) · `ICU_RISK_THRESHOLD` (0.5) · `TELEMETRY_RISK_THRESHOLD` (0.25)

## Key Metrics

- 800-bed inventory across ICU / Telemetry / General units, maintained in memory
- MILP allocation (PuLP + HiGHS): 500 assignments, **Optimal** in **0.36 s**
- 24-point hourly arrival forecast with confidence band, derived from the replay window

## Disclaimer

Demo/prototype only — replay data derived from MIMIC-IV demo, not for clinical use.
