# AHOP — Absolute-Detail Technical Summary

Full-stack ED bed allocation and forecasting decision-support demo, restructured into a realtime-only `backend/` layout and verified end-to-end via a WebSocket proxy. This file is the authoritative spec: read it before touching the realtime contract, the replay engine, or the frontend event handling.

**Stack:** Python 3.14 · FastAPI + Uvicorn · PuLP + HiGHS · pandas / NumPy · React 19 + Vite 8 + Tailwind 4 · pytest

---

## 1. System Architecture

```
data/replay_events.csv                MIMIC-derived replay rows (197 events)
        │  (produced by backend/streamer/mimic_replay_mapping.py)
        ▼
backend/streamer/live_telemetry_replay.py     LiveTelemetryReplay — advance_to(clock_min)
        │  emits raw events: arrival / discharge / telemetry
        ▼
backend/app/realtime.py                       LiveReplayHub (module-level singleton `hub`)
        │  in-memory ED mirror: patients, queue, beds, forecasts
        │  maps wire types: arrival→PATIENT_ARRIVED, discharge→PATIENT_DISCHARGED,
        │                   telemetry→telemetry; emits clock/snapshot/queue_update
        │  live allocation via backend/ml/bed_allocation_solver.py (MILP, PuLP + HiGHS)
        ▼
backend/app/routers/realtime.py               WS /realtime/ws + REST /realtime/status, /realtime/control
        │  every WS message is the envelope {type, payload}
        ▼
frontend/src/realtime.js                      envelope-aware WebSocket client
        ▼
frontend/src/components/*.jsx                 React 19 dashboard panels
```

- No database. All state is a process-level singleton `LiveReplayHub` (`backend/app/realtime.py`) that imports lazily (no background task on import, so tests never spawn one). The replay task starts on the first WS connection or an explicit control call.
- The hub reuses the same MILP solver as the REST path, applied to the hub's in-memory queue.

## 2. Run Commands (repo root `D:\AHOP`)

```powershell
# backend
D:\AHOP\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload

# frontend
cd D:\AHOP\frontend; npm install; npm run dev        # http://localhost:5173

# tests
D:\AHOP\.venv\Scripts\python.exe -m pytest -q         # backend/tests — 6 passing
cd D:\AHOP\frontend; npm run lint                     # oxlint — 0 warnings / 0 errors
cd D:\AHOP\frontend; npm run build                    # vite → frontend/dist/
```

Notes:
- Use the repo venv — the system Python only has numpy.
- The app is replay-driven; no database seeding.
- If port 8000 is held by a stale worker, kill the owning PID from `Get-NetTCPConnection -LocalPort 8000`.

## 3. WebSocket Contract (authoritative)

Endpoint: `ws://127.0.0.1:8000/api/v1/realtime/ws`

**Every message** is the envelope `{ "type": <str>, "payload": <object> }`. Frame-counting and the frontend gate on the outer `type`.

### 3.1 Connect handshake
On accept, the hub sends, in order:
1. `hello` — payload is the clock payload (below).
2. `snapshot` — full current state (below).

The hub then `start()`s the replay task.

### 3.2 Message types

| type | frequency | payload |
|---|---|---|
| `hello` | once on connect | clock payload |
| `clock` | every tick (1s real) | clock payload |
| `snapshot` | on reset; after any allocation batch | `{clock, queue[], admitted[], beds[], bed_summary, forecast, events_sent, allocations_made}` |
| `queue_update` | after each event batch | `{queue: [...]}` (patient views) |
| `PATIENT_ARRIVED` | on arrival | raw replay arrival event |
| `telemetry` | per admitted patient every 5 sim-min | `{type:"telemetry", patient_id, sim_min, vitals}` |
| `PATIENT_DISCHARGED` | on discharge | raw replay discharge event |
| `BED_ALLOCATED` | one per placement | `{patient_id, bed_id, unit_name, bed_number}` |

### 3.3 Clock payload keys
`sim_min` (float), `sim_iso` (ISO-8601, rebased epoch `2000-01-01T00:00Z`), `speed`, `paused`, `running`, `patients_in_ed`, `patients_seen`, `discharged`, `arrivals_remaining`, `total_patients`.

### 3.4 Snapshot payload keys
- `clock` — clock payload
- `queue` / `admitted` — arrays of patient views
- `beds` — array of bed dicts (`bed_id, unit_name, bed_number, is_telemetry_equipped, is_isolation_capable, status`)
- `bed_summary` — `{total: 800, occupied, available}`
- `forecast` — `{actual[{t,value}], predicted[{t,value,lower,upper}], residual_std}` (24 hourly buckets, CI = value ± 1.96·std, refresh every 60 ticks)
- `events_sent`, `allocations_made` — counters

### 3.5 Patient view keys
`patient_id, mrn, esi_level, gender, chief_complaint, icu_escalation_flag, icu_risk, risk_tier, isolation_required, arrival_min, discharge_min, wait_minutes, admitted, bed_id, unit_name, bed_number, vitals`.

### 3.6 Raw event payloads (arrival / telemetry / discharge)
These are forwarded verbatim from the replay engine and therefore carry a nested `"type"` key in the payload (e.g. `payload.type == "arrival"`). Consumers should key on the **outer** envelope type.

- `PATIENT_ARRIVED` payload: `{type:"arrival", patient_id, mrn, esi_level, gender, chief_complaint, disposition, icu_escalation_flag, arrival_min, discharge_min, sim_min, vitals}`
- `telemetry` payload: `{type:"telemetry", patient_id, sim_min, vitals}`
- `PATIENT_DISCHARGED` payload: `{type:"discharge", patient_id, discharge_min, sim_min, disposition, icu_escalation_flag}`

`vitals` keys: `temperature_c, resprate, pain, heartrate, sbp, dbp, o2sat`.

## 4. REST API (`/api/v1/realtime`)

- `GET /health` → `{"status": "ok"}`
- `GET /realtime/status` → **flat dict** (NOT the envelope): clock payload keys plus `queue_length, admitted, available_beds, total_beds, events_sent, allocations_made`.
- `POST /realtime/control` — body `{"action": <str>, "speed": <float|None>}` where `action ∈ {start, pause, resume, reset, speed}`. Returns the same flat status dict.
  - `speed` requires `speed` > 0 (400 if missing); `0 < speed ≤ 100` (schema bound `gt=0, le=100`). Unknown action → 422.
  - Frontend pattern: `clock`/`hello` → `setClock(msg.payload)` guarded by `if (res?.sim_iso)`; `snapshot` → `setSnapshot(msg.payload)` and mirror clock; `queue_update` → merge queue; `PATIENT_ARRIVED` / `telemetry` / `PATIENT_DISCHARGED` / `BED_ALLOCATED` → capped event feed (MAX_EVENTS=60).

## 5. Replay Engine (`backend/streamer/live_telemetry_replay.py`)

- Loads `data/replay_events.csv`, sorts by `arrival_min`, keeps a rebased sim clock (sim-minute 0 == earliest ED visit).
- `step()` advances the clock by `speed` sim-minutes per call; `advance_to(clock_min)` emits events in order **arrivals → discharges → telemetry**.
- Defaults: `tick_seconds=1.0`, `speed=1.0`, `telemetry_interval_min=5.0`, `rng_seed=42`.
- Vitals drift each telemetry sample with seeded noise clipped to `VITAL_BOUNDS`.
- `data/replay_events.csv`: 198 lines incl. header → **197 events** (matches `total_patients`).
- The demo timeline is intentionally sparse — most ticks emit nothing but a clock tick. Do not "fix" this.

## 6. Live Risk & Allocation (`backend/app/realtime.py`)

- ICU risk proxy (no XGBoost at runtime): `ESI_BASE_RISK {1:0.85, 2:0.55, 3:0.30, 4:0.12, 5:0.05}` + `0.15` escalation bump, capped 0.95.
- Risk tier: `icu_risk ≥ 0.5` → HIGH, `≥ 0.25` → MEDIUM, else LOW (thresholds are env-configurable).
- Isolation required: `crc32(patient_id) % 10 == 0`.
- Bed plan (800 beds, in-memory, id `{unit}:{idx:03d}`, number `{prefix}-{idx:03d}`):
  - ICU_NORTH 60, ICU_SOUTH 40, TELEMETRY_WEST 100, TELEMETRY_EAST 100, GENERAL_1 120, GENERAL_2 120, GENERAL_3 130, GENERAL_4 130.
  - Telemetry-equipped prefixes: ICU, TELE. Isolation-capable: `idx % 5 < 2`.

## 7. MILP Solver (`backend/ml/bed_allocation_solver.py`)

- PuLP + HiGHS. Objective weights: wait penalty 1.0, acuity mismatch 5.0, distance 1.5, plus soft penalties for medium-risk→General and low-risk→higher-acuity placements, with a slack penalty above any placement.
- Acuity floor: patients above the ICU threshold may only go to ICU beds.
- Capacity classes aggregate to ~58k candidate pairs (from ~313k raw), keeping the solve fast.
- Demo run: 500 assignments from 800 beds — **Optimal, 0.36 s**.

## 8. Frontend (`frontend/`)

- React 19 + Vite 8 + Tailwind 4. Panels: `LiveQueue`, `LiveBeds`, `InflowChart` (forecast), `EventFeed` (capped at 60 events).
- `realtime.js` is the envelope-aware WS client — connect once, dispatch on `msg.type`, mirror `snapshot`/`clock` into state.
- Lint: oxlint clean. Build: vite production build clean (`frontend/dist/`).

## 9. E2E Verification (`e2e_proxy.py`, repo root)

```powershell
D:\AHOP\.venv\Scripts\python.exe e2e_proxy.py --seconds 15 --min-events 4 --reset-before --speed 20
```

- Connects to the WS **first**, then POSTs control `reset` (and an optional `--speed N`) — the ordering makes the run deterministic.
- Asserts the full lifecycle within the window: `PATIENT_ARRIVED`, telemetry, `BED_ALLOCATED`, `PATIENT_DISCHARGED`, and validates the snapshot and `BED_ALLOCATED` payload contracts.
- Expected result: `RESULT: PASS` with envelope-frame and per-type event counts.
- Base URL derived from the WS url (`ws://`→`http://`, strip `/api/v1/realtime/ws`).

## 10. Config (env vars)

| var | default |
|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `ICU_RISK_THRESHOLD` | `0.5` |
| `TELEMETRY_RISK_THRESHOLD` | `0.25` |

## 11. Test Matrix (`backend/tests/test_api.py`)

`/health` · `/realtime/status` (keys + `total_beds == 800`) · control `speed` · control `speed` missing → 400 · control invalid action → 422 · control `pause`/`resume` round-trip. All pass.

## 12. Disclaimer

Demo/prototype only — replay data derived from MIMIC-IV demo; not for clinical use.
