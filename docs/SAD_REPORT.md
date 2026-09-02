# AHOP — Software Architecture Document (Realtime Bed Allocation)

**Version:** 0.1.0 · **Status:** Current (realtime-only) · **Applies to:** `D:\AHOP`

---

## 1. Purpose

This document describes the software architecture of the AHOP Emergency Department (ED) bed allocation & forecasting system. AHOP demonstrates a live, WebSocket-driven decision-support loop:

> replay MIMIC-IV-ED-derived arrivals/vitals → derive an ICU-escalation risk proxy per patient → forecast hourly arrivals → solve a bed-assignment MILP → broadcast clock, snapshots, events, and allocations to a dashboard.

The system is deliberately **database-free and model-free**: all state lives in a process-level hub and all data is produced by the replay layer. This document is the authoritative reference for the current architecture and supersedes any earlier drafts describing a database-backed or trained-model pipeline.

---

## 2. System Context

### 2.1 External actors

| Actor | Interaction |
|---|---|
| **Clinician / demo operator** | Opens the React dashboard; starts/pauses/resets the replay; adjusts replay speed; reads queue, bed map, forecast, and allocation events. |
| **API client** | `GET /health`, `GET /api/v1/realtime/status`, `POST /api/v1/realtime/control`, `GET /api/v1/realtime/ws`. |
| **Data author** | Regenerates `data/replay_events.csv` via the mapping script. |

### 2.2 Runtime environment

- Windows host, Python 3.14 (`cpython-314`) in `D:\AHOP\.venv`.
- Backend: Uvicorn/FastAPI on `127.0.0.1:8000`.
- Frontend: Vite dev server on `http://localhost:5173` (binds `[::1]`); proxies `/api` and `ws` to `127.0.0.1:8000`.
- No external services, message brokers, or databases.

---

## 3. Architectural Goals & Constraints

| ID | Goal / Constraint | How it is met |
|---|---|---|
| **A1** | Deterministic, reproducible demo | Seeded RNG (`rng_seed = 42`) in the replay engine; all derived values (risk, isolation) are pure functions of input data. |
| **A2** | No persistence | Process-level `LiveReplayHub` singleton; in-memory 800-bed inventory. Reset returns to `sim_min = 0` with no I/O. |
| **A3** | Live updates without polling | Single WebSocket (`/api/v1/realtime/ws`) streams `clock`, `snapshot`, `queue_update`, `arrival`, `telemetry`, `discharge`, and `allocation` messages. |
| **A4** | No import side effects | The replay task starts **lazily** on first WS registration, so API tests never spawn background work. |
| **A5** | MILP tractability | Bed-class aggregation shrinks the model from ~313k to ~58k variables; reference 500/800 instance solves **Optimal in 0.36 s** (2.0 s budget). |
| **A6** | Idempotent, fail-safe control | `start` is idempotent; `speed` requires a schema-validated `speed`; unknown actions → 422; solver exceptions swallowed. |
| **C1** | Static files | None served by the backend; the SPA is served by Vite (CORS-restricted). |

---

## 4. Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            FastAPI app                               │
│                                                                      │
│  app/main.py ── router ── app/routers/realtime.py ──► LiveReplayHub │
│    (CORS, /health)           (WS /status /control)    app/realtime.py│
│                                                                      │
│  app/config.py  (CORS_ORIGINS, ICU_RISK_THRESHOLD,                   │
│                  TELEMETRY_RISK_THRESHOLD)                           │
│  app/schemas.py (RealtimeControlRequest)                             │
└──────────┬───────────────────────────────────────────────────────────┘
           │  sys.path injection
           ▼
┌─────────────────────────────┐        ┌───────────────────────────────┐
│  src/streamer               │        │  src/ml                       │
│  ├─ mimic_replay_mapping.py │        │  └─ bed_allocation_solver.py │
│  │   → data/replay_events   │        │      solve_allocation(...)   │
│  └─ live_telemetry_replay.py│        └───────────────────────────────┘
└─────────────────────────────┘
           ▲
           │ reads
   data/replay_events.csv (+ gz/demo)
           ▲ writes
   src/streamer/mimic_replay_mapping.py (from MIMIC-IV-ED demo tables)

   frontend/ (React 19 + Vite 8 + Tailwind 4)
     App.jsx · useRealtime hook · InflowChart · LiveQueue · LiveBeds · EventFeed
```

### 4.1 Component responsibilities

| Component | Responsibility |
|---|---|
| `app/main.py` | Create the FastAPI app, install CORS, mount the realtime router at `/api/v1`, serve `/health`. |
| `app/config.py` | Read env vars; provide typed defaults (`CORS_ORIGINS`, `ICU_RISK_THRESHOLD = 0.5`, `TELEMETRY_RISK_THRESHOLD = 0.25`). |
| `app/schemas.py` | `RealtimeControlRequest` — `action: Literal["start","pause","resume","reset","speed"]`, `speed: float \| None` validated `gt=0, le=100`. |
| `app/routers/realtime.py` | HTTP/WS surface: status snapshot, control actions, WS connection handler. |
| `app/realtime.py` | `LiveReplayHub` — in-memory ED mirror (queue, admitted patients, 800-bed inventory), replay task loop, live MILP allocation, 24-point forecast cache, WebSocket broadcast. |
| `src/streamer/mimic_replay_mapping.py` | Transform MIMIC-IV-ED demo tables into the event set `data/replay_events.csv` (rebased times, derived escalation flag, normalized vitals). |
| `src/streamer/live_telemetry_replay.py` | `LiveTelemetryReplay` — deterministic tick engine emitting arrivals/discharges/telemetry against a simulated clock with speed control. |
| `src/ml/bed_allocation_solver.py` | MILP solver (PuLP + HiGHS). Library entry `solve_allocation()`; standalone CLI with synthetic 500/800 demo. |
| `frontend/` | Dashboard SPA; `useRealtime` hook owns the WS lifecycle and optimistic clock; components render queue, beds, forecast, and events. |
| `tests/test_api.py` | 6 smoke tests for health/status/control; avoid `start`/`reset` so no background task runs. |

---

## 5. Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| In-memory hub singleton | Live-state demo without persistence; trivial reset; fits the FR-4 latency budget. |
| WebSocket-first UI | Continuous clock/bed updates with minimal overhead; single connection per dashboard. |
| Lazy replay start | Keeps module import side-effect-free; `pytest` stays deterministic and fast. |
| Class-aggregated MILP | Transport-structure integrality + ~5.4× variable reduction keeps solves sub-second. |
| Hard acuity floor + isolation, soft mismatch/wait/distance | Safety constraints are non-negotiable; comfort/distance are optimized as weights. |
| Replay-based risk proxy (ESI base + escalation bump) | Honest scope: no trained model on the MIMIC demo rows; the proxy preserves the decision-support shape. |
| Statistical 24-hour forecast (bincount + ±1.96σ band) | Cheap, refreshable arrival profile; future work can swap in a learned forecaster. |

---

## 6. Runtime View — Request / Message Flows

### 6.1 Dashboard connect & replay start

```
Client                          LiveReplayHub
   │  WS /api/v1/realtime/ws          │
   ├──────────────────────────────────►│ register() → start() (idempotent, lazy)
   │◄───────────── hello (clock) ─────┤
   │◄─────────── snapshot (full) ─────┤
   │◄──── clock (every 1.0 s tick) ───┤
   │◄ queue_update / arrival / ───────┤  (per event batch)
   │◄ telemetry / discharge / ────────┤
   │◄ allocation / snapshot ──────────┤  (after MILP solve)
```

### 6.2 Control action semantics

| Action | Behaviour | Failure mode |
|---|---|---|
| `start` | Idempotently ensure replay task is running | — |
| `pause` | Stop advancing the sim clock | — |
| `resume` | Continue advancing the sim clock | — |
| `reset` | Zero queue/admitted/counters/forecast; free all 800 beds; broadcast `clock`+`snapshot`; **preserve speed** | — |
| `speed` | Set clock rate; requires valid `speed` | missing → 400; out of range → 422 |

### 6.3 Event processing per tick

1. `step()` the replay engine → `(arrivals, discharges, telemetry)`.
2. `_handle_arrival`: build `LivePatient` (risk proxy, isolation flag, base vitals), enqueue, broadcast `arrival` + `queue_update`.
3. `_handle_discharge`: remove patient; free bed if admitted; broadcast `discharge` + `queue_update`.
4. `_handle_telemetry`: update vitals; broadcast `telemetry`.
5. If any arrival/discharge occurred → project queue+beds to solver dicts → `solve_allocation(...)`; apply assignments (admit, occupy bed, `allocations_made += 1`); broadcast `allocation` + `snapshot`.
6. Clear the forecast cache on the `FORECAST_REFRESH_TICKS = 60` cadence.
7. Broadcast `clock` every tick.

---

## 7. Data Design

No database. Persistent artifact: `data/replay_events.csv` (columns `stay_id, subject_id, gender, esi_level, chief_complaint, disposition, icu_escalation_flag, arrival_min, discharge_min, temperature_c, resprate, pain, heartrate, sbp, dbp, o2sat`), produced by `mimic_replay_mapping.py` from the MIMIC-IV-ED demo tables (`edstays` + `triage`, joined on `stay_id`; `_demo.csv` fallbacks).

### 7.1 In-memory state (hub)

| Structure | Contents |
|---|---|
| `queue` | `LivePatient` dicts awaiting assignment (risk, isolation, wait, vitals). |
| `admitted` | Patients with a resolved `bed_id`/`bed_number`/`unit_name`. |
| `beds` | 800 `LiveBed`s: `{bed_id, bed_number, unit, unit_type, telemetry, isolation_capable, status}`. |
| `sim_clock` | Float sim-minutes since epoch; `speed` multiplier; `paused` flag. |
| `events_sent`, `allocations_made` | Cumulative counters surfaced in status/snapshot. |
| `forecast_cache` | 24-point profile + `residual_std`, refreshed on the tick cadence. |

### 7.2 Bed plan (800 total)

| Unit | Beds | Type | Telemetry |
|---|---|---|---|
| ICU_NORTH / ICU_SOUTH | 60 / 40 | ICU | yes |
| TELEMETRY_WEST / EAST | 100 / 100 | Telemetry | yes |
| GENERAL_1..GENERAL_4 | 120 / 120 / 130 / 130 | General | no |

Isolation capability: `bed_number_idx % 5 < 2`. Bed identifiers `f"{unit}:{idx:03d}"`, numbers `f"{prefix}-{idx:03d}"`.

### 7.3 Derived values

- **Risk proxy:** `min(ESI_BASE_RISK[esi] + 0.15·escalation, 0.95)`; `ESI_BASE_RISK = {1:0.85, 2:0.55, 3:0.30, 4:0.12, 5:0.05}`. Tiers: HIGH ≥ 0.5, MEDIUM ≥ 0.25, else LOW.
- **Escalation flag:** disposition in {ICU, CRITICAL} OR `acuity == 1` OR (`o2sat < 90` AND `sbp < 95`).
- **Isolation:** `zlib.crc32(patient_id) % 10 == 0`.
- **Forecast:** `bincount(arrival_hour % 24, minlength=24)`; band `±1.96·residual_std`; timestamps anchored at `REPLAY_EPOCH = 2000-01-01T00:00Z`.

---

## 8. Algorithm Design — MILP (`src/ml/bed_allocation_solver.py`)

**Inputs:** patient list (`patient_id, esi_level, icu_risk, isolation_required, wait_minutes, current_unit, acuity_label, location`) and bed list (`bed_id, unit_type, telemetry, isolation_capable, location`); thresholds `icu_threshold = 0.5`, `telemetry_threshold = 0.25`.

**Variables:** continuous `x_{i,c}` ∈ [0,1] (patient → bed class) + unassigned slack `u_i`.

**Objective (minimize):**
```
w₁·Σ wait_time(i)·x   +   w₂·Σ mismatch_penalty(i,c)·x   +   w₃·Σ transfer_distance·x   +   Σ u·big_penalty
   w₁ = 1.0, w₂ = 5.0, w₃ = 1.5
```

**Hard constraints:**
- `Σ_c x_{i,c} + u_i = 1` — every patient assigned at most once.
- `Σ_i x_{i,c} ≤ capacity(c)` — class capacity.
- `icu_risk > 0.5 ⇒ ICU-only bed` — acuity floor (non-negotiable).
- `isolation_required ⇒ isolation_capable bed` — isolation (non-negotiable).

**Soft penalties:** medium-risk → general bed `+10.0`; low-risk → higher-acuity bed `+2.0`; unit-step + intra-ED-location transfer-distance surrogates.

**Scale reduction:** classes group beds with identical `(unit_type, telemetry, isolation_capable, location)`; ~313k → ~58k variables on the 500/800 instance. Integrality holds via total unimodularity of the transport structure.

**Reference result:** 500 patients / 800 beds → status **Optimal**, solve **0.36 s** (budget 2.0 s). CLI: `python src/ml/bed_allocation_solver.py [--inputs ... --output ...]`.

---

## 9. Deployment / Configuration

- Backend: `D:\AHOP\.venv\Scripts\python.exe -m uvicorn app.main:app --reload` (from `D:\AHOP`).
- Frontend: `cd frontend; npm run dev` → `http://localhost:5173`.
- Env vars: `CORS_ORIGINS` (comma-separated), `ICU_RISK_THRESHOLD`, `TELEMETRY_RISK_THRESHOLD`.
- No secrets, no persistence, no external infra.

---

## 10. Quality Attributes

| Attribute | Evidence |
|---|---|
| Correctness | 6/6 API smoke tests pass (`pytest -q`). |
| Performance | MILP solves sub-second on the reference instance; UI fully WS-driven with no polling. |
| Determinism | Seeded replay RNG; reproducible runs. |
| Maintainability | Small, single-responsibility modules; README/summary/this doc consistent with code. |
| Testability | Lazy task start ⇒ tests free of background tasks. |

---

## 11. Known Limitations / Future Work

1. **Risk proxy, not a trained model** — the heuristic (ESI base + escalation bump) has no learned weights; a trained classifier requires richer data.
2. **Statistical forecast** — the 24-hour profile (bincount + σ band) is not a time-series model; a learned forecaster is future work.
3. **Synthetic inventory** — the 800-bed plan and isolation flags are in-memory fixtures; production would bind real bed state.
4. **Event-triggered solving** — the MILP runs on event batches, not continuously.
5. **Demo scope** — data is derived from MIMIC-IV-ED demo material for demonstration, not clinical use.
