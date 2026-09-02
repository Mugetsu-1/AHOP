# A Real-Time Emergency Department Bed-Allocation and Forecasting System

_A short systems note describing the current architecture of AHOP (realtime-only)._

**AHOP Project · v0.1.0**

---

## Abstract

Hospital emergency departments (ED) must place patients into appropriate beds quickly, balancing clinical urgency (ICU-level risk, isolation needs), waiting time, and patient–unit fit. This note describes a real-time demonstration system that replays a MIMIC-IV-ED-derived event stream, derives a per-patient ICU-escalation risk proxy, forecasts hourly arrivals, and solves a mixed-integer linear program (MILP) to allocate patients to an in-memory 800-bed inventory. A WebSocket hub streams the simulated clock, snapshots, and live allocation events to a browser dashboard. The system is database-free and model-free: all state is process-local and all data originates from the replay layer.

---

## I. Introduction

Bed allocation is a bottleneck in ED throughput and patient safety. A decision-support tool must (a) react continuously to arrivals, discharges, and vitals; (b) respect hard clinical constraints (ICU acuity floor, isolation); and (c) optimize soft objectives (waiting time, unit mismatch, transfer distance) within interactive latency. AHOP demonstrates all three properties on a simulated, reproducible replay of ED activity.

---

## II. System Design

**Data replay.** A mapping script transforms the MIMIC-IV-ED demo tables (ED stays joined with triage on `stay_id`) into an event set `data/replay_events.csv`: times are rebased to minutes-since-first-visit, an ICU-escalation flag is derived (disposition in {ICU, CRITICAL}, or `acuity == 1`, or `o2sat < 90 ∧ sbp < 95`), and vitals are normalized. A deterministic tick engine (`LiveTelemetryReplay`, seeded RNG) advances a simulated clock at a user-controlled speed (default 1 sim-minute per real second), emitting arrivals, discharges, and Gaussian-drifted telemetry every 5 sim-minutes, clipped to physiological bounds.

**Hub.** `LiveReplayHub` maintains an in-memory ED mirror: a triage queue, admitted patients, and 800 beds (100 ICU, 200 telemetry-equipped, 500 general) across ICU_NORTH/SOUTH, TELEMETRY_WEST/EAST, and GENERAL_1–4. Each patient receives a risk proxy `min(ESI_BASE_RISK[esi] + 0.15·escalation, 0.95)` with `ESI_BASE_RISK = {1:0.85, 2:0.55, 3:0.30, 4:0.12, 5:0.05}` (tiers: HIGH ≥ 0.5, MEDIUM ≥ 0.25), and a deterministic isolation flag (`crc32(patient_id) % 10 == 0`). A 24-point hourly arrival forecast is derived from the event window via `bincount(arrival_hour % 24)` with a `±1.96σ` confidence band, refreshed on a fixed tick cadence.

**Allocation.** On every event batch the hub projects the queue and free beds into solver dictionaries and calls the MILP. The formulation minimizes

`w₁·Σ wait + w₂·Σ mismatch + w₃·Σ distance + Σ unassigned·penalty`

with weights (1.0, 5.0, 1.5), subject to single-assignment, class capacity, ICU acuity floor (`risk > 0.5 ⇒ ICU-only`), and isolation capability constraints; medium-risk/general and low-risk/higher-acuity placements incur soft penalties. Interchangeable beds are aggregated into capacity classes, reducing the model from ~313k to ~58k variables. On the reference 500-patient/800-bed instance the solve is **Optimal in 0.36 s** (2.0 s budget) using PuLP with the HiGHS solver.

**Interface.** FastAPI serves `GET /health`, `GET /api/v1/realtime/status`, `POST /api/v1/realtime/control` (actions `start`/`pause`/`resume`/`reset`/`speed`; `speed` ∈ (0, 100]), and `GET /api/v1/realtime/ws`. The WebSocket broadcasts `hello`, `clock`, `snapshot`, `queue_update`, `arrival`, `telemetry`, `discharge`, and `allocation` messages. The replay task starts lazily on the first connection, keeping module imports side-effect-free. A React 19/Vite/Tailwind dashboard consumes the stream and renders queue, bed map, inflow forecast, and event feed.

---

## III. Results

- **Correctness:** 6/6 API smoke tests pass (`pytest -q`), including `total_beds == 800` and control-validation cases.
- **Performance:** MILP solves the reference instance optimally in 0.36 s; the dashboard is fully WebSocket-driven with no polling.
- **Determinism:** seeded replay produces reproducible event streams and derived values.
- **Latency budget:** the 2.0 s FR-4 solve budget is met with an order-of-magnitude margin.

---

## IV. Discussion

The system demonstrates the full decision-support loop at interactive latency without persistence or trained models. Key design choices are the in-memory hub (instant reset, no infrastructure), lazy replay start (clean tests), hard-vs-soft constraint separation (safety over comfort), and class aggregation (tractable MILP). Limitations: the risk score is a heuristic proxy rather than a learned classifier, the forecast is a statistical profile rather than a time-series model, and the bed inventory is synthetic. Production use would source live bed state and a richer risk model.

---

## V. Conclusion

AHOP shows that a real-time ED bed-allocation loop — replay, risk proxy, forecast, and MILP assignment — is achievable within interactive latency on a single process. The architecture is deliberately minimal (no database, no trained model) and provides a clean base for adding learned forecasting and risk components.

---

_Author: AHOP project. Software Architecture Document: `docs/SAD_REPORT.md`; technical deep-dive: `summary.md`._
