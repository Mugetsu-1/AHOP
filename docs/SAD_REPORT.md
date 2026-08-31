# AHOP — Hospital Bed Allocation & Prediction Platform
## Systems Analysis and Design (SAD) Report

**Version:** 1.0  
**Author:** Chief System Architect  
**Scope:** End-to-end analysis and design of the AHOP platform (triage scoring, arrival forecasting, ICU-risk prediction, and MILP-based bed allocation).

---

## 1. Executive Summary & System Scope

### 1.1 Executive Summary

AHOP is a decision-support platform for Emergency Department (ED) operations. It ingests live triage assessments, scores each patient's probability of ICU escalation with a gradient-boosted model, forecasts hourly ED arrivals over a 24-hour horizon with a recursive LightGBM model, and resolves the resulting bed-placement problem with a mixed-integer linear program (MILP) solved by PuLP/HiGHS.

The system is engineered around a **fast, hard-constraint-respecting optimizer**: clinical rules (acuity floor and isolation requirements) are enforced as hard constraints, while telemetry preference, wait-time equity, and transfer distance are soft objective terms. On a reference instance of **661 pending patients against an 800-bed, 8-unit inventory**, the solver returns an `Optimal` solution in **76 ms**, well inside the 2-second FR-4 budget.

Measured model performance (held-out test sets):

| Capability | Artifact | Metric | Value |
|---|---|---|---|
| ICU escalation risk | XGBoost classifier | ROC-AUC | **0.8211** |
| ICU escalation risk | XGBoost classifier | PR-AUC | **0.6033** |
| Hourly arrivals (t+24h) | LightGBM recursive | MAE / RMSE / WAPE | **2.801 / 3.783 / 25.47%** |
| Bed allocation solve | PuLP/HiGHS MILP | Status / time | **Optimal / 76 ms** |
| ED length-of-stay survival | Cox proportional-hazards | C-index | **0.7156** |

### 1.2 System Scope

**In scope**
- Live triage assessment and persistence (`Patient` + immutable `TriageEvent` rows).
- ICU-escalation probability scoring with explainability signals (top contributing features).
- Next-24h hourly ED arrival forecasting.
- Optimal bed allocation across 8 units / 800 beds with hard clinical constraint enforcement.
- Real-time dashboard metrics (bed occupancy by unit, arrival forecast, queue state).
- Offline training/evaluation pipeline producing model artifacts and benchmark reports.

**Out of scope (targeted follow-up releases)**
- Direct EHR/EMR integration (patients are submitted via REST or seeded synthetically).
- Real-time patient monitoring (vital ingestion is on-assessment, not continuous).
- Mobile application and clinician mobile notifications.
- Multi-hospital / multi-tenant fleet management.
- Automated discharge/re-bedding lifecycle (bed release is currently manual `status` transitions).

### 1.3 Operational Context

| Aspect | Description |
|---|---|
| Users | ED charge nurses, bed coordinators, hospital operations analysts |
| Deployment | Single-host FastAPI service + React SPA; PostgreSQL in production (SQLite in dev) |
| Trigger | `POST /api/v1/triage/assess` per patient; `POST /api/v1/allocation/optimize` on a re-balance cadence |
| Cadence | Forecast refresh on dashboard poll (60 s frontend interval); allocation on demand |

---

## 2. Software Requirements Specification (SRS)

### 2.1 Functional Requirements

**FR-1 — Triage assessment.** The system SHALL accept a triage assessment payload via `POST /api/v1/triage/assess` containing demographics, ESI level, chief complaint, comorbidity index, vitals, and isolation flag, and SHALL return the ICU-escalation probability, risk category, and recommended unit. *(Implemented: `app/routers/triage.py`, `app/services/icu_risk.py`)*

**FR-2 — Patient & triage-event persistence.** On every assessment the system SHALL upsert a `Patient` record and append an immutable, timestamped `TriageEvent` containing the vitals snapshot and computed `icu_escalation_prob`. Duplicate assessments for the same patient SHALL NOT overwrite prior events. *(Implemented: `app/routers/triage.py`, `app/models.py`)*

**FR-3 — Bed inventory management.** The system SHALL maintain an authoritative inventory of **800 beds across 8 units** with status (`AVAILABLE`/`OCCUPIED`), telemetry capability, and isolation capability per bed. *(Implemented: `app/seed.py` — ICU_NORTH 60, ICU_SOUTH 40, TELEMETRY_WEST 100, TELEMETRY_EAST 100, GENERAL_1..4 120/120/130/130)*

**FR-4 — Optimal bed allocation.** The system SHALL, on `POST /api/v1/allocation/optimize`, compute an assignment for every pending patient (48-hour look-back window, excluding already-allocated patients) and return solver status, execution time, assignment count, and per-assignment details. The solve SHALL complete within **2000 ms** on the reference 661-patient/800-bed instance. *(Implemented: `app/routers/allocation.py`, `app/services/allocation.py`, `src/ml/bed_allocation_solver.py`; verified: 76 ms, `Optimal`, 661 assignments)*

**FR-5 — Hard clinical constraints.** The allocation SHALL treat the ICU acuity floor (`icu_escalation_prob > 0.5` → ICU bed only) and isolation requirements (isolation-required patients → isolation-capable bed only) as **hard constraints** that can never be violated. Telemetry preference for medium-risk patients SHALL be a soft penalty. *(Implemented: `_eligible_beds` / `_mismatch_penalty` in `src/ml/bed_allocation_solver.py`)*

**FR-6 — 24-hour arrival forecasting.** The system SHALL produce hourly arrival forecasts for the next 24 hours via a recursive LightGBM t+1h model, with t+6h and t+24h trained variants available, and SHALL expose actuals alongside predictions. *(Implemented: `app/services/forecast.py`, `src/ml/forecasting_and_risk.py`)*

**FR-7 — Dashboard metrics.** The system SHALL expose `GET /api/v1/dashboard/metrics` returning per-unit and global bed occupancy, the arrival forecast series, and a UTC timestamp, for the React dashboard. *(Implemented: `app/routers/dashboard.py`)*

**FR-8 — Audit & non-duplication.** Each optimization run SHALL be tagged with a unique `solver_execution_id`, persist `BedAllocation` rows (`assigned_at`, `expected_discharge_at`), and mark assigned beds `OCCUPIED` so no patient is re-queued and no bed double-assigned. *(Implemented: `app/models.py`, `app/services/allocation.py`)*

### 2.2 Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| **NFR-1** | Performance | Allocation solve ≤ 2.0 s (`SOLVER_TIME_BUDGET_SEC=2.0`); measured 76 ms. Forecast scoring completes in well under 1 s. |
| **NFR-2** | Scalability | Solver uses bed-class aggregation to reduce LP size from ~313k to ~58k variables; models are single-pass batch scorers. Supports the full seeded 189,222-event history without degradation. |
| **NFR-3** | Reliability | Stateless API workers; schema auto-created on startup (`Base.metadata.create_all`); solver failures surface as HTTP 500 with a descriptive detail; Pydantic validation maps user errors to 422. |
| **NFR-4** | Security & Privacy (HIPAA) | Transport encryption via TLS in production; CORS restricted to a configurable allowlist (default `localhost:5173`, `127.0.0.1:5173`); no free-text PHI beyond chief complaint; structured clinical fields minimize exposure; audit trail via immutable triage events and allocation records. |
| **NFR-5** | Data Integrity | Composite primary key on `triage_events(event_id, recorded_at)`; foreign keys across all child tables; single-commit allocation transaction (bed update + allocation insert). |
| **NFR-6** | Maintainability / Operability | Environment-driven configuration (`AHOP_*` env vars) with sane defaults; modular `app/services`, `app/routers`, `src/ml`, `src/analysis` packages; reproducible single-command seeding (`python -m app.seed --reset`); health endpoint (`GET /health`). |

---

## 3. System Architecture Blueprint

### 3.1 Logical Architecture (ASCII)

```
                      ┌──────────────────────────────────────────────────┐
                      │                    CLIENT TIER                   │
                      │   React SPA (Vite :5173)    │   API consumer / CLI │
                      │   BedMatrix · InflowChart · TriageQueue          │
                      └───────────────────────┬──────────────────────────┘
                                              │ /api, /health (Vite proxy)
                                              v
                      ┌──────────────────────────────────────────────────┐
                      │                 API TIER (FastAPI :8000)         │
                      │  GET /health                                     │
                      │  POST /api/v1/triage/assess                      │
                      │  POST /api/v1/allocation/optimize                │
                      │  GET  /api/v1/dashboard/metrics                  │
                      │  CORS middleware · OpenAPI /docs                 │
                      └────────────┬──────────────────┬─────────────────┘
                                   │                  │
              ┌────────────────────v─────┐   ┌────────v───────────────────┐
              │     SERVICE TIER         │   │    OPTIMIZATION TIER       │
              │  services/icu_risk.py    │   │  src/ml/bed_allocation_    │
              │  services/forecast.py    │   │  solver.py (PuLP + HiGHS)  │
              │  services/allocation.py  │   │  hard acuity/isolation     │
              └────────────┬─────────────┘   │  constraints · class agg.  │
                           │                 └────────────┬───────────────┘
              ┌────────────▼─────────────┐                 │
              │        MODEL TIER        │                 │
              │  models/xgboost_icu.json │                 │
              │  models/lightgbm_*.txt   │                 │
              │  models/encoders.json    │                 │
              └────────────┬─────────────┘                 │
                           └──────────────┬────────────────┘
                                          v
                      ┌──────────────────────────────────────────────────┐
                      │            PERSISTENCE TIER (SQLAlchemy)         │
                      │   Dev: SQLite (ahop.db)   ·   Prod: PostgreSQL   │
                      │   patients · triage_events · beds · bed_allocations │
                      └──────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram — Level 0 (Context)

```
         ┌───────────────────────────────────────────────────────────┐
  ED      │                                                           │   Dashboard
  Triage  │                        AHOP SYSTEM                       │    User
  Operator│                                                           │
 ─────────┼─ triage packet (FR-1) ──┐         ┌── metrics (FR-7) ────┼─────────
          │                         │         │                      │
          │    allocate (FR-4)      ▼         │                      │
 ─────────┼─────────────────────▶ AHOP ───────┼──────────────────────┼─────────
          │                    (bed alloc.   │                      │
          │                     + prediction)│                      │
          │                                  │                      │
          └──────────────────────────────────┴──────────────────────┘
                External entities: ED Triage Operator, Dashboard User
                (Future: EHR/EMR, bed-release automation)
```

### 3.3 Data Flow Diagram — Level 1

```
  ┌────────────┐  triage packet    ┌──────────────────┐
  │ ED Triage  ├──────────────────▶│ P1 Triage Assess │────▶ D1 Patients
  │ Operator   │                   └────────┬─────────┘────▶ D2 TriageEvents
  └────────────┘                            │
                                            v
                                   ┌──────────────────┐     D5 Model Artifacts
                                   │ P2 ICU Risk Score├────▶ (xgboost_icu.json)
                                   └────────┬─────────┘
                                            │ icu_escalation_prob, risk_category
                                            v
  ┌────────────┐  optimize trigger  ┌──────────────────┐     D3 Beds (AVAILABLE)
  │ Bed Coord. ├───────────────────▶│ P3 Optimize      ├────▶ D4 BedAllocations
  └────────────┘                    │ (MILP / PuLP)    ├────▶ D3 Beds (→OCCUPIED)
                                   └────────┬─────────┘
                                            │ status / assignments
                                            v
  ┌────────────┐                   ┌──────────────────┐     D5 Model Artifacts
  │ Dashboard  │◀── metrics ───────│ P5 Dashboard     │     (lightgbm_*.txt)
  │ User       │                   │ P4 Forecast (24h)│◀──── D6 Arrival history
  └────────────┘                   └──────────────────┘     (data/ed_hourly_arrivals.csv)
```

**Processes:** P1 Triage Assessment · P2 ICU Risk Scoring · P3 Bed Allocation Optimization · P4 Arrival Forecasting · P5 Dashboard Metrics  
**Data stores:** D1 Patients · D2 TriageEvents · D3 Beds · D4 BedAllocations · D5 Model Artifacts (file system) · D6 Arrival history (CSV/db)

---

## 4. Relational Database Design

### 4.1 Entity-Relationship Diagram

```
   PATIENTS 1 ──── N TRIAGE_EVENTS          PATIENTS 1 ──── N BED_ALLOCATIONS
  ┌────────────────────┐   ▲          ▲     ┌─────────────────────────────┐
  │ patient_id  (PK)   │   │  FK      │ FK  │ allocation_id       (PK)    │
  │ mrn          (UQ)  │   │          │     │ patient_id          (FK)    │
  │ age                │   │          │     │ bed_id              (FK)    │
  │ gender             │   │          │     │ assigned_at                 │
  │ is_isolation_required│ │          │     │ expected_discharge_at       │
  │ created_at         │   │          │     │ actual_discharge_at         │
  └────────────────────┘   │          │     │ solver_execution_id         │
                           │          │     └─────────────┬───────────────┘
  ┌────────────────────┐   │          │                   │ FK
  │ TRIAGE_EVENTS      │   │          │     ┌─────────────▼───────────────┐
  │ event_id     (PK)  │◀──┘          │     │ BEDS                         │
  │ recorded_at  (PK)  │              └────▶│ bed_id            (PK)       │
  │ patient_id   (FK)  │                    │ unit_name                    │
  │ esi_level          │                    │ bed_number                   │
  │ chief_complaint    │                    │ is_telemetry_equipped        │
  │ heart_rate…temp_c  │                    │ is_isolation_capable         │
  │ icu_escalation_prob│                    │ status AVAILABLE|OCCUPIED    │
  └────────────────────┘                    └──────────────────────────────┘
```

### 4.2 PostgreSQL DDL (production target)

```sql
CREATE TABLE patients (
    patient_id            VARCHAR(36)  PRIMARY KEY,
    mrn                   VARCHAR(64)  UNIQUE,
    age                   INTEGER,
    gender                VARCHAR(16),
    is_isolation_required BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at            TIMESTAMP    NOT NULL DEFAULT (now() AT TIME ZONE 'utc')
);

CREATE TABLE triage_events (
    event_id              VARCHAR(36),
    patient_id            VARCHAR(36)  NOT NULL REFERENCES patients(patient_id),
    esi_level             INTEGER      NOT NULL CHECK (esi_level BETWEEN 1 AND 5),
    chief_complaint       VARCHAR(255) NOT NULL,
    heart_rate            INTEGER,
    sys_bp                INTEGER,
    dia_bp                INTEGER,
    spo2                  REAL,
    temp_c                REAL,
    icu_escalation_prob   REAL,
    recorded_at           TIMESTAMP    NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    PRIMARY KEY (event_id, recorded_at)
);
CREATE INDEX idx_triage_patient   ON triage_events (patient_id);
CREATE INDEX idx_triage_recorded  ON triage_events (recorded_at);

CREATE TABLE beds (
    bed_id                VARCHAR(36)  PRIMARY KEY,
    unit_name             VARCHAR(64)  NOT NULL,
    bed_number            VARCHAR(16)  NOT NULL,
    is_telemetry_equipped BOOLEAN      NOT NULL DEFAULT FALSE,
    is_isolation_capable  BOOLEAN      NOT NULL DEFAULT FALSE,
    status                VARCHAR(32)  NOT NULL DEFAULT 'AVAILABLE',
    CONSTRAINT chk_bed_status CHECK (status IN ('AVAILABLE', 'OCCUPIED'))
);
CREATE INDEX idx_beds_status ON beds (status);

CREATE TABLE bed_allocations (
    allocation_id         VARCHAR(36)  PRIMARY KEY,
    patient_id            VARCHAR(36)  NOT NULL REFERENCES patients(patient_id),
    bed_id                VARCHAR(36)  NOT NULL REFERENCES beds(bed_id),
    assigned_at           TIMESTAMP    NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    expected_discharge_at TIMESTAMP,
    actual_discharge_at   TIMESTAMP,
    solver_execution_id   VARCHAR(64)  NOT NULL
);
CREATE INDEX idx_alloc_execution ON bed_allocations (solver_execution_id);
CREATE INDEX idx_alloc_patient   ON bed_allocations (patient_id);
```

> **Note.** The dev/staging layer runs SQLite via `DATABASE_URL=sqlite:///<BASE_DIR>/ahop.db`. The DDL above is the exact production translation of the SQLAlchemy models in `app/models.py`; swap `DATABASE_URL` to a PostgreSQL DSN and run `Base.metadata.create_all` (or the equivalent migration) to promote.

---

## 5. REST API — OpenAPI 3.0 Contracts

Base path: `/api/v1` · Swagger UI: `/docs` · CORS allowlist: env `CORS_ORIGINS`.

### 5.1 `POST /api/v1/triage/assess`

**Request — `application/json`** (superset schema; nested `vitals` merged with flat fields):

```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "age": 67,
  "gender": "M",
  "esi_level": 3,
  "chief_complaint": "chest pain",
  "comorbidity_index": 2,
  "heart_rate": 118,
  "sys_bp": 92,
  "dia_bp": 58,
  "spo2": 0.91,
  "temp_c": 37.8,
  "lactate": 2.4,
  "ed_wait_time_min": 45,
  "is_isolation_required": false,
  "is_surge_arrival": 0,
  "vitals": { "heart_rate": 118, "sys_bp": 92, "dia_bp": 58, "spo2": 0.91, "temp_c": 37.8 }
}
```

**Validation constraints:** `age ∈ [0,120]`, `esi_level ∈ [1,5]`, `comorbidity_index ≥ 0`, `gender ∈ {M,F,O}`; unknown complaint categories fall back to a General bucket. Missing/violated fields → `422`.

**Response `200` — `TriageAssessResponse`:**

```json
{
  "patient_id": "550e8400-e29b-41d4-a716-446655440000",
  "icu_escalation_probability": 0.7241,
  "risk_category": "HIGH_RISK",
  "recommended_unit": "ICU",
  "shap_factors": [
    { "feature": "spO2", "impact": 0.312 },
    { "feature": "heart_rate", "impact": 0.198 },
    { "feature": "sys_bp", "impact": 0.144 },
    { "feature": "esi_level", "impact": 0.121 },
    { "feature": "chief_complaint_category", "impact": 0.087 }
  ]
}
```

**Risk→unit mapping (implemented thresholds):**

| Category | Condition | Recommended unit |
|---|---|---|
| `HIGH_RISK` | `prob > ICU_RISK_THRESHOLD (0.5)` | `ICU` |
| `MEDIUM_RISK` | `prob ≥ TELEMETRY_RISK_THRESHOLD (0.25)` | `TELEMETRY` |
| `LOW_RISK` | otherwise | `GENERAL` |

**Error responses:** `422` (validation/ValueError) · `500` (unexpected server/model failure).

### 5.2 `POST /api/v1/allocation/optimize`

**Request:**

```json
{ "max_solver_time_sec": 2.0, "enforce_strict_isolation": true }
```

> **Compatibility note.** `max_solver_time_sec` (`1 < v ≤ 30`) and `enforce_strict_isolation` are accepted for API stability. In the current implementation isolation is **always** enforced as a hard constraint and the solver runs to optimality (no early-termination timeout); both parameters are no-ops.

**Response `200` — `AllocationOptimizeResponse`:**

```json
{
  "solver_status": "Optimal",
  "execution_time_ms": 76,
  "assignments_made": 661,
  "allocations": [
    {
      "patient_id": "550e8400-…",
      "assigned_bed_id": "bed-uuid",
      "unit_name": "ICU_NORTH",
      "bed_number": "ICU-N-001",
      "expected_wait_reduction_min": 35
    }
  ]
}
```

**Error responses:** `500` with descriptive detail (solver/DB failure).

### 5.3 `GET /api/v1/dashboard/metrics`

**Response `200` — `DashboardMetricsResponse`:**

```json
{
  "bed_occupancy": {
    "total_beds": 800,
    "occupied_beds": 661,
    "available_beds": 139,
    "occupancy_pct": 82.6,
    "by_unit": [
      { "unit_name": "ICU_NORTH", "total": 60, "occupied": 60, "available": 0 },
      { "unit_name": "ICU_SOUTH", "total": 40, "occupied": 40, "available": 0 },
      { "unit_name": "TELEMETRY_WEST", "total": 100, "occupied": 100, "available": 0 },
      { "unit_name": "TELEMETRY_EAST", "total": 100, "occupied": 100, "available": 0 },
      { "unit_name": "GENERAL_1", "total": 120, "occupied": 88, "available": 32 }
    ]
  },
  "arrival_forecast": {
    "actual":   [ { "timestamp": "2026-08-30T09:00:00Z", "value": 4.0 } ],
    "predicted": [ { "timestamp": "2026-08-30T10:00:00Z", "value": 3.6 } ]
  },
  "last_updated_utc": "2026-08-31T08:30:00.000Z"
}
```

### 5.4 `GET /health`

```json
{ "status": "ok" }
```

---

## 6. Security, HIPAA Compliance, and Disaster Recovery

### 6.1 Security Architecture

| Control | Implementation |
|---|---|
| Transport security | TLS termination required in front of the API in production (reverse proxy / LB). Local dev uses plain HTTP on `127.0.0.1:8000`. |
| Browser origin control | CORS middleware restricted to env-configured `CORS_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`). |
| Input validation | Pydantic schema validation on every request (`age`, `esi_level` ranges, enum gender); invalid payloads → `422` before business logic runs. |
| PHI minimization | Structured clinical fields (vitals, scores) only; free-text exposure limited to `chief_complaint` (≤255 chars); no SSN/name fields persisted. `mrn` is optional and unique. |
| Audit trail | Immutable `TriageEvent` rows (composite PK, append-only) and `BedAllocation` rows keyed by `solver_execution_id` provide a full decision record. |
| Secrets | Configuration is env-driven (`DATABASE_URL`, thresholds); no credentials are hard-coded; secrets must be injected via environment/secret manager in production. |

### 6.2 HIPAA-Oriented Data Handling

- **Minimum necessary:** only the attributes needed for risk scoring and placement are collected.
- **Access control:** production deployments must sit behind an authenticated gateway (OIDC/SAML) with role-based access; the API itself is designed to be a protected internal service, not exposed publicly.
- **Auditing & accounting of disclosure:** every model scoring and every allocation decision is recorded (triage event + `solver_execution_id`), enabling reconstruction of any decision for compliance review.
- **BAA:** deployment partners must establish a Business Associate Agreement; data residency and retention policies are configuration concerns of the host institution.

### 6.3 Availability & Disaster Recovery

- **Startup self-healing:** schema auto-creation on service start (`Base.metadata.create_all`) — a fresh instance can be brought up with models + seed data (`python -m app.seed --reset`).
- **Model artifact immutability:** models live on the filesystem (`models/`) and are loaded at request time; artifacts can be restored from version control / object store independently of the database.
- **Data recovery:** production uses PostgreSQL with point-in-time recovery; the full 189,222-event dataset is regenerable from the seeded synthetic pipeline, so a lost dev database is fully reconstructible.
- **Statelessness:** API workers hold no session state, allowing horizontal scale-out behind a load balancer and rolling restarts without downtime.
- **Observability:** `GET /health` liveness probe; allocation performance and forecast metrics are surfaced in `reports/` for SLO monitoring.

---

*End of SAD Report — grounded in the implemented AHOP codebase (`app/`, `src/`, `models/`, `reports/`).*
