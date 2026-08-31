"""Prompt 3 (Optimization): Dynamic bed allocation via PuLP MILP.

Decision variable: x_{i,j} in {0,1}  (assign patient i to bed j)
Objective:        min sum_{i,j} (w1*WaitCost_i + w2*MismatchPenalty_ij + w3*TransferDistance_ij) * x_ij
Constraints:
  1. Single assignment:     sum_j x_ij <= 1            for all patients
  2. Bed capacity:          sum_i x_ij <= 1            for all beds
  3. Acuity floor (hard):   icu_risk_i > tau  =>  only ICU beds eligible
  4. Isolation (hard):      isolation_i     =>  only isolation-capable beds eligible
  5. Telemetry preference:  medium risk     =>  telemetry-capable preferred (penalty, not hard)

Performance note: beds sharing (unit_type, telemetry, isolation_capable, location)
are interchangeable for every patient (same eligibility and placement cost), so
they are aggregated into capacity classes. The resulting transportation LP is
exactly equivalent (same optimal objective) yet shrinks the model from ~313k to
~58k variables, keeping the FR-4 2s solve budget on the 500/800 instance.

Inputs (JSON-ish dicts):
    patients: [{patient_id, esi_level (1-5), icu_risk (0-1), isolation_required (bool),
                wait_minutes, current_unit (str), acuity_label}]
    beds:     [{bed_id, unit_type (ICU|Telemetry|General), telemetry (bool),
                isolation_capable (bool), location (str)}]
Output: patient-to-bed mappings JSON + summary statistics.

Usage:
    python src/ml/bed_allocation_solver.py            # runs demo with synthetic queue/inventory
    python src/ml/bed_allocation_solver.py --inputs patients.json beds.json --output result.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pulp

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_WEIGHTS = {"wait": 1.0, "mismatch": 5.0, "distance": 1.5}
ACUITY_THRESHOLD = 0.5  # tau: icu_risk > tau => mandatory ICU placement
DEFAULT_TELEMETRY_RISK_THRESHOLD = 0.25

UNIT_TYPE_ORDER = {"General": 0, "Telemetry": 1, "ICU": 2}


def _log(msg: str) -> None:
    print(msg)


def _eligible_beds(patient: dict, beds: list[dict], telemetry_threshold: float) -> list[int]:
    """Return indices of beds satisfying the hard acuity + isolation constraints."""
    idx: list[int] = []
    for j, bed in enumerate(beds):
        if patient["icu_risk"] > ACUITY_THRESHOLD and bed["unit_type"] != "ICU":
            continue
        if patient.get("isolation_required") and not bed.get("isolation_capable", False):
            continue
        idx.append(j)
    return idx


def _mismatch_penalty(patient: dict, bed: dict, telemetry_threshold: float) -> float:
    """Soft placement penalty: placing a patient below their needed care level."""
    if patient["icu_risk"] > ACUITY_THRESHOLD:
        # hard constraint already filters; if still mismatched (shouldn't happen) penalize hard
        return 0.0 if bed["unit_type"] == "ICU" else 1000.0
    if patient["icu_risk"] >= telemetry_threshold:
        if bed["unit_type"] == "ICU":
            return 0.0
        if bed.get("telemetry", False) or bed["unit_type"] == "Telemetry":
            return 0.0
        return 10.0  # general bed for medium-risk patient
    # low risk
    return 0.0 if bed["unit_type"] == "General" else 2.0  # prefer not to occupy higher care bed


def _transfer_distance(patient: dict, bed: dict) -> float:
    """Surrogate transfer cost: unit-type step + intra-ED distance between locations."""
    src = UNIT_TYPE_ORDER.get(str(patient.get("current_unit", "General")).capitalize(), 0)
    dst = UNIT_TYPE_ORDER.get(str(bed.get("unit_type", "General")).capitalize(), 0)
    unit_steps = abs(src - dst)

    p_loc = str(patient.get("location", "0"))
    b_loc = str(bed.get("location", "0"))
    try:
        loc_dist = abs(int(p_loc) - int(b_loc))
    except (TypeError, ValueError):
        loc_dist = 0.0
    return float(unit_steps * 5.0 + loc_dist)


def _aggregate_beds(beds: list[dict]) -> list[dict]:
    """Group interchangeable beds into capacity classes.

    Beds sharing (unit_type, telemetry, isolation_capable, location) are perfect
    substitutes for every patient (identical eligibility and placement cost), so
    the solver works with one variable per (patient, class) and a single capacity
    constraint per class instead of per bed.
    """
    groups: dict[tuple[str, bool, bool, str], dict] = {}
    for j, bed in enumerate(beds):
        key = (
            str(bed.get("unit_type", "")),
            bool(bed.get("telemetry", False)),
            bool(bed.get("isolation_capable", False)),
            str(bed.get("location", "")),
        )
        if key not in groups:
            groups[key] = {
                "unit_type": key[0],
                "telemetry": key[1],
                "isolation_capable": key[2],
                "location": key[3],
                "capacity": 0,
                "bed_indices": [],
            }
        groups[key]["capacity"] += 1
        groups[key]["bed_indices"].append(j)
    return list(groups.values())


def solve_allocation(
    patients: list[dict],
    beds: list[dict],
    weights: dict | None = None,
    telemetry_threshold: float = DEFAULT_TELEMETRY_RISK_THRESHOLD,
) -> dict:
    """Solve the MILP and return {assignments, unassigned, objective, solve_time_s, status}."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    n = len(patients)
    classes = _aggregate_beds(beds)
    k = len(classes)

    # Eligibility + placement cost per (patient, class). Costs depend only on the
    # class attributes, so every bed inside a class yields the identical cost.
    max_wait = max((p.get("wait_minutes", 0) for p in patients), default=1.0) or 1.0
    wait_priority = [p.get("wait_minutes", 0) / max_wait for p in patients]
    eligible: list[list[int]] = [_eligible_beds(p, classes, telemetry_threshold) for p in patients]
    costs: dict[tuple[int, int], float] = {}
    for i in range(n):
        for c in eligible[i]:
            wait_term = -weights["wait"] * wait_priority[i]  # negative => longer waiters placed first
            mismatch = weights["mismatch"] * _mismatch_penalty(patients[i], classes[c], telemetry_threshold)
            distance = weights["distance"] * _transfer_distance(patients[i], classes[c])
            costs[(i, c)] = wait_term + mismatch + distance

    log_re = pulp.LpProblem("BedAllocation", pulp.LpMinimize)

    # NOTE: transportation structure (patient rows + class-capacity rows) is totally
    # unimodular, so continuous [0,1] vars give the same integral optimum as binary.
    x = {
        (i, c): pulp.LpVariable(f"x_{i}_{c}", lowBound=0, upBound=1, cat="Continuous")
        for i in range(n)
        for c in eligible[i]
    }
    # Unassigned slack: u_i = 1 iff patient i is not placed. Heavily penalised so
    # every patient with at least one eligible bed is placed whenever capacity allows;
    # patients whose eligible classes are capacity-exhausted (e.g. more HIGH-risk
    # patients than ICU beds) wait in the queue instead of making the model infeasible.
    u = {
        i: pulp.LpVariable(f"u_{i}", lowBound=0, upBound=1, cat="Continuous")
        for i in range(n)
    }

    # Constraint 1: single assignment per patient (placement or unassigned slack)
    for i in range(n):
        log_re += pulp.lpSum(x[(i, c)] for c in eligible[i]) + u[i] == 1

    # Constraint 2: class capacity (one row per bed class instead of per bed)
    for c in range(k):
        patients_for_class = [x[(i, c)] for i in range(n) if (i, c) in x]
        if patients_for_class:
            log_re += pulp.lpSum(patients_for_class) <= classes[c]["capacity"]

    # Objective: placement cost + heavy penalty for leaving a patient unassigned
    # (sized above any single placement cost so placement always wins).
    unassigned_penalty = 1.0 + max(
        (abs(v) for v in costs.values()), default=0.0
    )
    log_re += (
        pulp.lpSum(costs[(i, c)] * x[(i, c)] for (i, c) in costs)
        + pulp.lpSum(unassigned_penalty * u[i] for i in u)
    )

    start = time.perf_counter()
    status = log_re.solve(pulp.HiGHS(msg=False))
    solve_time_s = time.perf_counter() - start

    # Reconstruct per-bed assignments: each class fills its beds with distinct
    # patients (class placements never exceed capacity at the LP optimum).
    class_bed_used = [0] * k
    assignments = []
    unassigned = []
    for i in range(n):
        placed = [c for c in eligible[i] if x[(i, c)].value() is not None and x[(i, c)].value() > 0.5]
        if placed and class_bed_used[placed[0]] < len(classes[placed[0]]["bed_indices"]):
            c = placed[0]
            bed = beds[classes[c]["bed_indices"][class_bed_used[c]]]
            class_bed_used[c] += 1
            assignments.append(
                {
                    "patient_id": patients[i]["patient_id"],
                    "bed_id": bed["bed_id"],
                    "unit_type": bed["unit_type"],
                    "icu_risk": round(float(patients[i]["icu_risk"]), 4),
                    "esi_level": patients[i].get("esi_level"),
                    "isolation_required": bool(patients[i].get("isolation_required", False)),
                    "wait_minutes": patients[i].get("wait_minutes", 0),
                    "telemetry": bool(bed.get("telemetry", False)),
                    "isolation_capable": bool(bed.get("isolation_capable", False)),
                }
            )
        else:
            unassigned.append(patients[i]["patient_id"])

    return {
        "assignments": assignments,
        "unassigned": unassigned,
        "objective": pulp.value(log_re.objective),
        "solve_time_s": solve_time_s,
        "status": pulp.LpStatus[status],
    }


def _make_demo(queue_size: int = 500, bed_count: int = 800, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Generate a synthetic waiting queue + bed inventory for the demo run."""
    rng = np.random.default_rng(seed)

    esi_weights = np.array([0.05, 0.15, 0.35, 0.30, 0.15])
    patients = []
    for i in range(queue_size):
        esi = int(rng.choice([1, 2, 3, 4, 5], p=esi_weights))
        # ICU risk correlated with ESI severity
        base = {1: 0.85, 2: 0.55, 3: 0.30, 4: 0.12, 5: 0.05}[esi]
        icu_risk = float(np.clip(base + rng.normal(0, 0.12), 0.02, 0.98))
        isolation = bool(rng.random() < 0.10)
        wait_min = int(rng.integers(5, 360))
        current_unit = rng.choice(["General", "Telemetry", "ICU"], p=[0.6, 0.3, 0.1])
        patients.append(
            {
                "patient_id": f"P{i + 1:04d}",
                "esi_level": esi,
                "icu_risk": icu_risk,
                "isolation_required": isolation,
                "wait_minutes": wait_min,
                "current_unit": current_unit,
                "location": str(int(rng.integers(0, 20))),
            }
        )

    unit_split = {"ICU": 0.20, "Telemetry": 0.35, "General": 0.45}
    beds = []
    for j in range(bed_count):
        unit = rng.choice(list(unit_split.keys()), p=list(unit_split.values()))
        beds.append(
            {
                "bed_id": f"B{j + 1:04d}",
                "unit_type": unit,
                "telemetry": unit == "ICU" or (unit == "Telemetry" and rng.random() < 0.8),
                "isolation_capable": bool(rng.random() < 0.25),
                "location": str(int(rng.integers(0, 20))),
            }
        )
    return patients, beds


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic bed allocation via PuLP MILP")
    parser.add_argument("--inputs", nargs=2, metavar=("PATIENTS_JSON", "BEDS_JSON"),
                        help="Optional input JSON files (else runs demo)")
    parser.add_argument("--output", default=str(REPORTS_DIR / "bed_allocation_result.json"))
    args = parser.parse_args()

    if args.inputs:
        patients = json.loads(Path(args.inputs[0]).read_text(encoding="utf-8"))
        beds = json.loads(Path(args.inputs[1]).read_text(encoding="utf-8"))
    else:
        _log("[Solver] No inputs provided; running demo with synthetic queue + inventory (500 patients / 800 beds)...")
        patients, beds = _make_demo()

    _log(f"[Solver] Queue={len(patients)} patients, Beds={len(beds)}")
    result = solve_allocation(patients, beds)
    n_assigned = len(result["assignments"])
    violations = sum(
        1 for a in result["assignments"] if a["icu_risk"] > ACUITY_THRESHOLD and a["unit_type"] != "ICU"
    )

    _log(f"  Status       : {result['status']}")
    _log(f"  Objective    : {result['objective']:.2f}")
    _log(f"  Assigned     : {n_assigned}/{len(patients)}")
    _log(f"  Unassigned   : {len(result['unassigned'])}")
    _log(f"  Solve time   : {result['solve_time_s']:.3f}s  (FR-4 target < 2.0s)")
    _log(f"  Acuity viol. : {violations}  (must be 0)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log(f"  Result JSON  : {args.output}")


if __name__ == "__main__":
    main()
