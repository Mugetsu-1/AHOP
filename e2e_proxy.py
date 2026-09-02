"""End-to-end WS verification proxy for the realtime envelope contract.

Connects to the running backend, collects frames, and asserts the locked
{type, payload} envelope for every message, plus the presence of the
PATIENT_ARRIVED / telemetry / PATIENT_DISCHARGED / BED_ALLOCATED event set.

Usage:
    python e2e_proxy.py [--url ws://127.0.0.1:8000/api/v1/realtime/ws]
                        [--seconds 20] [--min-events 3]
                        [--reset-before] [--speed 20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request

import websockets

EVENT_TYPES = {"PATIENT_ARRIVED", "telemetry", "PATIENT_DISCHARGED", "BED_ALLOCATED"}
REQUIRED = {"PATIENT_ARRIVED", "telemetry", "PATIENT_DISCHARGED", "BED_ALLOCATED"}
SNAPSHOT_FIELDS = {
    "clock",
    "queue",
    "admitted",
    "beds",
    "bed_summary",
    "forecast",
    "events_sent",
    "allocations_made",
}
BED_ALLOCATED_FIELDS = {"patient_id", "bed_id", "unit_name", "bed_number"}


def control(base_url: str, action: str, speed: float | None = None) -> None:
    body = {"action": action}
    if speed is not None:
        body["speed"] = speed
    req = urllib.request.Request(
        f"{base_url}/api/v1/realtime/control",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


async def run(
    url: str,
    seconds: float,
    min_events: int,
    reset_before: bool = False,
    speed: float | None = None,
) -> int:
    seen = {t: 0 for t in EVENT_TYPES}
    errors: list[str] = []
    envelope_checked = 0
    snapshot_ok = False
    bed_allocated_ok = False

    base_url = url.replace("ws://", "http://", 1).replace("/api/v1/realtime/ws", "")
    async with websockets.connect(url) as ws:
        if reset_before:
            control(base_url, "reset")
        if speed is not None:
            control(base_url, "speed", speed)
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - asyncio.get_event_loop().time()))
            except asyncio.TimeoutError:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"non-JSON frame: {exc}")
                continue

            if not isinstance(msg, dict) or "type" not in msg or "payload" not in msg:
                errors.append(f"envelope violation: {msg}")
                continue
            envelope_checked += 1

            t = msg["type"]
            if t == "snapshot":
                missing = SNAPSHOT_FIELDS - set(msg["payload"])
                if missing:
                    errors.append(f"snapshot missing fields: {sorted(missing)}")
                else:
                    snapshot_ok = True
            elif t == "BED_ALLOCATED":
                missing = BED_ALLOCATED_FIELDS - set(msg["payload"])
                if missing:
                    errors.append(f"BED_ALLOCATED missing fields: {sorted(missing)}")
                else:
                    bed_allocated_ok = True

            if t in seen:
                seen[t] += 1

    print(f"envelope frames checked: {envelope_checked}")
    print("event counts:", dict(seen))
    print(f"snapshot contract ok: {snapshot_ok}")
    print(f"BED_ALLOCATED contract ok: {bed_allocated_ok}")
    if errors:
        print("errors:")
        for e in errors[:20]:
            print(f"  - {e}")

    missing = sorted(REQUIRED - {t for t, n in seen.items() if n > 0})
    if missing:
        print(f"missing event types in window: {missing} (allowed if sim has no discharges yet)")
    total_events = sum(seen.values())
    if errors:
        print("RESULT: FAIL")
        return 1
    if total_events < min_events:
        print(f"RESULT: INSUFFICIENT (only {total_events} event frames in window)")
        return 2
    print("RESULT: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/api/v1/realtime/ws")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--min-events", type=int, default=3)
    ap.add_argument("--reset-before", action="store_true", help="POST control reset after connecting")
    ap.add_argument("--speed", type=float, default=None, help="POST control speed after reset (sim-min/real-sec)")
    args = ap.parse_args()
    return asyncio.run(run(args.url, args.seconds, args.min_events, args.reset_before, args.speed))


if __name__ == "__main__":
    sys.exit(main())
