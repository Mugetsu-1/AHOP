"""Realtime replay: WebSocket stream + REST status/control endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..realtime import hub
from ..schemas import RealtimeControlRequest

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.websocket("/ws")
async def realtime_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await hub.register(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.unregister(websocket)


@router.get("/status")
def status() -> dict:
    hub._ensure_beds()
    clock = hub._clock_payload()
    bed_summary = {"total": len(hub.beds), "occupied": 0, "available": 0}
    if hub._beds_loaded:
        bed_summary["occupied"] = sum(
            1 for b in hub.beds.values() if b["status"] == "OCCUPIED"
        )
        bed_summary["available"] = bed_summary["total"] - bed_summary["occupied"]
    return {
        **clock,
        "queue_length": len(hub.queue),
        "admitted": sum(1 for p in hub.patients.values() if p.admitted),
        "available_beds": bed_summary["available"],
        "total_beds": bed_summary["total"],
        "events_sent": hub.events_sent,
        "allocations_made": hub.allocations_made,
    }


@router.post("/control")
async def control(payload: RealtimeControlRequest) -> dict:
    action = payload.action
    if action == "start":
        hub.start()
    elif action == "pause":
        hub.paused = True
    elif action == "resume":
        hub.paused = False
    elif action == "reset":
        await hub.reset()
    elif action == "speed":
        if payload.speed is None:
            raise HTTPException(status_code=400, detail="speed is required for action='speed'")
        hub.replay.set_speed(payload.speed)
    else:
        raise HTTPException(status_code=422, detail=f"unknown action: {action}")
    return status()
