"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RealtimeControlRequest(BaseModel):
    action: Literal["start", "pause", "resume", "reset", "speed"]
    speed: float | None = Field(default=None, gt=0, le=100)
