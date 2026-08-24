"""API response models."""

from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Overall readiness plus a per-dependency breakdown.

    Always returned with HTTP 200 so the dashboard can render which dependency
    is down instead of just seeing a failed request.
    """

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    dependencies: dict[str, DependencyStatus]


class TaskDispatched(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: dict | None = None
