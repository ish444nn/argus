"""Development-only task endpoints.

These exist to verify the FastAPI -> Redis -> worker chain. Real batch control
endpoints arrive in Phase 3 and will report progress from the `batch_runs`
table rather than from Celery's result backend.
"""

from celery.result import AsyncResult
from fastapi import APIRouter

from argus.api.schemas import TaskDispatched, TaskStatus
from argus.jobs.celery_app import celery_app
from argus.jobs.tasks import ping

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("/ping", response_model=TaskDispatched)
def dispatch_ping(message: str = "ping") -> TaskDispatched:
    result = ping.delay(message)
    return TaskDispatched(task_id=result.id)


@router.get("/{task_id}", response_model=TaskStatus)
def get_task(task_id: str) -> TaskStatus:
    result = AsyncResult(task_id, app=celery_app)
    payload = result.result if result.successful() else None
    return TaskStatus(
        task_id=task_id,
        state=result.state,
        result=payload if isinstance(payload, dict) else None,
    )
