"""Batch replay control.

Dispatch is by task *name*, not by importing the task function: importing
`argus.jobs.tasks` is fine (its heavy imports are inside the task bodies), but
addressing by name keeps the API from depending on the worker's module graph
at all.

Status is read from `batch_runs`, never from Celery's result backend, so it
survives a worker restart and means the same thing to every caller.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from argus.api.deps import SessionDep, SettingsDep
from argus.api.schemas import BatchRunOut, ReplayDispatched
from argus.jobs.celery_app import celery_app
from argus.services import queue as queue_service

router = APIRouter(prefix="/api/batches", tags=["batches"])


@router.post("/{timestep}/replay", response_model=ReplayDispatched, status_code=202)
def start_replay(timestep: int, session: SessionDep, settings: SettingsDep) -> ReplayDispatched:
    """Queue a replay of one time step.

    Returns 202 immediately: scoring plus evidence gathering takes minutes,
    which is exactly why this is a job and not a request.
    """
    if not settings.replay_min_timestep <= timestep <= settings.replay_max_timestep:
        raise HTTPException(
            status_code=400,
            detail=(
                f"time step {timestep} is outside the replayable range "
                f"{settings.replay_min_timestep}-{settings.replay_max_timestep}. "
                "Earlier time steps are training data and are never scored."
            ),
        )

    async_result = celery_app.send_task("argus.replay_batch", args=[timestep])
    return ReplayDispatched(
        task_id=async_result.id,
        timestep=timestep,
        alert_budget=settings.alert_budget,
        status_url=f"/api/batches/{timestep}",
    )


@router.get("", response_model=list[BatchRunOut])
def list_batches(session: SessionDep) -> list[BatchRunOut]:
    return [BatchRunOut(**run) for run in queue_service.list_batch_runs(session)]


@router.get("/{timestep}", response_model=BatchRunOut)
def get_batch(timestep: int, session: SessionDep) -> BatchRunOut:
    run = queue_service.get_batch_run(session, timestep)
    if run is None:
        raise HTTPException(status_code=404, detail=f"time step {timestep} has not been replayed")
    return BatchRunOut(**run)
