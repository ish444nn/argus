"""Batch replay control.

Dispatch is by task *name*, not by importing the task function: importing
`argus.jobs.tasks` is fine (its heavy imports are inside the task bodies), but
addressing by name keeps the API from depending on the worker's module graph
at all.

Status is read from `batch_runs`, never from Celery's result backend, so it
survives a worker restart and means the same thing to every caller.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from argus.api.deps import SessionDep, SettingsDep, dispatch
from argus.api.schemas import (
    AvailableBatches,
    BatchAvailability,
    BatchRemoved,
    BatchRunOut,
    BudgetApplied,
    ReplayDispatched,
)
from argus.services import batches as batch_service
from argus.services import queue as queue_service

router = APIRouter(prefix="/api/batches", tags=["batches"])


def _applied_budget(session: SessionDep) -> float | None:
    """The budget every stored batch was replayed at, or None if they differ."""
    return session.execute(
        text("""
        SELECT CASE WHEN count(DISTINCT alert_budget) = 1 THEN min(alert_budget) END
        FROM batch_runs
        """)
    ).scalar_one_or_none()


@router.post("/apply-budget", response_model=BudgetApplied, status_code=202)
def apply_budget(
    session: SessionDep,
    settings: SettingsDep,
    budget: float = Query(gt=0, le=0.5, description="Alert budget to apply, as a fraction."),
) -> BudgetApplied:
    """Rebuild every replayed batch's queue at a new alert budget.

    The alert budget is not a display preference: it is the fraction of each
    batch that becomes an alert, and the only way to change which transactions
    are alerts is to re-run the selection. So this dispatches the same
    `replay_batch` task the Batches panel does, once per replayed time step,
    with the new budget -- the existing top-k-per-batch rule, not a second
    implementation of it.

    Cheaper alternatives were rejected on purpose. Re-cutting the stored scores
    at read time would give a queue of transactions with no evidence, no
    confidence and no case to open, because evidence is gathered during replay;
    the count would move and the product behind it would not.
    """
    timesteps = list(
        session.execute(text("SELECT timestep FROM batch_runs ORDER BY timestep")).scalars().all()
    )
    if not timesteps:
        raise HTTPException(
            status_code=409,
            detail="No batch has been replayed yet, so there is no queue to rebuild.",
        )

    dispatched = [
        {"timestep": int(ts), "task_id": dispatch("argus.replay_batch", int(ts), budget)}
        for ts in timesteps
    ]
    return BudgetApplied(
        alert_budget=budget,
        timesteps=[int(ts) for ts in timesteps],
        task_ids=[item["task_id"] for item in dispatched],
        status_url="/api/overview",
    )


@router.post("/{timestep}/replay", response_model=ReplayDispatched, status_code=202)
def start_replay(
    timestep: int,
    session: SessionDep,
    settings: SettingsDep,
    budget: float | None = Query(
        default=None,
        gt=0,
        le=0.5,
        description=(
            "Alert budget for this replay. Omit to use the budget the rest of "
            "the queue is already built at, falling back to the configured "
            "default."
        ),
    ),
) -> ReplayDispatched:
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

    # A batch imported while the queue sits at 3% must join it at 3%, or the
    # overview would report one applied budget for some batches and another
    # for the rest.
    effective = budget if budget is not None else _applied_budget(session)
    if effective is None:
        effective = settings.alert_budget

    task_id = dispatch("argus.replay_batch", timestep, effective)
    return ReplayDispatched(
        task_id=task_id,
        timestep=timestep,
        alert_budget=effective,
        status_url=f"/api/batches/{timestep}",
    )


@router.get("/available", response_model=AvailableBatches)
def list_available(session: SessionDep, settings: SettingsDep) -> AvailableBatches:
    """Time steps that can be replayed, and which already have been.

    The UI needs this to offer a batch to import: without it an analyst has to
    guess a number, and guessing a training time step gets a 400.
    """
    replayed = set(session.execute(text("SELECT timestep FROM batch_runs")).scalars().all())
    every = list(range(settings.replay_min_timestep, settings.replay_max_timestep + 1))
    sizes = dict(
        session.execute(
            text(
                "SELECT timestep, count(*) FROM transactions "
                "WHERE timestep BETWEEN :lo AND :hi GROUP BY timestep"
            ),
            {"lo": settings.replay_min_timestep, "hi": settings.replay_max_timestep},
        ).all()
    )
    return AvailableBatches(
        replayable_range=[settings.replay_min_timestep, settings.replay_max_timestep],
        alert_budget=settings.alert_budget,
        batches=[
            BatchAvailability(
                timestep=ts,
                replayed=ts in replayed,
                transactions=int(sizes.get(ts, 0)),
            )
            for ts in every
        ],
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


@router.delete("/{timestep}", response_model=BatchRemoved)
def remove_batch(timestep: int, session: SessionDep, settings: SettingsDep) -> BatchRemoved:
    """Undo a batch's replay, returning the time step to the available list.

    Synchronous, unlike replay: this is a scoped delete rather than minutes of
    scoring, so making it a job would mean a worker had to be running to tidy
    up after one that already ran.

    Reviewed cases are kept -- see `services.batches.remove_batch`.
    """
    if not settings.replay_min_timestep <= timestep <= settings.replay_max_timestep:
        raise HTTPException(
            status_code=400,
            detail=(
                f"time step {timestep} is outside the replayable range "
                f"{settings.replay_min_timestep}-{settings.replay_max_timestep}."
            ),
        )
    if queue_service.get_batch_run(session, timestep) is None:
        raise HTTPException(status_code=404, detail=f"time step {timestep} has not been replayed")

    return BatchRemoved(**batch_service.remove_batch(session, timestep).to_dict())
