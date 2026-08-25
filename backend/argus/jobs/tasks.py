"""Celery tasks.

Two real tasks, as locked in Phase 0.1:

`replay_batch`     score a time step, build its queue, gather deterministic
                   evidence.
`investigate_case` Phase 4 -- the LangGraph investigation, narrative and
                   deterministic confidence.

Plus `ping`, which exists to prove the broker path end to end.

Every heavy import happens **inside** a task body. The API dispatches these
tasks and therefore imports this module, and the API image carries neither
xgboost nor torch; a module-level import would break it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from argus.jobs.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(name="argus.ping")
def ping(message: str = "ping") -> dict[str, Any]:
    """Round-trip a message through the broker to prove the worker is alive."""
    log.info("ping task received: %s", message)
    return {"pong": message, "worker_pid": os.getpid()}


@celery_app.task(name="argus.replay_batch", bind=True)
def replay_batch(self, timestep: int, alert_budget: float | None = None) -> dict[str, Any]:
    """Replay one batch. Idempotent: re-running restates rather than appends.

    Progress and the final outcome are written to `batch_runs`, so a caller
    polls the database rather than Celery's result backend and job state
    survives a worker restart.
    """
    from argus.db.session import SessionLocal
    from argus.services import replay

    log.info("replay_batch starting for timestep %s", timestep)
    with SessionLocal() as session:
        result = replay.replay_batch(session, timestep, alert_budget=alert_budget)

    log.info("replay_batch finished for timestep %s: %d queued", timestep, result.queued_count)
    return result.to_dict()


@celery_app.task(
    name="argus.investigate_case",
    bind=True,
    # Retry on transient failures -- a rate-limited or briefly unreachable
    # model call should not lose the job. Retries are safe because the task is
    # idempotent: it replaces the case's typology evidence and report rather
    # than appending, and never touches the deterministic evidence.
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=3,
)
def investigate_case(self, case_id: int) -> dict[str, Any]:
    """Run the LangGraph investigation for one case."""
    from argus.db.session import SessionLocal
    from argus.services import investigation

    log.info("investigate_case starting for case %s", case_id)
    with SessionLocal() as session:
        result = investigation.investigate(session, case_id)

    log.info(
        "investigate_case finished for case %s: confidence %s, tier %s",
        case_id,
        result.get("confidence"),
        result.get("queue_tier"),
    )
    return result
