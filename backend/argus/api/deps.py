"""Shared FastAPI dependencies.

`Annotated` aliases rather than `Depends(...)` defaults -- the current FastAPI
idiom, and it keeps ruff's B008 quiet.
"""

import logging
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from argus.core.config import Settings, get_settings
from argus.db.session import get_session

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def dispatch(task_name: str, *args: object) -> str:
    """Queue a Celery task, or fail cleanly if the broker is unreachable.

    Without this a dispatch with Redis down hangs the request while Celery
    retries the connection. A 503 naming the broker is far more useful than a
    spinner, and it is the honest answer: the job genuinely cannot be queued.
    """
    from fastapi import HTTPException

    from argus.jobs.celery_app import celery_app

    try:
        # Prove the broker is reachable before publishing. Celery's transport
        # options do not bound the publish path, so `send_task` alone still
        # blocks for tens of seconds when Redis is gone; `ensure_connection`
        # with no retries and a short timeout turns that into an immediate
        # failure we can report.
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=0, timeout=3)
        return celery_app.send_task(task_name, args=list(args)).id
    except HTTPException:
        raise
    except Exception as exc:
        logging.getLogger(__name__).warning("could not queue %s: %s", task_name, exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The job queue is unavailable, so this cannot be started. "
                "Check that Redis and the worker are running."
            ),
        ) from exc
