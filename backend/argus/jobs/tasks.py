"""Celery tasks.

Phase 1 ships only `ping`, which exists to prove the
FastAPI -> Redis -> worker chain end to end.

Phase 3/4 will add exactly two real tasks: `replay_batch` and `investigate_case`.
"""

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
