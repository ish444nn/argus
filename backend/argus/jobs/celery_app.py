"""Celery application.

One app, one default queue, one worker. Redis is the broker; the result backend
exists only for the development ping task. Real batch progress is written to the
`batch_runs` table so the UI polls Postgres, not Celery internals.
"""

from celery import Celery

from argus.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "argus",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["argus.jobs.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Investigations are long and LLM-bound; fetching one task at a time keeps
    # work distributed evenly if a second worker is ever added.
    worker_prefetch_multiplier=1,
    result_expires=3600,
)
