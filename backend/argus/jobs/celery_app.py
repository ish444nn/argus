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
    # Fail fast when the broker is unreachable.
    #
    # Celery's defaults retry a publish for a very long time, so dispatching a
    # task with Redis down made the API request hang rather than return. An
    # analyst pressing a button deserves an error, not a spinner: two quick
    # attempts, then raise, and the route turns that into a 503 naming the
    # cause.
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
        "max_retries": 1,
    },
    task_publish_retry=True,
    task_publish_retry_policy={
        "max_retries": 1,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.5,
    },
)
