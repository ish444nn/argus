"""Health endpoint.

Checks the three things the application cannot run without: Postgres, the
pgvector extension, and Redis. pgvector is checked separately from Postgres
because a reachable database with no vector extension fails only later, at
retrieval time, which is a confusing place to discover it.
"""

from functools import lru_cache
from typing import Annotated

import redis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from argus import __version__
from argus.api.schemas import DependencyStatus, HealthResponse
from argus.core.config import Settings, get_settings
from argus.db.session import get_session

router = APIRouter(tags=["health"])

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _check_postgres(session: Session) -> DependencyStatus:
    try:
        session.execute(text("SELECT 1"))
        return DependencyStatus(status="ok")
    except Exception as exc:
        return DependencyStatus(status="error", detail=str(exc)[:200])


def _check_pgvector(session: Session) -> DependencyStatus:
    try:
        version = session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        if version is None:
            return DependencyStatus(status="error", detail="extension 'vector' not installed")
        return DependencyStatus(status="ok", detail=f"v{version}")
    except Exception as exc:
        return DependencyStatus(status="error", detail=str(exc)[:200])


@lru_cache
def _redis_client(url: str) -> redis.Redis:
    """One pooled client per URL.

    The dashboard polls /health every few seconds; building a fresh connection
    pool per request would leak sockets.
    """
    return redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)


def _check_redis(settings: Settings) -> DependencyStatus:
    try:
        _redis_client(settings.redis_url).ping()
        return DependencyStatus(status="ok")
    except Exception as exc:
        return DependencyStatus(status="error", detail=str(exc)[:200])


def _check_worker(settings: Settings) -> DependencyStatus:
    """Is a Celery worker listening?

    Asks the broker which workers have registered rather than dispatching a
    task and waiting: a health check must not queue work, and must not block
    on a worker that is busy or gone.
    """
    try:
        from argus.jobs.celery_app import celery_app

        # `limit=1` returns as soon as one worker answers instead of
        # waiting out the timeout to collect every reply. Without it this
        # check cost a flat second on every call -- and /health is polled by
        # the dashboard and by the platform's own health probe.
        replies = celery_app.control.ping(timeout=0.5, limit=1)
        if not replies:
            return DependencyStatus(
                status="error", detail="no worker responded; investigations will queue"
            )
        return DependencyStatus(status="ok", detail=f"{len(replies)} worker(s)")
    except Exception as exc:
        return DependencyStatus(status="error", detail=str(exc)[:200])


@router.get("/health", response_model=HealthResponse)
def health(session: SessionDep, settings: SettingsDep) -> HealthResponse:
    postgres = _check_postgres(session)
    dependencies = {
        "api": DependencyStatus(status="ok"),
        "postgres": postgres,
        # Skip the extension probe if the connection itself is broken, so the
        # response reports one root cause instead of two identical errors.
        "pgvector": (
            _check_pgvector(session)
            if postgres.status == "ok"
            else DependencyStatus(status="error", detail="postgres unreachable")
        ),
        "redis": _check_redis(settings),
    }
    # The worker is checked through the broker, so it is only meaningful when
    # the broker itself is reachable.
    dependencies["worker"] = (
        _check_worker(settings)
        if dependencies["redis"].status == "ok"
        else DependencyStatus(status="error", detail="broker unreachable")
    )

    # A missing worker degrades the system but does not break the read-only
    # product: the queue, cases and evidence all still serve. Reporting it
    # separately keeps that distinction visible instead of collapsing every
    # fault into one red light.
    degraded = any(dep.status == "error" for dep in dependencies.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        environment=settings.app_env,
        dependencies=dependencies,
    )
