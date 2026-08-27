"""Health endpoint tests.

The unit tests stub the dependencies so they run anywhere. The integration test
hits the real Compose stack and is skipped when it is not up.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from argus.api.main import app
from argus.db.session import get_session


class _FakeSession:
    """Answers the two queries /health makes."""

    def __init__(self, *, vector_installed: bool = True, fail: bool = False):
        self.vector_installed = vector_installed
        self.fail = fail

    def execute(self, statement):
        if self.fail:
            raise RuntimeError("connection refused")
        if "pg_extension" in str(statement):
            return _FakeResult("0.8.6" if self.vector_installed else None)
        return _FakeResult(1)


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


@pytest.fixture
def override_session():
    def _apply(session):
        app.dependency_overrides[get_session] = lambda: session

    yield _apply
    app.dependency_overrides.clear()


def test_health_reports_every_dependency(client, override_session, monkeypatch):
    override_session(_FakeSession())
    # No broker in a unit test, and the worker probe would otherwise wait out
    # its timeout on every call.
    monkeypatch.setattr(
        "argus.jobs.celery_app.celery_app.control.ping",
        lambda **kwargs: [{"celery@test": {"ok": "pong"}}],
    )
    body = client.get("/health").json()

    assert set(body["dependencies"]) == {
        "api",
        "postgres",
        "pgvector",
        "redis",
        "worker",
    }
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["pgvector"]["status"] == "ok"
    assert body["environment"] == "test"


def test_missing_pgvector_extension_is_reported_distinctly_from_a_healthy_postgres(
    client, override_session
):
    override_session(_FakeSession(vector_installed=False))
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["dependencies"]["postgres"]["status"] == "ok"
    assert body["dependencies"]["pgvector"]["status"] == "error"


def test_a_broken_database_does_not_produce_two_identical_errors(client, override_session):
    override_session(_FakeSession(fail=True))
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["dependencies"]["postgres"]["status"] == "error"
    assert body["dependencies"]["pgvector"]["detail"] == "postgres unreachable"


def test_a_deployment_without_a_broker_is_not_degraded(client, override_session):
    """No broker configured is an architecture, not a fault.

    The hosted deployment runs the read side only -- batch replay and
    investigation happen locally and their results are seeded in. Reporting
    that as `error` would have made a correctly configured production API look
    permanently broken on its own health page.
    """
    from argus.api.main import app
    from argus.core.config import Settings, get_settings

    override_session(_FakeSession())
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, redis_url="")
    try:
        body = client.get("/health").json()
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert body["dependencies"]["redis"]["status"] == "disabled"
    assert body["dependencies"]["worker"]["status"] == "disabled"
    assert body["status"] == "ok"


def test_health_returns_200_even_when_degraded(client, override_session):
    """The dashboard renders the breakdown, so the body must always arrive."""
    override_session(_FakeSession(fail=True))
    assert client.get("/health").status_code == 200


@pytest.mark.integration
def test_health_against_the_real_stack():
    from argus.db.session import SessionLocal

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"compose stack not reachable: {exc}")

    with TestClient(app) as real_client:
        body = real_client.get("/health").json()

    assert body["status"] == "ok", body["dependencies"]
    assert body["dependencies"]["pgvector"]["status"] == "ok"
    assert body["dependencies"]["redis"]["status"] == "ok"
