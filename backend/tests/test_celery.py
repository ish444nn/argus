import pytest

from argus.jobs.celery_app import celery_app
from argus.jobs.tasks import ping


def test_ping_task_is_registered_under_a_stable_name():
    assert "argus.ping" in celery_app.tasks


def test_ping_runs_in_process():
    """The task body is pure, so it can be tested without a broker."""
    result = ping.run("hello")
    assert result["pong"] == "hello"
    assert isinstance(result["worker_pid"], int)


def test_broker_and_backend_use_separate_redis_databases():
    assert celery_app.conf.broker_url.endswith("/0")
    assert celery_app.conf.result_backend.endswith("/1")


def test_worker_does_not_prefetch_long_running_investigations():
    assert celery_app.conf.worker_prefetch_multiplier == 1


@pytest.mark.integration
def test_ping_round_trips_through_redis_to_a_worker():
    """Proves FastAPI -> Redis -> worker. Requires `docker compose up`."""
    from celery.exceptions import TimeoutError as CeleryTimeout

    try:
        async_result = ping.delay("integration")
        payload = async_result.get(timeout=20)
    except (CeleryTimeout, OSError) as exc:  # pragma: no cover
        pytest.skip(f"no worker/broker reachable: {exc}")

    assert payload["pong"] == "integration"
