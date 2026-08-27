import pytest
from pydantic import ValidationError

from argus.core.config import Settings


def test_defaults_use_stub_llm_so_a_fresh_clone_runs_without_an_api_key():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "stub"
    assert settings.gemini_api_key is None


def test_gemini_provider_requires_an_api_key():
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, llm_provider="gemini", gemini_api_key=None)


def test_gemini_provider_accepts_a_key():
    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="test-key")
    assert settings.gemini_api_key.get_secret_value() == "test-key"


def test_celery_result_backend_defaults_to_redis_db_1():
    settings = Settings(_env_file=None, redis_url="redis://localhost:6380/0")
    assert settings.broker_url == "redis://localhost:6380/0"
    assert settings.result_backend == "redis://localhost:6380/1"


def test_explicit_celery_urls_win():
    settings = Settings(
        _env_file=None,
        redis_url="redis://localhost:6380/0",
        celery_broker_url="redis://other:6379/5",
        celery_result_backend="redis://other:6379/6",
    )
    assert settings.broker_url == "redis://other:6379/5"
    assert settings.result_backend == "redis://other:6379/6"


def test_api_key_is_not_exposed_by_repr():
    """A leaked key in a log line or traceback would be a real incident."""
    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="super-secret")
    assert "super-secret" not in repr(settings)


def test_production_accepts_an_empty_redis_url_as_a_deliberate_absence():
    """The hosted deployment has no broker, and says so.

    Render's free tier covers a web service and a static site but neither
    Redis nor a background worker, so the deployed API dispatches nothing. An
    empty REDIS_URL is how that architecture is declared; the alternative --
    the placeholder `redis://localhost:6379/0` that docs/deployment.md used to
    recommend -- points a container at itself and is refused below.
    """
    settings = Settings(
        _env_file=None,
        app_env="production",
        database_url="postgresql+psycopg://u:p@db.example.com:5432/argus",
        redis_url="",
        cors_origins="https://argus-web.example.com",
    )

    assert settings.has_broker is False
    assert settings.broker_url == ""


def test_production_still_refuses_a_localhost_redis_url():
    with pytest.raises(ValidationError, match="REDIS_URL"):
        Settings(
            _env_file=None,
            app_env="production",
            database_url="postgresql+psycopg://u:p@db.example.com:5432/argus",
            redis_url="redis://localhost:6379/0",
            cors_origins="https://argus-web.example.com",
        )


def test_a_configured_redis_url_means_there_is_a_broker():
    assert Settings(_env_file=None, redis_url="redis://localhost:6380/0").has_broker
