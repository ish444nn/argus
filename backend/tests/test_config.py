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


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        # The shape a managed PostgreSQL connection string usually takes.
        (
            "postgresql://postgres:pw@db.example.com:5432/postgres",
            "postgresql+psycopg://postgres:pw@db.example.com:5432/postgres",
        ),
        # The older Heroku-style scheme.
        ("postgres://u:p@h:5432/d", "postgresql+psycopg://u:p@h:5432/d"),
        # Already correct: left exactly as it is.
        (
            "postgresql+psycopg://argus:argus@localhost:5433/argus",
            "postgresql+psycopg://argus:argus@localhost:5433/argus",
        ),
        # A deliberate other driver is not overridden.
        ("postgresql+asyncpg://u:p@h/d", "postgresql+asyncpg://u:p@h/d"),
    ],
)
def test_a_database_url_is_bound_to_the_driver_that_is_installed(supplied, expected):
    """A bare `postgresql://` URL means psycopg2 to SQLAlchemy, and this
    project installs psycopg 3.

    Writing a URL without the `+psycopg` suffix would therefore fail at
    import with `No module named 'psycopg2'` -- an error naming a package the
    project does not install.
    """
    assert Settings(_env_file=None, database_url=supplied).database_url == expected


def test_the_normalised_url_resolves_to_the_psycopg3_dialect():
    """The point of the rewrite, asserted against SQLAlchemy rather than a string."""
    from sqlalchemy.engine import make_url

    settings = Settings(_env_file=None, database_url="postgresql://u:p@h:5432/d")
    dialect = make_url(settings.database_url).get_dialect()

    assert dialect.__module__.endswith("postgresql.psycopg")
    assert not dialect.__module__.endswith("psycopg2")
