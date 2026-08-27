"""Application configuration.

All settings come from environment variables (or a repo-root `.env` for local
development). There is exactly one Settings object, cached, imported everywhere.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/argus/core/config.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The one PostgreSQL driver this project installs. SQLAlchemy picks a driver
# from the URL scheme, and a bare `postgresql://` still means **psycopg2**,
# which is not a dependency here and never will be -- `psycopg[binary]` (v3) is.
DB_DRIVER = "postgresql+psycopg"


def normalise_database_url(url: str) -> str:
    """Bind a PostgreSQL URL to the driver that is actually installed.

    SQLAlchemy chooses a driver from the URL scheme, and a bare
    `postgresql://` (or the older `postgres://`) selects psycopg2, which is
    not installed here. Writing a URL without the `+psycopg` suffix would
    otherwise fail at import with `ModuleNotFoundError: No module named
    'psycopg2'` -- an error that names a package the project does not use and
    points nowhere near the mistake.

    Rewriting the scheme here rather than in `create_engine` means Alembic,
    the worker, the API and every test read the same corrected URL. A URL that
    already names a driver is left alone, so `postgresql+asyncpg://` or a
    deliberate `+psycopg2` still says what it says.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return DB_DRIVER + "://" + url[len(prefix) :]
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Repo-root .env first, then a .env in the current working directory.
        # Inside Docker neither exists and values come from the environment.
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test"] = "local"
    log_level: str = "INFO"

    # Origins allowed to call the API from a browser. The Vite dev server
    # proxies `/api` in normal use, so this matters only when the frontend is
    # served from a different origin than the API.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Host-side defaults use non-standard ports: the Compose services are
    # published on 5433/6380 so they never collide with a natively installed
    # PostgreSQL or Redis. Inside Compose these are overridden with the
    # service hostnames and standard ports.
    database_url: str = "postgresql+psycopg://argus:argus@localhost:5433/argus"
    redis_url: str = "redis://localhost:6380/0"

    # Celery uses Redis db 0 as broker and db 1 as result backend. The result
    # backend exists only for the development ping task -- real batch progress
    # lives in the `batch_runs` table.
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Fraction of each batch that becomes an alert. Applied by ranking the
    # batch and taking the top slice, never as a stored probability cutoff:
    # score distributions shift between time steps, and a frozen cutoff
    # collapses recall on the later ones.
    alert_budget: float = Field(default=0.01, gt=0, le=1)
    # Time steps the demo replay is allowed to touch. Earlier ones are training
    # data; scoring them would be meaningless.
    replay_min_timestep: int = 35
    replay_max_timestep: int = 49

    llm_provider: Literal["gemini", "stub"] = "stub"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = Field(default=768, ge=1)

    @field_validator("database_url")
    @classmethod
    def _bind_the_installed_driver(cls, value: str) -> str:
        return normalise_database_url(value)

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _check_llm_credentials(self) -> "Settings":
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set")
        return self

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        if self.celery_result_backend:
            return self.celery_result_backend
        base, _, _ = self.redis_url.rpartition("/")
        return f"{base}/1" if base else self.redis_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
