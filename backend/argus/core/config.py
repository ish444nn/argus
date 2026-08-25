"""Application configuration.

All settings come from environment variables (or a repo-root `.env` for local
development). There is exactly one Settings object, cached, imported everywhere.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/argus/core/config.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Repo-root .env first, then a .env in the current working directory.
        # Inside Docker neither exists and values come from the environment.
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"

    # Host-side defaults use non-standard ports: the Compose services are
    # published on 5433/6380 so they never collide with a natively installed
    # PostgreSQL or Redis. Inside Compose these are overridden with the
    # service hostnames and standard ports.
    database_url: str = "postgresql+psycopg://argus:argus@localhost:5433/argus"
    redis_url: str = "redis://localhost:6380/0"

    # Celery uses Redis db 0 as broker and db 1 as result backend. The result
    # backend exists only for the development ping task -- real batch progress
    # lives in the `batch_runs` table (see CLAUDE.md).
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Fraction of each batch that becomes an alert. Applied by ranking the
    # batch, never as a stored probability cutoff -- Phase 2 measured a frozen
    # threshold alerting 0.12% of the test range instead of 1%.
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
