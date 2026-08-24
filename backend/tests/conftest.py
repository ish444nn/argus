import os

import pytest

# Set before any argus module imports Settings, so the cached settings object
# is built in test mode with the stub LLM provider.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_PROVIDER", "stub")

from fastapi.testclient import TestClient  # noqa: E402

from argus.api.main import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
