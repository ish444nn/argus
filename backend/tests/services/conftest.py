"""Fixtures for tests that need the live stack.

These run against the Compose database with the Elliptic data ingested. They
skip rather than fail when it is not there, so a plain `pytest` still works on
a fresh clone.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

REPLAY_TIMESTEP = 35


@pytest.fixture(scope="session")
def db_session():
    from argus.db.session import SessionLocal

    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - depends on local environment
        pytest.skip(f"database not reachable: {exc}")

    yield session
    session.close()


@pytest.fixture(scope="session")
def ingested(db_session):
    """Skip unless the transaction graph has been loaded."""
    count = db_session.execute(text("SELECT count(*) FROM transactions")).scalar_one()
    if not count:
        pytest.skip("no transactions ingested (`python -m argus.ml.cli ingest`)")
    return count


@pytest.fixture(scope="session")
def embedded(db_session, ingested):
    """Skip unless GraphSAGE embeddings exist."""
    count = db_session.execute(text("SELECT count(*) FROM transaction_embeddings")).scalar_one()
    if not count:
        pytest.skip("no embeddings (`python -m argus.ml.cli embed`)")
    return count


@pytest.fixture(scope="session")
def replayed(db_session, embedded):
    """A replayed batch to assert against.

    Replay is idempotent, so running it here is safe even against a database
    that already has this batch.
    """
    from argus.ml import registry
    from argus.services import replay

    try:
        registry.load_metadata("xgb-all166")
    except FileNotFoundError:
        pytest.skip("no trained model (`python -m argus.ml.cli train`)")

    return replay.replay_batch(db_session, REPLAY_TIMESTEP)
