"""Engine and session management. Synchronous SQLAlchemy on purpose."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from argus.core.config import get_settings

# pgvector settings applied to every connection.
#
# `hnsw.iterative_scan` is a correctness requirement, not a tuning knob. The
# HNSW index post-filters: it collects `ef_search` approximate neighbours and
# only then applies the WHERE clause. Structural similarity filters to the
# reference pool (~13% of rows), so the default settings return *zero rows*
# for a query that has perfectly good answers -- confirmed against real data
# and in the query planner's own EXPLAIN output.
#
# Setting it here rather than per-query means it cannot be forgotten, and it
# does not depend on transaction scope the way `SET LOCAL` does.
PGVECTOR_OPTIONS = "-c hnsw.iterative_scan=relaxed_order -c hnsw.ef_search=200"

engine = create_engine(
    get_settings().database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={"options": PGVECTOR_OPTIONS},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
