"""ORM models for the full Argus schema.

Created up front (Phase 1) because the schema is already specified in the PRD
and Phase 0.1; building it piecemeal would mean churning migrations every phase.
Tables stay deliberately thin -- no columns are added speculatively.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, REAL
from sqlalchemy.orm import Mapped, mapped_column, relationship

from argus.db import enums
from argus.db.base import Base, created_at_column, updated_at_column

# Dimensions are fixed by the locked architecture: Gemini text embeddings for
# the typology corpus, GraphSAGE node embeddings for transaction similarity.
TEXT_EMBEDDING_DIM = 768
TX_EMBEDDING_DIM = 64


def _check(column: str, enum_cls: type[enums.StrEnum], name: str) -> CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in enums.values(enum_cls))
    return CheckConstraint(f"{column} IN ({allowed})", name=name)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = created_at_column()


class Transaction(Base):
    """One Elliptic transaction. tx_id is the dataset's own txId."""

    __tablename__ = "transactions"

    tx_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    timestep: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # The 166 raw Elliptic features. Nullable because the hosted demo snapshot
    # ships without them -- only the worker needs feature vectors.
    features: Mapped[list[float] | None] = mapped_column(ARRAY(REAL), nullable=True)
    in_degree: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    out_degree: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (_check("label", enums.Label, "label_valid"),)


class Edge(Base):
    """Directed transaction to transaction flow."""

    __tablename__ = "edges"

    src_tx_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.tx_id", ondelete="CASCADE"), primary_key=True
    )
    dst_tx_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.tx_id", ondelete="CASCADE"), primary_key=True
    )

    # Forward lookups are served by the primary key; this index covers reverse
    # neighbourhood queries (which transactions send to this one).
    __table_args__ = (Index("ix_edges_dst_tx_id", "dst_tx_id"),)


class BatchRun(Base):
    """One replayed batch. A batch is one Elliptic timestep."""

    __tablename__ = "batch_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    timestep: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=enums.BatchStatus.PENDING
    )
    model_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # The budget this run actually applied, so a queue size is always
    # explainable after the fact.
    alert_budget: Mapped[float | None] = mapped_column(Double, nullable=True)
    scored_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    queued_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    investigated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        _check("status", enums.BatchStatus, "status_valid"),
        # One run per time step: replay is idempotent, so re-running a
        # batch updates its row rather than appending a second history.
        UniqueConstraint("timestep"),
    )


class RiskScore(Base):
    """A model's illicit-probability for one transaction.

    model_version is a plain string (e.g. "xgb-all166@a1b2c3") rather than a
    foreign key to a model table -- one string keeps every score traceable to a
    training run without adding an eleventh table.
    """

    __tablename__ = "risk_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tx_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("transactions.tx_id", ondelete="CASCADE"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    score: Mapped[float] = mapped_column(Double, nullable=False)
    batch_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("batch_runs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        UniqueConstraint("tx_id", "model_version"),
        CheckConstraint("score >= 0 AND score <= 1", name="score_range"),
        Index("ix_risk_scores_score", "score"),
    )


class TransactionEmbedding(Base):
    """GraphSAGE node embedding, used for structural-similarity evidence.

    One row per transaction: retraining replaces the row rather than versioning
    it, so similarity search never has to filter by model version.
    """

    __tablename__ = "transaction_embeddings"

    tx_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(TX_EMBEDDING_DIM), nullable=False)
    # The model's illicit probability, computed in the same pass as the
    # embedding. Kept here rather than in `risk_scores` because it is a second
    # opinion quoted as evidence, not a score that gates the queue.
    graph_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        Index(
            "ix_transaction_embeddings_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class TypologyReference(Base):
    """One retrievable chunk of the AML typology corpus."""

    __tablename__ = "typology_references"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    typology_id: Mapped[str] = mapped_column(String(120), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    publisher: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    section_heading: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Pattern tags (e.g. fan_out, structuring). Retrieval filters on this first
    # and ranks by cosine second, so an unrelated typology can never be
    # returned for a fired heuristic.
    patterns: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(TEXT_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        UniqueConstraint("typology_id", "chunk_index"),
        Index("ix_typology_references_patterns", "patterns", postgresql_using="gin"),
        Index(
            "ix_typology_references_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class CaseReport(Base):
    """An investigation of one flagged transaction."""

    __tablename__ = "case_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tx_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("transactions.tx_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    batch_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("batch_runs.id", ondelete="SET NULL"), nullable=True
    )
    # Snapshot of the score that caused the case, so the queue can be ordered
    # without joining risk_scores.
    risk_score: Mapped[float] = mapped_column(Double, nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    # Position within its batch, 1 = highest scoring. Stored because it is
    # fixed the moment the batch is ranked, and recomputing a window function
    # on every queue page load would be wasted work.
    queue_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Snapshot of the GraphSAGE second opinion, for display and sorting. The
    # citable form is the `graph_model_corroboration` evidence item; this is
    # the queryable copy, same rationale as `risk_score` above.
    graph_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=enums.CaseStatus.QUEUED, index=True
    )
    # Null until the investigation runs. Computed deterministically from
    # evidence -- never self-reported by the language model.
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    confidence_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    queue_tier: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = updated_at_column()

    evidence_items: Mapped[list["EvidenceItem"]] = relationship(
        back_populates="case_report", cascade="all, delete-orphan"
    )

    __table_args__ = (
        _check("status", enums.CaseStatus, "status_valid"),
        _check("queue_tier", enums.QueueTier, "queue_tier_valid"),
        _check("narrative_source", enums.NarrativeSource, "narrative_source_valid"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
    )


class EvidenceItem(Base):
    """One cited fact supporting a case report.

    Provenance is a real foreign key, not free text: a neighbour claim points at
    the transaction, a typology claim points at the corpus chunk. That is what
    makes a report checkable by the analyst.
    """

    __tablename__ = "evidence_items"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    case_report_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("case_reports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # strength = normalised signal magnitude (0..1); weight = its contribution
    # to the deterministic confidence score.
    strength: Mapped[float] = mapped_column(Double, nullable=False)
    weight: Mapped[float] = mapped_column(Double, nullable=False)
    neighbour_tx_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("transactions.tx_id", ondelete="SET NULL"), nullable=True
    )
    typology_reference_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("typology_references.id", ondelete="SET NULL"), nullable=True
    )
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    case_report: Mapped[CaseReport] = relationship(back_populates="evidence_items")

    __table_args__ = (
        _check("kind", enums.EvidenceKind, "kind_valid"),
        CheckConstraint("strength >= 0 AND strength <= 1", name="strength_range"),
    )


class Review(Base):
    """An analyst's recorded decision on a case."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    case_report_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("case_reports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (_check("decision", enums.Decision, "decision_valid"),)
