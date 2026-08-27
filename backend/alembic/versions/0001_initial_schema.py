"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-25 00:52:57.007023
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Must run before any vector column is created. Also present in the
    # Compose postgres init script so /health is green before migrating,
    # and needed here for a database this project did not initialise itself.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "batch_runs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("timestep", sa.SmallInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=True),
        sa.Column("scored_count", sa.Integer(), nullable=False),
        sa.Column("queued_count", sa.Integer(), nullable=False),
        sa.Column("investigated_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name=op.f("ck_batch_runs_status_valid"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_batch_runs")),
    )
    op.create_index(op.f("ix_batch_runs_timestep"), "batch_runs", ["timestep"], unique=False)
    op.create_table(
        "transactions",
        sa.Column("tx_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("timestep", sa.SmallInteger(), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("features", sa.ARRAY(sa.REAL()), nullable=True),
        sa.Column("in_degree", sa.Integer(), nullable=False),
        sa.Column("out_degree", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "label IN ('illicit', 'licit', 'unknown')", name=op.f("ck_transactions_label_valid")
        ),
        sa.PrimaryKeyConstraint("tx_id", name=op.f("pk_transactions")),
    )
    op.create_index(op.f("ix_transactions_label"), "transactions", ["label"], unique=False)
    op.create_index(op.f("ix_transactions_timestep"), "transactions", ["timestep"], unique=False)
    op.create_table(
        "typology_references",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("typology_id", sa.String(length=120), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("publisher", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("year", sa.SmallInteger(), nullable=True),
        sa.Column("section_heading", sa.String(length=255), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("patterns", sa.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=768), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_typology_references")),
        sa.UniqueConstraint(
            "typology_id",
            "chunk_index",
            name=op.f("uq_typology_references_typology_id_chunk_index"),
        ),
    )
    op.create_index(
        "ix_typology_references_patterns",
        "typology_references",
        ["patterns"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "case_reports",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tx_id", sa.BigInteger(), nullable=False),
        sa.Column("batch_run_id", sa.BigInteger(), nullable=True),
        sa.Column("risk_score", sa.Double(), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("confidence_version", sa.String(length=40), nullable=True),
        sa.Column("queue_tier", sa.String(length=16), nullable=True),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("narrative_source", sa.String(length=16), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "narrative_source IN ('llm', 'template')",
            name=op.f("ck_case_reports_narrative_source_valid"),
        ),
        sa.CheckConstraint(
            "queue_tier IN ('primary', 'secondary')", name=op.f("ck_case_reports_queue_tier_valid")
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'investigating', 'ready', 'failed')",
            name=op.f("ck_case_reports_status_valid"),
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name=op.f("ck_case_reports_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["batch_run_id"],
            ["batch_runs.id"],
            name=op.f("fk_case_reports_batch_run_id_batch_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_case_reports_tx_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_case_reports")),
        sa.UniqueConstraint("tx_id", name=op.f("uq_case_reports_tx_id")),
    )
    op.create_index(
        op.f("ix_case_reports_queue_tier"), "case_reports", ["queue_tier"], unique=False
    )
    op.create_index(op.f("ix_case_reports_status"), "case_reports", ["status"], unique=False)
    op.create_table(
        "edges",
        sa.Column("src_tx_id", sa.BigInteger(), nullable=False),
        sa.Column("dst_tx_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["dst_tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_edges_dst_tx_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["src_tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_edges_src_tx_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("src_tx_id", "dst_tx_id", name=op.f("pk_edges")),
    )
    op.create_index("ix_edges_dst_tx_id", "edges", ["dst_tx_id"], unique=False)
    op.create_table(
        "risk_scores",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("tx_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("batch_run_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("score >= 0 AND score <= 1", name=op.f("ck_risk_scores_score_range")),
        sa.ForeignKeyConstraint(
            ["batch_run_id"],
            ["batch_runs.id"],
            name=op.f("fk_risk_scores_batch_run_id_batch_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_risk_scores_tx_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_risk_scores")),
        sa.UniqueConstraint(
            "tx_id", "model_version", name=op.f("uq_risk_scores_tx_id_model_version")
        ),
    )
    op.create_index("ix_risk_scores_score", "risk_scores", ["score"], unique=False)
    op.create_table(
        "transaction_embeddings",
        sa.Column("tx_id", sa.BigInteger(), autoincrement=False, nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_transaction_embeddings_tx_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("tx_id", name=op.f("pk_transaction_embeddings")),
    )
    op.create_table(
        "evidence_items",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("case_report_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("strength", sa.Double(), nullable=False),
        sa.Column("weight", sa.Double(), nullable=False),
        sa.Column("neighbour_tx_id", sa.BigInteger(), nullable=True),
        sa.Column("typology_reference_id", sa.BigInteger(), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('flagged_neighbour', 'confirmed_neighbour', 'heuristic', 'typology_reference', 'structural_similarity', 'graph_model_corroboration')",
            name=op.f("ck_evidence_items_kind_valid"),
        ),
        sa.CheckConstraint(
            "strength >= 0 AND strength <= 1", name=op.f("ck_evidence_items_strength_range")
        ),
        sa.ForeignKeyConstraint(
            ["case_report_id"],
            ["case_reports.id"],
            name=op.f("fk_evidence_items_case_report_id_case_reports"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["neighbour_tx_id"],
            ["transactions.tx_id"],
            name=op.f("fk_evidence_items_neighbour_tx_id_transactions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["typology_reference_id"],
            ["typology_references.id"],
            name=op.f("fk_evidence_items_typology_reference_id_typology_references"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_items")),
    )
    op.create_index(
        op.f("ix_evidence_items_case_report_id"), "evidence_items", ["case_report_id"], unique=False
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("case_report_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision IN ('confirmed', 'dismissed', 'needs_more_evidence')",
            name=op.f("ck_reviews_decision_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["case_report_id"],
            ["case_reports.id"],
            name=op.f("fk_reviews_case_report_id_case_reports"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_reviews_user_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reviews")),
    )
    op.create_index(op.f("ix_reviews_case_report_id"), "reviews", ["case_report_id"], unique=False)

    # Approximate-nearest-neighbour indexes for the two vector spaces.
    # Cosine distance for both: typology retrieval ranks text similarity,
    # transaction similarity compares GraphSAGE embedding direction.
    op.create_index(
        "ix_typology_references_embedding",
        "typology_references",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_transaction_embeddings_embedding",
        "transaction_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_transaction_embeddings_embedding")
    op.execute("DROP INDEX IF EXISTS ix_typology_references_embedding")
    op.drop_index(op.f("ix_reviews_case_report_id"), table_name="reviews")
    op.drop_table("reviews")
    op.drop_index(op.f("ix_evidence_items_case_report_id"), table_name="evidence_items")
    op.drop_table("evidence_items")
    op.drop_table("transaction_embeddings")
    op.drop_index("ix_risk_scores_score", table_name="risk_scores")
    op.drop_table("risk_scores")
    op.drop_index("ix_edges_dst_tx_id", table_name="edges")
    op.drop_table("edges")
    op.drop_index(op.f("ix_case_reports_status"), table_name="case_reports")
    op.drop_index(op.f("ix_case_reports_queue_tier"), table_name="case_reports")
    op.drop_table("case_reports")
    op.drop_table("users")
    op.drop_index(
        "ix_typology_references_patterns", table_name="typology_references", postgresql_using="gin"
    )
    op.drop_table("typology_references")
    op.drop_index(op.f("ix_transactions_timestep"), table_name="transactions")
    op.drop_index(op.f("ix_transactions_label"), table_name="transactions")
    op.drop_table("transactions")
    op.drop_index(op.f("ix_batch_runs_timestep"), table_name="batch_runs")
    op.drop_table("batch_runs")
    # ### end Alembic commands ###
