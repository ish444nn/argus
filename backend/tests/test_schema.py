"""Schema shape tests.

These run without a database. They guard the decisions that are expensive to
get wrong later: the two vector dimensions, and the fact that evidence
provenance is a real foreign key rather than free text.
"""

from argus.db import enums, models
from argus.db.base import Base

EXPECTED_TABLES = {
    "users",
    "transactions",
    "edges",
    "risk_scores",
    "transaction_embeddings",
    "case_reports",
    "evidence_items",
    "reviews",
    "typology_references",
    "batch_runs",
}


def test_all_locked_tables_exist_and_no_extra_ones_crept_in():
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_vector_dimensions_match_the_locked_architecture():
    typology = Base.metadata.tables["typology_references"].c.embedding
    transaction = Base.metadata.tables["transaction_embeddings"].c.embedding
    assert typology.type.dim == 768, "Gemini text embeddings"
    assert transaction.type.dim == 64, "GraphSAGE node embeddings"


def test_evidence_provenance_is_a_foreign_key_not_free_text():
    evidence = Base.metadata.tables["evidence_items"]
    targets = {
        fk.column.table.name
        for column in (evidence.c.neighbour_tx_id, evidence.c.typology_reference_id)
        for fk in column.foreign_keys
    }
    assert targets == {"transactions", "typology_references"}


def test_check_constraints_cover_every_enum_column():
    checks = {
        (table.name, constraint.name)
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if constraint.__class__.__name__ == "CheckConstraint"
    }
    names = {name for _, name in checks}
    for expected in (
        "ck_transactions_label_valid",
        "ck_case_reports_status_valid",
        "ck_case_reports_queue_tier_valid",
        "ck_evidence_items_kind_valid",
        "ck_reviews_decision_valid",
        "ck_batch_runs_status_valid",
    ):
        assert expected in names, f"missing {expected}; got {sorted(names)}"


def test_confidence_is_nullable_because_it_is_computed_after_investigation():
    assert Base.metadata.tables["case_reports"].c.confidence.nullable


def test_transaction_features_are_nullable_for_the_hosted_demo_snapshot():
    assert Base.metadata.tables["transactions"].c.features.nullable


def test_enum_values_helper_matches_members():
    assert enums.values(enums.Decision) == (
        "confirmed",
        "dismissed",
        "needs_more_evidence",
    )
    assert models.TX_EMBEDDING_DIM == 64
