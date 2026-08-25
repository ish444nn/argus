"""Replay, evidence and similarity against the live stack.

All marked `integration`: they need Postgres with the Elliptic graph ingested,
embeddings computed and a trained model on disk. Each skips rather than fails
when a prerequisite is missing.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import text

from argus.agent.tools import graph_tools, similarity
from argus.db.enums import BatchStatus, CaseStatus, EvidenceKind, Label
from argus.ml.splits import TEST_TIMESTEPS, TRAIN_TIMESTEPS, VAL_TIMESTEPS
from argus.services import queue as queue_service
from argus.services import replay as replay_service

from .conftest import REPLAY_TIMESTEP

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def test_replay_selects_exactly_the_alert_budget(db_session, replayed):
    batch_size = db_session.execute(
        text("SELECT count(*) FROM transactions WHERE timestep = :ts"),
        {"ts": REPLAY_TIMESTEP},
    ).scalar_one()

    expected = math.ceil(batch_size * replayed.alert_budget)
    assert replayed.scored_count == batch_size
    assert replayed.queued_count == expected


def test_replay_records_the_batch_run(db_session, replayed):
    run = queue_service.get_batch_run(db_session, REPLAY_TIMESTEP)

    assert run is not None
    assert run["status"] == BatchStatus.COMPLETED.value
    assert run["scored_count"] == replayed.scored_count
    assert run["queued_count"] == replayed.queued_count
    assert run["alert_budget"] == replayed.alert_budget
    # Every score must be traceable to the model that produced it.
    assert run["model_version"].startswith("xgb-")
    assert run["finished_at"] is not None


def test_queued_cases_are_the_highest_scoring_transactions(db_session, replayed):
    """The queue must be the top of the ranking, not an arbitrary subset."""
    rows = db_session.execute(
        text("""
        SELECT r.tx_id, r.score, (c.id IS NOT NULL) AS queued
        FROM risk_scores r
        JOIN transactions t ON t.tx_id = r.tx_id
        LEFT JOIN case_reports c ON c.tx_id = r.tx_id
        WHERE t.timestep = :ts
        ORDER BY r.score DESC
        """),
        {"ts": REPLAY_TIMESTEP},
    ).all()

    queued_flags = [row.queued for row in rows]
    # Everything queued sits above everything not queued.
    assert queued_flags[: replayed.queued_count] == [True] * replayed.queued_count
    assert not any(queued_flags[replayed.queued_count :])


def test_queue_ranks_are_dense_and_start_at_one(db_session, replayed):
    ranks = (
        db_session.execute(
            text("""
        SELECT c.queue_rank FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank
        """),
            {"ts": REPLAY_TIMESTEP},
        )
        .scalars()
        .all()
    )

    assert ranks == list(range(1, replayed.queued_count + 1))


def test_rank_one_has_the_highest_score(db_session, replayed):
    top = db_session.execute(
        text("""
        SELECT c.risk_score, c.queue_rank FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.risk_score DESC LIMIT 1
        """),
        {"ts": REPLAY_TIMESTEP},
    ).one()
    assert top.queue_rank == 1


def test_replay_is_idempotent(db_session, replayed):
    """Re-running must restate, not duplicate."""
    before = _snapshot(db_session, REPLAY_TIMESTEP)
    again = replay_service.replay_batch(db_session, REPLAY_TIMESTEP)
    after = _snapshot(db_session, REPLAY_TIMESTEP)

    assert before == after
    assert again.queued_count == replayed.queued_count
    assert again.scored_count == replayed.scored_count
    assert again.batch_run_id == replayed.batch_run_id  # one run per time step


def _snapshot(session, timestep: int) -> dict:
    return (
        session.execute(
            text("""
        SELECT
          (SELECT count(*) FROM batch_runs WHERE timestep = :ts) AS runs,
          (SELECT count(*) FROM case_reports c JOIN transactions t ON t.tx_id=c.tx_id
            WHERE t.timestep = :ts) AS cases,
          (SELECT count(*) FROM evidence_items e JOIN case_reports c ON c.id=e.case_report_id
            JOIN transactions t ON t.tx_id=c.tx_id WHERE t.timestep = :ts) AS evidence,
          (SELECT count(*) FROM risk_scores r JOIN transactions t ON t.tx_id=r.tx_id
            WHERE t.timestep = :ts) AS scores
        """),
            {"ts": timestep},
        )
        .one()
        ._asdict()
    )


def test_cases_are_created_with_their_scores_and_model_version(db_session, replayed):
    row = db_session.execute(
        text("""
        SELECT c.risk_score, c.model_version, c.status, c.queue_rank, c.batch_run_id
        FROM case_reports c JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank LIMIT 1
        """),
        {"ts": REPLAY_TIMESTEP},
    ).one()

    assert 0.0 <= row.risk_score <= 1.0
    assert row.model_version
    assert row.status == CaseStatus.QUEUED.value
    assert row.batch_run_id == replayed.batch_run_id


def test_replay_refuses_a_time_step_with_no_transactions(db_session, embedded):
    with pytest.raises(ValueError, match="no transactions"):
        replay_service.replay_batch(db_session, 999)

    # The run row is created before anything that can fail, so the failure is
    # visible -- but a non-existent time step must not linger in the overview.
    db_session.execute(text("DELETE FROM batch_runs WHERE timestep = 999"))
    db_session.commit()


# --------------------------------------------------------------------------
# Deterministic graph evidence
# --------------------------------------------------------------------------


def test_neighbourhood_profile_matches_the_edge_table(db_session, ingested):
    tx_id = db_session.execute(
        text("SELECT tx_id FROM transactions WHERE in_degree > 3 LIMIT 1")
    ).scalar_one()

    profile = graph_tools.neighbourhood_profile(db_session, tx_id)
    counted = db_session.execute(
        text("""
        SELECT (SELECT count(*) FROM edges WHERE dst_tx_id = :tx) AS in_deg,
               (SELECT count(*) FROM edges WHERE src_tx_id = :tx) AS out_deg
        """),
        {"tx": tx_id},
    ).one()

    assert profile.in_degree == counted.in_deg
    assert profile.out_degree == counted.out_deg
    assert profile.total_degree == profile.in_degree + profile.out_degree
    # Distinct neighbours cannot exceed the number of edge endpoints.
    assert profile.neighbour_count <= profile.total_degree


def test_all_neighbours_are_in_the_same_batch(db_session, ingested):
    """Elliptic has zero cross-time-step edges; this asserts it end to end."""
    tx_id = db_session.execute(
        text("SELECT tx_id FROM transactions WHERE in_degree > 2 AND out_degree > 2 LIMIT 1")
    ).scalar_one()

    profile = graph_tools.neighbourhood_profile(db_session, tx_id)
    assert profile.same_batch_neighbours == profile.neighbour_count


def test_fan_out_heuristic_fires_and_scales(db_session, ingested):
    tx_id = db_session.execute(
        text("""
        SELECT tx_id FROM transactions
        WHERE out_degree >= 8 AND in_degree <= 2
        ORDER BY out_degree DESC LIMIT 1
        """)
    ).scalar_one()

    fired = {
        d.details["heuristic"]: d for d in graph_tools.structural_heuristics(db_session, tx_id)
    }
    assert "fan_out" in fired
    assert 0.0 < fired["fan_out"].strength <= 1.0
    assert fired["fan_out"].details["pattern"] == "structuring"


def test_fan_in_heuristic_fires(db_session, ingested):
    tx_id = db_session.execute(
        text("""
        SELECT tx_id FROM transactions
        WHERE in_degree >= 8 AND out_degree <= 2
        ORDER BY in_degree DESC LIMIT 1
        """)
    ).scalar_one()

    fired = {
        d.details["heuristic"]: d for d in graph_tools.structural_heuristics(db_session, tx_id)
    }
    assert "fan_in" in fired
    assert fired["fan_in"].details["pattern"] == "funnelling"


def test_chain_length_is_one_for_a_high_degree_transaction(db_session, ingested):
    """Only pass-through transactions sit in a chain."""
    tx_id = db_session.execute(
        text("SELECT tx_id FROM transactions WHERE in_degree > 5 AND out_degree > 3 LIMIT 1")
    ).scalar_one()
    assert graph_tools.chain_length(db_session, tx_id) == 1


def test_heuristics_do_not_fire_on_an_isolated_shape(db_session, ingested):
    """A plain one-in/one-out transaction is 36% of the graph, so it is not
    evidence on its own -- only a chain of them is."""
    tx_id = db_session.execute(
        text("""
        SELECT t.tx_id FROM transactions t
        WHERE t.in_degree = 1 AND t.out_degree = 1
          AND NOT EXISTS (
            SELECT 1 FROM edges e JOIN transactions n ON n.tx_id = e.dst_tx_id
            WHERE e.src_tx_id = t.tx_id AND n.in_degree <= 1 AND n.out_degree <= 1)
          AND NOT EXISTS (
            SELECT 1 FROM edges e JOIN transactions n ON n.tx_id = e.src_tx_id
            WHERE e.dst_tx_id = t.tx_id AND n.in_degree <= 1 AND n.out_degree <= 1)
        LIMIT 1
        """)
    ).scalar_one_or_none()
    if tx_id is None:
        pytest.skip("no isolated pass-through transaction found")

    names = {d.details["heuristic"] for d in graph_tools.structural_heuristics(db_session, tx_id)}
    assert "relay_chain" not in names
    assert "layering_chain" not in names


# --------------------------------------------------------------------------
# Evidence persistence
# --------------------------------------------------------------------------


def test_evidence_is_persisted_with_provenance(db_session, replayed):
    rows = db_session.execute(
        text("""
        SELECT e.kind, e.strength, e.weight, e.neighbour_tx_id, e.details
        FROM evidence_items e
        JOIN case_reports c ON c.id = e.case_report_id
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts
        """),
        {"ts": REPLAY_TIMESTEP},
    ).all()

    assert rows, "replay produced no evidence at all"
    for row in rows:
        assert 0.0 <= row.strength <= 1.0
        assert row.weight >= 0.0
        assert row.details is not None
        # Anything about another transaction must name it.
        if row.kind in {
            EvidenceKind.STRUCTURAL_SIMILARITY.value,
            EvidenceKind.FLAGGED_NEIGHBOUR.value,
            EvidenceKind.CONFIRMED_NEIGHBOUR.value,
        }:
            assert row.neighbour_tx_id is not None


def test_every_case_carries_the_graph_model_second_opinion(db_session, replayed):
    missing = db_session.execute(
        text("""
        SELECT count(*) FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts AND c.graph_score IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM evidence_items e
            WHERE e.case_report_id = c.id AND e.kind = :kind)
        """),
        {"ts": REPLAY_TIMESTEP, "kind": EvidenceKind.GRAPH_MODEL_CORROBORATION.value},
    ).scalar_one()
    assert missing == 0


def test_graph_score_never_gates_the_queue(db_session, replayed):
    """Selection is XGBoost's alone: the queue is not ordered by graph score."""
    rows = db_session.execute(
        text("""
        SELECT c.risk_score, c.graph_score, c.queue_rank
        FROM case_reports c JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank
        """),
        {"ts": REPLAY_TIMESTEP},
    ).all()

    scores = [r.risk_score for r in rows]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# Similarity and leakage
# --------------------------------------------------------------------------


def test_similarity_returns_results_for_queued_cases(db_session, replayed):
    """The regression test for the HNSW filtering trap.

    The reference pool is ~13% of the table. With default settings pgvector's
    HNSW index post-filters and returns zero rows -- reproduced in Phase 2.
    If `hnsw.iterative_scan` is ever dropped, this goes back to empty.
    """
    tx_ids = (
        db_session.execute(
            text("""
        SELECT c.tx_id FROM case_reports c JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank LIMIT 10
        """),
            {"ts": REPLAY_TIMESTEP},
        )
        .scalars()
        .all()
    )

    empty = [tx for tx in tx_ids if not similarity.find_similar(db_session, int(tx))]
    assert not empty, f"filtered HNSW query returned nothing for {empty}"


def test_approximate_similarity_agrees_with_exact_search(db_session, replayed):
    tx_ids = (
        db_session.execute(
            text("""
        SELECT c.tx_id FROM case_reports c JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank LIMIT 10
        """),
            {"ts": REPLAY_TIMESTEP},
        )
        .scalars()
        .all()
    )

    report = similarity.verify_against_exact(db_session, [int(t) for t in tx_ids])

    assert report["empty_approximate_results"] == 0
    # HNSW is approximate; a systematic shortfall is the failure that matters.
    assert report["mean_recall"] >= 0.9, report["samples"]


def test_similarity_excludes_the_queried_transaction(db_session, embedded):
    tx_id = db_session.execute(
        text("SELECT tx_id FROM transactions WHERE timestep <= 29 AND label = 'illicit' LIMIT 1")
    ).scalar_one()

    matches = similarity.find_similar(db_session, int(tx_id), k=5)

    assert matches, "expected a self-excluded query to still return neighbours"
    assert all(m.tx_id != tx_id for m in matches)


def test_reference_pool_holds_only_labelled_training_transactions(db_session, embedded):
    """The leakage guard on similarity.

    A validation or test transaction in the pool would mean citing an answer
    the system is not supposed to have yet.
    """
    tx_ids = (
        db_session.execute(text("SELECT tx_id FROM case_reports ORDER BY random() LIMIT 15"))
        .scalars()
        .all()
    )

    for tx_id in tx_ids:
        for match in similarity.find_similar(db_session, int(tx_id), k=5):
            assert match.timestep in TRAIN_TIMESTEPS
            assert match.timestep not in VAL_TIMESTEPS
            assert match.timestep not in TEST_TIMESTEPS
            assert match.label != Label.UNKNOWN.value


def test_no_persisted_evidence_cites_a_non_training_transaction(db_session, replayed):
    """The end-to-end leakage assertion, over what actually got written.

    Every evidence item naming another transaction must name one from the
    training range. If a test-range label ever reached a case report, this is
    the test that fails.
    """
    offenders = db_session.execute(
        text("""
        SELECT e.id, e.kind, e.neighbour_tx_id, nt.timestep, nt.label
        FROM evidence_items e
        JOIN transactions nt ON nt.tx_id = e.neighbour_tx_id
        WHERE e.kind = :kind AND nt.timestep > :train_max
        """),
        {
            "kind": EvidenceKind.STRUCTURAL_SIMILARITY.value,
            "train_max": max(TRAIN_TIMESTEPS),
        },
    ).all()

    assert not offenders, f"similarity evidence cites non-training rows: {offenders}"


def test_flagged_neighbour_evidence_never_rests_on_a_same_batch_label(db_session, replayed):
    """Neighbour evidence may cite a flag or a review, never raw ground truth
    from an unreviewed transaction in the batch being scored."""
    rows = db_session.execute(
        text("""
        SELECT e.details->>'source' AS source, nt.timestep, c.tx_id
        FROM evidence_items e
        JOIN case_reports c ON c.id = e.case_report_id
        JOIN transactions t ON t.tx_id = c.tx_id
        JOIN transactions nt ON nt.tx_id = e.neighbour_tx_id
        WHERE e.kind IN (:flagged, :confirmed)
        """),
        {
            "flagged": EvidenceKind.FLAGGED_NEIGHBOUR.value,
            "confirmed": EvidenceKind.CONFIRMED_NEIGHBOUR.value,
        },
    ).all()

    for row in rows:
        assert row.source in {"model_flagged", "analyst_review", "historical_label"}
        if row.source == "historical_label":
            # Only legitimate from a batch that has already been processed.
            assert row.timestep < REPLAY_TIMESTEP


def test_default_hnsw_settings_would_return_nothing(db_session, embedded):
    """Proves the trap is real, so the fix is not cargo cult.

    Runs the same filtered similarity query with `hnsw.iterative_scan` turned
    off, inside a savepoint so the setting cannot escape. Under the default
    the index post-filters and yields far fewer rows than asked for -- often
    none. This is the failure Phase 2 hit, pinned down as an assertion.
    """
    tx_id = db_session.execute(
        text("SELECT c.tx_id FROM case_reports c ORDER BY c.risk_score DESC LIMIT 1")
    ).scalar_one()

    with_fix = len(similarity.find_similar(db_session, int(tx_id), k=5))
    assert with_fix == 5, "the configured query should fill its k"

    with db_session.begin_nested():
        db_session.execute(text("SET LOCAL hnsw.iterative_scan = off"))
        naive = db_session.execute(
            text("""
            SELECT t.tx_id
            FROM transaction_embeddings e
            JOIN transactions t ON t.tx_id = e.tx_id
            WHERE t.timestep <= 29 AND t.label <> 'unknown' AND t.tx_id <> :tx
            ORDER BY e.embedding <=> (
                SELECT embedding FROM transaction_embeddings WHERE tx_id = :tx)
            LIMIT 5
            """),
            {"tx": int(tx_id)},
        ).all()

    assert len(naive) < with_fix, (
        "expected the un-tuned HNSW scan to under-return; if this ever passes "
        "trivially, pgvector's filtering behaviour has changed and the "
        "connection settings should be re-examined"
    )


def test_a_failed_replay_is_visible_rather_than_silent(db_session, ingested, tmp_path):
    """A replay that dies must leave a `failed` batch run behind.

    The batch run is committed as soon as it starts, so a later failure
    updates it instead of vanishing with the rolled-back transaction. Without
    that, polling the batch reports "never replayed" for a job that ran and
    broke -- which is the least useful answer possible.
    """
    timestep = 48
    db_session.execute(text("DELETE FROM batch_runs WHERE timestep = :ts"), {"ts": timestep})
    db_session.commit()

    with pytest.raises(FileNotFoundError):
        # An empty models root: the manifest cannot be loaded.
        replay_service.replay_batch(db_session, timestep, models_root=tmp_path)

    run = queue_service.get_batch_run(db_session, timestep)
    assert run is not None, "a failed replay left no trace"
    assert run["status"] == BatchStatus.FAILED.value
    assert run["error"]

    db_session.execute(text("DELETE FROM batch_runs WHERE timestep = :ts"), {"ts": timestep})
    db_session.commit()
