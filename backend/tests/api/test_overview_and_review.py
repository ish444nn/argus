"""Overview aggregates, the neighbourhood graph and analyst decisions.

These back the Phase 5 screens, so the assertions are mostly "does this number
actually come from the database" -- the one failure mode that matters for an
operations view is a figure that looks plausible and is not real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from argus.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api():
    from argus.db.session import SessionLocal

    try:
        with SessionLocal() as session:
            cases = session.execute(text("SELECT count(*) FROM case_reports")).scalar_one()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database not reachable: {exc}")
    if not cases:
        pytest.skip("no cases; run a replay first")

    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def session():
    from argus.db.session import SessionLocal

    with SessionLocal() as s:
        yield s


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------


def test_overview_counts_match_the_database(api, session):
    body = api.get("/api/overview").json()

    cases = session.execute(text("SELECT count(*) FROM case_reports")).scalar_one()
    evidence = session.execute(text("SELECT count(*) FROM evidence_items")).scalar_one()

    assert body["cases"]["total"] == cases
    assert sum(body["evidence"].values()) == evidence


def test_overview_case_states_partition_the_total(api):
    cases = api.get("/api/overview").json()["cases"]
    lifecycle = cases["queued"] + cases["investigating"] + cases["ready"] + cases["failed"]
    assert lifecycle == cases["total"]


def test_risk_distribution_covers_every_scored_transaction(api, session):
    """Deliberately over `risk_scores`, not `case_reports`.

    The queue is by construction the top slice, so a distribution of queued
    cases puts everything in the highest band and shows nothing.
    """
    body = api.get("/api/overview").json()
    scored = session.execute(text("SELECT count(*) FROM risk_scores")).scalar_one()

    assert sum(band["count"] for band in body["risk_distribution"]) == scored
    assert sum(band["alerted"] for band in body["risk_distribution"]) == body["cases"]["total"]


def test_alerted_never_exceeds_the_band_it_sits_in(api):
    for band in api.get("/api/overview").json()["risk_distribution"]:
        assert band["alerted"] <= band["count"]


def test_realised_alert_rate_matches_the_counts(api):
    batches = api.get("/api/overview").json()["batches"]
    if not batches["scored"]:
        pytest.skip("nothing scored")
    assert batches["realised_alert_rate"] == pytest.approx(
        batches["queued"] / batches["scored"], abs=1e-5
    )


def test_awaiting_review_matches_cases_without_a_decision(api, session):
    body = api.get("/api/overview").json()
    expected = session.execute(
        text("""
        SELECT count(*) FROM case_reports c
        WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.case_report_id = c.id)
        """)
    ).scalar_one()
    assert body["cases"]["awaiting_review"] == expected


def test_overview_reports_the_configured_provider(api):
    body = api.get("/api/overview").json()
    assert body["llm_provider"] in {"gemini", "stub"}
    assert body["alert_budget"] == pytest.approx(0.01)


# --------------------------------------------------------------------------
# Neighbourhood
# --------------------------------------------------------------------------


def test_neighbourhood_matches_the_edge_table(api, session):
    tx_id = session.execute(
        text("SELECT tx_id FROM transactions WHERE in_degree > 2 AND out_degree > 1 LIMIT 1")
    ).scalar_one()

    body = api.get(f"/api/transactions/{tx_id}/neighbourhood").json()
    degree = session.execute(
        text("SELECT in_degree + out_degree FROM transactions WHERE tx_id = :tx"),
        {"tx": tx_id},
    ).scalar_one()

    assert body["total_degree"] == degree
    assert body["neighbours"]
    for neighbour in body["neighbours"]:
        assert neighbour["direction"] in {"in", "out"}
        assert neighbour["tx_id"] != tx_id


def test_neighbourhood_is_capped_and_says_so(api, session):
    """A partial picture must be labelled as one."""
    tx_id = session.execute(
        text("SELECT tx_id FROM transactions ORDER BY in_degree + out_degree DESC LIMIT 1")
    ).scalar_one()

    body = api.get(f"/api/transactions/{tx_id}/neighbourhood").json()
    if body["total_degree"] > len(body["neighbours"]):
        assert body["truncated"] is True


def test_all_neighbours_share_the_batch(api, session):
    """Elliptic has no cross-batch edges; the UI states this, so assert it."""
    row = session.execute(
        text("""
        SELECT c.tx_id, t.timestep FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.in_degree + t.out_degree > 1 LIMIT 1
        """)
    ).one_or_none()
    if row is None:
        pytest.skip("no multi-edge case")

    body = api.get(f"/api/transactions/{row.tx_id}/neighbourhood").json()
    assert all(n["timestep"] == row.timestep for n in body["neighbours"])


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------


def test_recording_a_decision_round_trips(api, session):
    case_id = session.execute(
        text("SELECT id FROM case_reports ORDER BY id DESC LIMIT 1")
    ).scalar_one()

    created = api.post(
        f"/api/cases/{case_id}/reviews",
        json={"decision": "dismissed", "note": "test decision"},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["decision"] == "dismissed"
    assert body["note"] == "test decision"
    assert body["analyst"]

    listed = api.get(f"/api/cases/{case_id}/reviews").json()
    assert listed[0]["review_id"] == body["review_id"]

    _cleanup(session, case_id)


def test_decisions_append_rather_than_replace(api, session):
    """A decision log that overwrites is not a log."""
    case_id = session.execute(
        text("SELECT id FROM case_reports ORDER BY id DESC LIMIT 1")
    ).scalar_one()

    api.post(f"/api/cases/{case_id}/reviews", json={"decision": "dismissed"})
    api.post(f"/api/cases/{case_id}/reviews", json={"decision": "confirmed"})

    history = api.get(f"/api/cases/{case_id}/reviews").json()
    assert len(history) >= 2
    # Newest first, so the queue reads the most recent.
    assert history[0]["decision"] == "confirmed"

    _cleanup(session, case_id)


def test_the_queue_shows_the_latest_decision(api, session):
    row = session.execute(
        text("""
        SELECT c.id, t.timestep FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        ORDER BY c.id DESC LIMIT 1
        """)
    ).one()
    case_id = row.id
    api.post(f"/api/cases/{case_id}/reviews", json={"decision": "needs_more_evidence"})

    # Filtered to its own batch: the queue spans more cases than one page.
    page = api.get("/api/queue", params={"timestep": row.timestep, "limit": 200}).json()
    entry = next(item for item in page["items"] if item["case_id"] == case_id)
    assert entry["latest_decision"] == "needs_more_evidence"

    _cleanup(session, case_id)


def test_undecided_filter_excludes_decided_cases(api, session):
    row = session.execute(
        text("""
        SELECT c.id, t.timestep FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        ORDER BY c.id DESC LIMIT 1
        """)
    ).one()
    case_id = row.id
    api.post(f"/api/cases/{case_id}/reviews", json={"decision": "dismissed"})

    page = api.get(
        "/api/queue",
        params={"undecided_only": True, "timestep": row.timestep, "limit": 200},
    ).json()
    assert all(item["case_id"] != case_id for item in page["items"])

    _cleanup(session, case_id)


def test_an_unknown_decision_is_rejected(api, session):
    case_id = session.execute(text("SELECT id FROM case_reports LIMIT 1")).scalar_one()
    response = api.post(f"/api/cases/{case_id}/reviews", json={"decision": "freeze"})
    assert response.status_code == 422


def test_a_review_on_a_missing_case_is_404(api):
    assert (
        api.post("/api/cases/99999999/reviews", json={"decision": "dismissed"}).status_code == 404
    )
    assert api.get("/api/cases/99999999/reviews").status_code == 404


def _cleanup(session, case_id: int) -> None:
    session.execute(text("DELETE FROM reviews WHERE case_report_id = :c"), {"c": case_id})
    session.commit()
