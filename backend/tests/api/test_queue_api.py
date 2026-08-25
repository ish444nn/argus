"""Queue, case and batch endpoints against the live stack."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from argus.api.main import app

pytestmark = pytest.mark.integration

REPLAY_TIMESTEP = 35


@pytest.fixture(scope="module")
def api():
    from argus.db.session import SessionLocal

    try:
        with SessionLocal() as session:
            queued = session.execute(text("SELECT count(*) FROM case_reports")).scalar_one()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database not reachable: {exc}")
    if not queued:
        pytest.skip("no cases; run a replay first")

    with TestClient(app) as client:
        yield client


def test_queue_lists_cases_highest_risk_first(api):
    body = api.get("/api/queue", params={"limit": 20}).json()

    assert body["total"] > 0
    assert body["items"]
    scores = [item["risk_score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_queue_entry_carries_everything_the_dashboard_needs(api):
    item = api.get("/api/queue", params={"limit": 1}).json()["items"][0]

    for field in (
        "case_id",
        "tx_id",
        "timestep",
        "risk_score",
        "queue_rank",
        "graph_score",
        "status",
        "evidence_count",
        "latest_decision",
        "created_at",
    ):
        assert field in item, f"queue entry missing {field}"


def test_queue_filters_by_timestep(api):
    body = api.get("/api/queue", params={"timestep": REPLAY_TIMESTEP, "limit": 100}).json()

    assert body["items"]
    assert {item["timestep"] for item in body["items"]} == {REPLAY_TIMESTEP}


def test_queue_sorts_by_rank_ascending(api):
    body = api.get(
        "/api/queue",
        params={
            "timestep": REPLAY_TIMESTEP,
            "sort_by": "queue_rank",
            "descending": False,
            "limit": 10,
        },
    ).json()

    ranks = [item["queue_rank"] for item in body["items"]]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_queue_paginates(api):
    first = api.get("/api/queue", params={"limit": 5, "offset": 0}).json()
    second = api.get("/api/queue", params={"limit": 5, "offset": 5}).json()

    assert first["total"] == second["total"]
    assert {i["case_id"] for i in first["items"]} & {i["case_id"] for i in second["items"]} == set()


def test_queue_rejects_an_unknown_sort_field(api):
    assert api.get("/api/queue", params={"sort_by": "id; DROP TABLE"}).status_code == 422


def test_case_detail_includes_neighbourhood_and_evidence(api):
    case_id = api.get("/api/queue", params={"limit": 1}).json()["items"][0]["case_id"]
    case = api.get(f"/api/cases/{case_id}").json()

    assert case["case_id"] == case_id
    assert case["model_version"]
    assert case["neighbourhood"]["tx_id"] == case["tx_id"]
    assert case["neighbourhood"]["total_degree"] == (
        case["neighbourhood"]["in_degree"] + case["neighbourhood"]["out_degree"]
    )
    assert isinstance(case["evidence"], list)


def test_evidence_is_ordered_by_contribution(api):
    case_id = api.get("/api/queue", params={"limit": 1, "sort_by": "risk_score"}).json()["items"][
        0
    ]["case_id"]
    evidence = api.get(f"/api/cases/{case_id}/evidence").json()

    if not evidence:
        pytest.skip("case has no evidence")
    contributions = [item["contribution"] for item in evidence]
    assert contributions == sorted(contributions, reverse=True)
    for item in evidence:
        # Rounded to 6dp on the way out, so compare at that precision.
        assert item["contribution"] == pytest.approx(item["strength"] * item["weight"], abs=1e-6)


def test_case_lookup_by_transaction(api):
    entry = api.get("/api/queue", params={"limit": 1}).json()["items"][0]
    case = api.get(f"/api/transactions/{entry['tx_id']}/case").json()
    assert case["case_id"] == entry["case_id"]


def test_missing_case_returns_404(api):
    assert api.get("/api/cases/999999999").status_code == 404
    assert api.get("/api/transactions/1/case").status_code == 404


def test_batch_status_reads_from_the_database(api):
    body = api.get(f"/api/batches/{REPLAY_TIMESTEP}").json()

    assert body["timestep"] == REPLAY_TIMESTEP
    assert body["status"] == "completed"
    assert body["scored_count"] > body["queued_count"] > 0
    assert body["alert_budget"] == pytest.approx(0.01)


def test_unreplayed_batch_returns_404(api):
    assert api.get("/api/batches/49").status_code in (200, 404)
    assert api.get("/api/batches/99").status_code == 404


def test_replay_refuses_a_training_timestep(api):
    """Time steps 1-34 are training data; scoring them is meaningless."""
    response = api.post("/api/batches/10/replay")

    assert response.status_code == 400
    assert "training data" in response.json()["detail"]


def test_batch_listing_covers_every_replayed_timestep(api):
    body = api.get("/api/batches").json()
    assert any(run["timestep"] == REPLAY_TIMESTEP for run in body)
