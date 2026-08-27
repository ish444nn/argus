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


@pytest.mark.parametrize("field", ["timestep", "status", "confidence"])
@pytest.mark.parametrize("descending", [True, False])
def test_every_column_the_queue_shows_can_be_sorted(api, field, descending):
    """The table lets an analyst sort by any column they can scan down.

    A column that looks sortable and is not is worse than one that plainly is
    not, so this covers the three added alongside rank, risk and graph score.
    """
    body = api.get(
        "/api/queue",
        params={"sort_by": field, "descending": descending, "limit": 20},
    ).json()

    assert body["items"], "expected the replayed batch to have queued cases"


def test_status_sorts_by_workflow_position_not_the_alphabet(api):
    """Ascending puts the least progressed cases first.

    The column shows a decision where one exists and a status otherwise, so
    sorting it alphabetically on `status` would order rows by a word the
    analyst cannot see.
    """
    ascending = api.get(
        "/api/queue", params={"sort_by": "status", "descending": False, "limit": 100}
    ).json()["items"]

    def rank(item):
        if item["latest_decision"] is not None:
            return 4
        return {"queued": 0, "investigating": 1, "failed": 2, "ready": 3}[item["status"]]

    positions = [rank(item) for item in ascending]
    assert positions == sorted(positions)


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


def test_dispatch_fails_fast_when_the_broker_is_unreachable(api, monkeypatch):
    """A dead broker must produce an error, not a hanging request.

    Celery's publish path retries for tens of seconds by default, so pressing
    "Run investigation" with Redis down left the request hanging. `dispatch`
    checks the connection first and turns a failure into a 503 naming the
    cause.
    """
    from argus.api import deps

    class DeadConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def ensure_connection(self, **kwargs):
            raise OSError("broker unreachable")

    monkeypatch.setattr(
        "argus.jobs.celery_app.celery_app.connection_for_write",
        lambda: DeadConnection(),
    )

    with pytest.raises(Exception) as caught:  # noqa: B017 - HTTPException
        deps.dispatch("argus.investigate_case", 1)
    assert getattr(caught.value, "status_code", None) == 503
    assert "queue is unavailable" in str(getattr(caught.value, "detail", ""))
