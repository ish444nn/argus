"""The alert budget selects the queue, and changing it changes the queue.

These are the regression tests for a bug where the budget control moved a
label and nothing else: the slider re-cut a *displayed* distribution while the
stored queue stayed at whatever the last replay used, so the overview and the
queue screen reported different sizes for the same thing.

The rule under test is the project's own: the budget is an exact top-k **by
rank within each batch**, never a probability cutoff, so the selection size is
`ceil(batch_size * budget)` summed over the replayed batches. Everything here
asserts against that computed expectation rather than against a number typed
into the test.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import text

from argus.services import overview as overview_service
from argus.services import queue as queue_service
from argus.services import replay as replay_service

pytestmark = pytest.mark.integration

# Two budgets either side of the canonical 1%, so the tests cover the queue
# growing and shrinking rather than only the easy direction.
BUDGETS = [0.01, 0.02, 0.03]


@pytest.fixture(scope="module")
def replayed_timesteps(db_session):
    timesteps = list(
        db_session.execute(text("SELECT timestep FROM batch_runs ORDER BY timestep"))
        .scalars()
        .all()
    )
    if not timesteps:
        pytest.skip("no replayed batches; run a replay first")
    return [int(ts) for ts in timesteps]


@pytest.fixture(scope="module")
def restore_budget(db_session, replayed_timesteps):
    """Put the queue back the way it was found, whatever these tests do to it."""
    original = db_session.execute(
        text(
            "SELECT CASE WHEN count(DISTINCT alert_budget) = 1 THEN min(alert_budget) END "
            "FROM batch_runs"
        )
    ).scalar_one_or_none()
    yield original
    if original is not None:
        for timestep in replayed_timesteps:
            replay_service.replay_batch(db_session, timestep, alert_budget=original)


@pytest.mark.parametrize("budget", BUDGETS)
def test_applying_a_budget_changes_the_actual_selection(
    db_session, replayed_timesteps, restore_budget, budget
):
    """The queue after a replay is exactly the top-k the budget asks for.

    Not a recount of a stored queue: `replay_batch` is the only thing that
    decides membership, so this drives it and then counts rows.
    """
    expected = 0
    for timestep in replayed_timesteps:
        size = db_session.execute(
            text("SELECT count(*) FROM transactions WHERE timestep = :ts"), {"ts": timestep}
        ).scalar_one()
        result = replay_service.replay_batch(db_session, timestep, alert_budget=budget)
        assert result.queued_count == math.ceil(size * budget), (
            f"batch {timestep} at {budget}: selection is not the top-k by rank"
        )
        expected += result.queued_count

    _, total = queue_service.list_queue(db_session, limit=1)
    assert total == expected, "the queue does not contain the selection that was just made"


def test_the_overview_and_the_queue_report_the_same_number(
    db_session, replayed_timesteps, restore_budget
):
    """The two screens must never disagree about how big the queue is.

    They used to: the overview summed `batch_runs.queued_count` while the
    queue listed `case_reports`, and those differ by exactly the cases a
    re-replay retained because someone had worked on them.
    """
    for timestep in replayed_timesteps:
        replay_service.replay_batch(db_session, timestep, alert_budget=0.02)

    ops = overview_service.operations(db_session)
    _, queue_total = queue_service.list_queue(db_session, limit=1)

    assert ops["batches"]["queued"] == queue_total


def test_the_applied_budget_is_read_from_the_batches_not_assumed(
    db_session, replayed_timesteps, restore_budget
):
    """`applied_alert_budget` is the budget the stored queue was built at.

    This is what lets the UI say "applied" or "preview" from data instead of
    comparing the chosen number against a hardcoded canonical one.
    """
    for timestep in replayed_timesteps:
        replay_service.replay_batch(db_session, timestep, alert_budget=0.03)

    assert overview_service.operations(db_session)["applied_alert_budget"] == pytest.approx(0.03)


def test_batches_at_different_budgets_report_no_single_applied_budget(
    db_session, replayed_timesteps, restore_budget
):
    """A part-applied change is a real state and is reported as one.

    Averaging the budgets, or picking the first, would let the UI claim the
    queue is something it is not while half of it is still the old selection.
    """
    if len(replayed_timesteps) < 2:
        pytest.skip("needs at least two replayed batches")

    replay_service.replay_batch(db_session, replayed_timesteps[0], alert_budget=0.01)
    replay_service.replay_batch(db_session, replayed_timesteps[1], alert_budget=0.02)

    assert overview_service.operations(db_session)["applied_alert_budget"] is None


def test_lowering_the_budget_keeps_cases_that_carry_an_investigation(
    db_session, replayed_timesteps, restore_budget
):
    """Shrinking the queue must not delete written work.

    Only reviewed cases were kept before, so re-applying a smaller budget
    destroyed every investigation nobody had decided on yet -- including, in
    practice, the whole demo.
    """
    timestep = replayed_timesteps[0]
    replay_service.replay_batch(db_session, timestep, alert_budget=0.03)

    # Take a case near the bottom of the 3% cut, so a 1% cut will drop it, and
    # give it a narrative as though it had been investigated.
    victim = db_session.execute(
        text("""
        SELECT c.id, c.tx_id FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :ts ORDER BY c.queue_rank DESC LIMIT 1
        """),
        {"ts": timestep},
    ).one()
    db_session.execute(
        text(
            "UPDATE case_reports SET narrative = :n, narrative_source = 'template', "
            "status = 'ready' WHERE id = :id"
        ),
        {"n": "written during a regression test", "id": victim.id},
    )
    db_session.commit()

    replay_service.replay_batch(db_session, timestep, alert_budget=0.01)

    survived = db_session.execute(
        text("SELECT narrative FROM case_reports WHERE id = :id"), {"id": victim.id}
    ).scalar_one_or_none()
    assert survived == "written during a regression test", (
        "a case with a written investigation was deleted by lowering the budget"
    )

    # Leave nothing behind: this case is outside the canonical queue now.
    db_session.execute(
        text(
            "UPDATE case_reports SET narrative = NULL, narrative_source = NULL, "
            "status = 'queued' WHERE id = :id"
        ),
        {"id": victim.id},
    )
    db_session.commit()
