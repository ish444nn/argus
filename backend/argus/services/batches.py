"""Batch lifecycle operations the API performs itself.

Deliberately separate from `argus.services.replay`. Replay imports the scorer,
which imports numpy — fine in the worker, fatal in the API image, which carries
neither numpy nor xgboost. Removing a batch is pure SQL and has to run in the
API process, so it lives in a module the API can import without dragging the
machine-learning stack behind it.

Nothing here imports `argus.ml`, directly or transitively. Keep it that way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import delete, text
from sqlalchemy.orm import Session

from argus.db.models import BatchRun, CaseReport

log = logging.getLogger(__name__)


@dataclass
class RemovalResult:
    timestep: int
    cases_removed: int
    reviewed_retained: int
    scores_removed: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def remove_batch(session: Session, timestep: int) -> RemovalResult:
    """Undo one batch's replay.

    This is a fixed research dataset, not an ingestion pipeline, so a replayed
    time step that is no longer interesting should be clearable without
    reaching for psql. "Remove" means exactly *undo the replay*: the run row,
    the scores it wrote and the cases it queued, all scoped to this time step.

    What it does not touch: transactions, edges and embeddings (they are the
    dataset, not the run), the typology corpus, and every other time step. The
    step simply becomes available to replay again.

    Reviewed cases survive. That is the same rule `replay._sync_queue` applies
    when a case falls out of the queue — an analyst's recorded decision
    outlives the machinery that surfaced it — and having one policy instead of
    two is worth more than the tidiness of a clean sweep. A retained case keeps
    its evidence and its report; only its `batch_run_id` goes null, which the
    foreign key does by itself.
    """
    cases = session.execute(
        text("""
        SELECT c.id,
               EXISTS (SELECT 1 FROM reviews r WHERE r.case_report_id = c.id) AS reviewed
        FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :timestep
        """),
        {"timestep": timestep},
    ).all()

    removed = 0
    retained = 0
    for row in cases:
        if row.reviewed:
            retained += 1
            continue
        # Evidence items cascade with the case.
        session.execute(delete(CaseReport).where(CaseReport.id == row.id))
        removed += 1

    scores = session.execute(
        text("""
        DELETE FROM risk_scores
        WHERE tx_id IN (SELECT tx_id FROM transactions WHERE timestep = :timestep)
        """),
        {"timestep": timestep},
    ).rowcount

    session.execute(delete(BatchRun).where(BatchRun.timestep == timestep))
    session.commit()

    log.info(
        "removed batch %s: %s cases deleted, %s reviewed retained, %s scores deleted",
        timestep,
        removed,
        retained,
        scores,
    )
    return RemovalResult(
        timestep=timestep,
        cases_removed=removed,
        reviewed_retained=retained,
        scores_removed=int(scores or 0),
    )
