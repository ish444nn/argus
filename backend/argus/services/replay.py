"""Batch replay: score a time step, build its queue, gather evidence.

One batch is one Elliptic time step. Replaying it:

1. loads the primary scorer from its manifest;
2. scores every transaction in the batch;
3. ranks them and takes exactly the alert budget;
4. writes scores, the batch run, and one case per alert;
5. runs the deterministic evidence tools over each case.

Idempotency
-----------
Re-running a batch must not duplicate anything or destroy analyst work. The
schema does most of it: `batch_runs.timestep`, `risk_scores(tx_id,
model_version)` and `case_reports.tx_id` are all unique, so every write is an
upsert. Evidence is derived, so it is replaced wholesale per case.

The one judgement call is a case that was in the queue last time and is not
this time -- possible if the model changes. Those are deleted, unless an
analyst has already reviewed them, in which case the case stays and is logged.
Silently discarding reviewed work would be worse than a slightly stale queue.

The narrative layer sits on top of this. Everything here is
deterministic: the same batch and the same model produce the same rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from argus.agent.evidence import EvidenceDraft, persist
from argus.agent.tools import graph_tools, similarity
from argus.core.config import get_settings
from argus.db.enums import BatchStatus, CaseStatus, EvidenceKind
from argus.db.models import BatchRun, CaseReport, EvidenceItem, Review, RiskScore

log = logging.getLogger(__name__)

# What replay itself produces. Everything else on a case -- typology
# citations -- is written by the investigation and is not replay's to delete.
DETERMINISTIC_KINDS = (
    EvidenceKind.HEURISTIC,
    EvidenceKind.FLAGGED_NEIGHBOUR,
    EvidenceKind.CONFIRMED_NEIGHBOUR,
    EvidenceKind.STRUCTURAL_SIMILARITY,
    EvidenceKind.GRAPH_MODEL_CORROBORATION,
)


@dataclass
class ReplayResult:
    batch_run_id: int
    timestep: int
    status: str
    model_version: str
    alert_budget: float
    scored_count: int
    queued_count: int
    evidence_count: int
    dropped_cases: int = 0
    retained_reviewed_cases: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _upsert_batch_run(
    session: Session, timestep: int, model_version: str | None, budget: float
) -> BatchRun:
    """Find or create this time step's run and mark it started."""
    run = session.scalar(select(BatchRun).where(BatchRun.timestep == timestep))
    if run is None:
        run = BatchRun(timestep=timestep)
        session.add(run)
    run.status = BatchStatus.RUNNING
    run.model_version = model_version
    run.alert_budget = budget
    run.started_at = datetime.now(UTC)
    run.finished_at = None
    run.error = None
    # Committed immediately, not merely flushed. A failure later in the replay
    # rolls back its own transaction; if this row were still uncommitted it
    # would vanish with it, and a caller polling the batch would be told the
    # time step had never been replayed rather than that it failed.
    session.commit()
    return run


def _write_scores(
    session: Session,
    tx_ids,
    scores,
    model_version: str,
    batch_run_id: int,
) -> int:
    """Upsert one risk score per transaction.

    `ON CONFLICT` on (tx_id, model_version) makes a re-run overwrite rather
    than fail, which is most of what idempotency means here.
    """
    rows = [
        {
            "tx_id": int(tx_id),
            "model_version": model_version,
            "score": float(score),
            "batch_run_id": batch_run_id,
        }
        for tx_id, score in zip(tx_ids.tolist(), scores.tolist(), strict=True)
    ]
    statement = insert(RiskScore).values(rows)
    session.execute(
        statement.on_conflict_do_update(
            index_elements=["tx_id", "model_version"],
            set_={
                "score": statement.excluded.score,
                "batch_run_id": statement.excluded.batch_run_id,
            },
        )
    )
    return len(rows)


def _graph_scores(session: Session, tx_ids: list[int]) -> dict[int, float]:
    """GraphSAGE's second opinion, where embeddings have been computed."""
    if not tx_ids:
        return {}
    rows = session.execute(
        text(
            "SELECT tx_id, graph_score FROM transaction_embeddings "
            "WHERE tx_id = ANY(:ids) AND graph_score IS NOT NULL"
        ),
        {"ids": tx_ids},
    ).all()
    return {int(row.tx_id): float(row.graph_score) for row in rows}


def _sync_queue(
    session: Session,
    timestep: int,
    selected: list[tuple[int, float, int]],
    model_version: str,
    batch_run_id: int,
) -> tuple[list[CaseReport], int, int]:
    """Make the stored queue match this run's selection.

    Returns the cases for the current selection, plus counts of cases dropped
    and of reviewed cases retained despite falling out of the queue.
    """
    selected_ids = {tx_id for tx_id, _, _ in selected}

    stale = session.execute(
        text("""
        SELECT c.id, c.tx_id,
               EXISTS (SELECT 1 FROM reviews r WHERE r.case_report_id = c.id) AS reviewed,
               c.narrative IS NOT NULL AS investigated
        FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        WHERE t.timestep = :timestep
        """),
        {"timestep": timestep},
    ).all()

    dropped = 0
    retained = 0
    for row in stale:
        if int(row.tx_id) in selected_ids:
            continue
        if row.reviewed or row.investigated:
            # Work has been done on this case -- an analyst's decision, or a
            # written and cited investigation. Lowering the alert budget is a
            # capacity decision about what to look at next; it is not a reason
            # to delete what has already been looked at. (Only reviews were
            # kept before, so re-applying a smaller budget silently destroyed
            # every investigation nobody had decided on yet.)
            retained += 1
            log.warning(
                "case %s (tx %s) left the queue but has %s; retaining",
                row.id,
                row.tx_id,
                "reviews" if row.reviewed else "a written investigation",
            )
            continue
        session.execute(delete(CaseReport).where(CaseReport.id == row.id))
        dropped += 1

    graph = _graph_scores(session, [tx_id for tx_id, _, _ in selected])

    cases: list[CaseReport] = []
    for tx_id, score, rank in selected:
        case = session.scalar(select(CaseReport).where(CaseReport.tx_id == tx_id))
        is_new = case is None
        if is_new:
            case = CaseReport(tx_id=tx_id)
            session.add(case)
        case.batch_run_id = batch_run_id
        case.risk_score = score
        case.model_version = model_version
        case.queue_rank = rank
        case.graph_score = graph.get(tx_id)
        if is_new:
            # Only a new case starts at `queued`. Re-replaying a batch
            # regenerates identical deterministic evidence, so an existing
            # report is still valid -- resetting its status would discard a
            # written investigation for no reason.
            case.status = CaseStatus.QUEUED
        cases.append(case)

    session.flush()
    return cases, dropped, retained


def gather_evidence(session: Session, case: CaseReport, timestep: int) -> int:
    """Run every deterministic tool over one case and persist the result.

    Deliberately not an LLM step. The investigation adds typology retrieval
    and a narrative on top of exactly these drafts.
    """
    drafts: list[EvidenceDraft] = []
    drafts.extend(graph_tools.structural_heuristics(session, case.tx_id))
    drafts.extend(graph_tools.flagged_neighbours(session, case.tx_id, timestep))
    drafts.extend(similarity.similarity_evidence(session, case.tx_id))

    if case.graph_score is not None:
        drafts.append(
            EvidenceDraft(
                kind=EvidenceKind.GRAPH_MODEL_CORROBORATION,
                summary=(
                    f"The neighbourhood-aware model independently scores this "
                    f"transaction {case.graph_score:.3f}."
                ),
                strength=float(case.graph_score),
                details={
                    "graph_score": round(float(case.graph_score), 6),
                    "primary_score": round(float(case.risk_score), 6),
                    "model": "graphsage",
                    "role": "second opinion; does not gate the queue",
                },
            )
        )

    # Scoped to the kinds this function produces. Typology citations belong
    # to the investigation and must survive a re-replay.
    written = persist(session, case.id, drafts, kinds=DETERMINISTIC_KINDS)

    # Evidence confidence is a pure function of the evidence just gathered, so
    # it is known now. Computing it here means the case page can show it
    # immediately -- an analyst should not have to run a language model to
    # find out how much evidence the system already holds.
    session.flush()
    score_confidence(session, case.id)
    return written


def score_confidence(session: Session, case_id: int) -> float:
    """Recompute and store a case's evidence confidence.

    Reads the persisted deterministic evidence and calls the one confidence
    implementation. The investigation calls this too, so there is a single
    definition of the number rather than one per caller.
    """
    from argus.agent import confidence as confidence_module
    from argus.agent.state import EvidenceRecord

    rows = session.execute(
        text("""
        SELECT id, kind, summary, strength, weight
        FROM evidence_items WHERE case_report_id = :case_id
        """),
        {"case_id": case_id},
    ).all()
    records = [
        EvidenceRecord(
            id=int(row.id),
            kind=row.kind,
            summary=row.summary,
            strength=float(row.strength),
            weight=float(row.weight),
        )
        for row in rows
    ]
    result = confidence_module.compute(records)
    session.execute(
        text(
            "UPDATE case_reports SET confidence = :value, "
            "confidence_version = :version WHERE id = :case_id"
        ),
        {"value": result.value, "version": result.version, "case_id": case_id},
    )
    return result.value


def replay_batch(
    session: Session,
    timestep: int,
    *,
    alert_budget: float | None = None,
    models_root: Path | None = None,
    with_evidence: bool = True,
) -> ReplayResult:
    """Score one time step and rebuild its queue. Safe to re-run."""
    # Imported here, not at module scope: this pulls in xgboost, and the API
    # dispatches replay without ever executing it.
    from argus.ml import scoring

    budget = alert_budget if alert_budget is not None else get_settings().alert_budget
    # The run row is created and committed before anything that can fail,
    # including loading the model. A missing manifest is exactly the kind of
    # failure a caller needs to see reported against the batch, and if the
    # model were loaded first that failure would leave no record at all.
    run = _upsert_batch_run(session, timestep, model_version=None, budget=budget)

    try:
        model = scoring.load_primary(root=models_root)
        run.model_version = model.version
        tx_ids, raw = scoring.load_batch_features(session, timestep)
        scores = scoring.score_batch(model, raw)
        order = scoring.select_alerts(scores, budget)

        scored = _write_scores(session, tx_ids, scores, model.version, run.id)
        selected = [
            (int(tx_ids[idx]), float(scores[idx]), rank)
            for rank, idx in enumerate(order.tolist(), start=1)
        ]
        cases, dropped, retained = _sync_queue(session, timestep, selected, model.version, run.id)

        evidence_count = 0
        if with_evidence:
            for case in cases:
                evidence_count += gather_evidence(session, case, timestep)

        run.status = BatchStatus.COMPLETED
        run.scored_count = scored
        run.queued_count = len(cases)
        run.investigated_count = len(cases) if with_evidence else 0
        run.failed_count = 0
        run.finished_at = datetime.now(UTC)
        session.commit()

        log.info(
            "replayed timestep %s: scored %d, queued %d, evidence %d",
            timestep,
            scored,
            len(cases),
            evidence_count,
        )
        warnings = []
        if retained:
            warnings.append(f"{retained} reviewed case(s) no longer in the queue were retained")
        return ReplayResult(
            batch_run_id=run.id,
            timestep=timestep,
            status=run.status,
            model_version=model.version,
            alert_budget=budget,
            scored_count=scored,
            queued_count=len(cases),
            evidence_count=evidence_count,
            dropped_cases=dropped,
            retained_reviewed_cases=retained,
            warnings=warnings,
        )
    except Exception as exc:
        session.rollback()
        # Record the failure on the run itself, so a caller polling the batch
        # sees why rather than a job that simply stopped.
        run = session.scalar(select(BatchRun).where(BatchRun.timestep == timestep))
        if run is not None:
            run.status = BatchStatus.FAILED
            run.error = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
        log.exception("replay of timestep %s failed", timestep)
        raise


def evidence_counts(session: Session, case_id: int) -> dict[str, int]:
    """Evidence tally by kind, for tests and status reporting."""
    rows = session.execute(
        select(EvidenceItem.kind, text("count(*)"))
        .where(EvidenceItem.case_report_id == case_id)
        .group_by(EvidenceItem.kind)
    ).all()
    return {row[0]: int(row[1]) for row in rows}


def has_reviews(session: Session, case_id: int) -> bool:
    return session.scalar(select(Review.id).where(Review.case_report_id == case_id)) is not None
