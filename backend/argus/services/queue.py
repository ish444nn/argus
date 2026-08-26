"""Read-side queries for the risk queue and case detail.

Pure SQL over what replay wrote. Imports nothing from `argus.ml`, so the API
image needs neither xgboost nor torch to serve any of this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.agent.evidence import OBSERVED_KINDS
from argus.agent.tools.graph_tools import NeighbourhoodProfile, neighbourhood_profile

# Bound once. Every "how much evidence is there" question in the read layer
# asks it of the same set, so the queue's count and the case page's list can
# never disagree. See `agent.evidence.OBSERVED_KINDS` for why the graph
# model's own score is not in it.
_OBSERVED = list(OBSERVED_KINDS)

SortField = Literal["risk_score", "queue_rank", "created_at", "graph_score"]
SORTABLE: dict[str, str] = {
    "risk_score": "c.risk_score",
    "queue_rank": "c.queue_rank",
    "created_at": "c.created_at",
    "graph_score": "c.graph_score",
}


@dataclass(frozen=True)
class QueueEntry:
    case_id: int
    tx_id: int
    timestep: int
    risk_score: float
    queue_rank: int | None
    graph_score: float | None
    status: str
    confidence: float | None
    evidence_count: int
    latest_decision: str | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["created_at"] = self.created_at.isoformat()
        return data


def list_queue(
    session: Session,
    *,
    timestep: int | None = None,
    status: str | None = None,
    decision: str | None = None,
    undecided_only: bool = False,
    sort_by: SortField = "risk_score",
    descending: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[QueueEntry], int]:
    """One page of the queue, plus the total matching count.

    `latest_decision` comes from a lateral join on the newest review, so the
    dashboard can separate handled cases from open ones without a second
    round trip.
    """
    if sort_by not in SORTABLE:
        raise ValueError(f"cannot sort by {sort_by!r}; choose from {sorted(SORTABLE)}")

    filters = ["1 = 1"]
    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "observed_kinds": _OBSERVED,
    }
    if timestep is not None:
        filters.append("t.timestep = :timestep")
        params["timestep"] = timestep
    if status is not None:
        filters.append("c.status = :status")
        params["status"] = status
    if decision is not None:
        filters.append("latest.decision = :decision")
        params["decision"] = decision
    if undecided_only:
        filters.append("latest.decision IS NULL")

    where = " AND ".join(filters)
    direction = "DESC" if descending else "ASC"
    # NULLS LAST so cases without a graph score sort to the bottom either way.
    order = f"{SORTABLE[sort_by]} {direction} NULLS LAST, c.tx_id ASC"

    base = f"""
        FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        LEFT JOIN LATERAL (
            SELECT r.decision FROM reviews r
            WHERE r.case_report_id = c.id
            ORDER BY r.created_at DESC LIMIT 1
        ) latest ON TRUE
        WHERE {where}
    """

    total = session.execute(text(f"SELECT count(*) {base}"), params).scalar_one()

    rows = session.execute(
        text(f"""
        SELECT c.id AS case_id, c.tx_id, t.timestep, c.risk_score, c.queue_rank,
               c.graph_score, c.status, c.confidence, c.created_at,
               latest.decision AS latest_decision,
               (SELECT count(*) FROM evidence_items e
                 WHERE e.case_report_id = c.id
                   AND e.kind = ANY(:observed_kinds)) AS evidence_count
        {base}
        ORDER BY {order}
        LIMIT :limit OFFSET :offset
        """),
        params,
    ).all()

    entries = [
        QueueEntry(
            case_id=int(row.case_id),
            tx_id=int(row.tx_id),
            timestep=int(row.timestep),
            risk_score=float(row.risk_score),
            queue_rank=int(row.queue_rank) if row.queue_rank is not None else None,
            graph_score=float(row.graph_score) if row.graph_score is not None else None,
            status=row.status,
            confidence=float(row.confidence) if row.confidence is not None else None,
            evidence_count=int(row.evidence_count),
            latest_decision=row.latest_decision,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return entries, int(total)


def get_case(session: Session, case_id: int) -> dict[str, Any] | None:
    """A case with its transaction, neighbourhood profile and evidence.

    The neighbourhood profile is computed here rather than stored: it is the
    raw material the heuristics were derived from, not a signal in its own
    right, and the queries behind it are indexed and cheap.
    """
    row = session.execute(
        text("""
        SELECT c.id AS case_id, c.tx_id, c.risk_score, c.model_version, c.queue_rank,
               c.graph_score, c.status, c.confidence, c.confidence_version,
               c.narrative, c.narrative_source, c.typology_assessment,
               c.recommended_action, c.investigation_meta,
               c.error, c.created_at, c.updated_at,
               t.timestep, t.label, t.in_degree, t.out_degree,
               b.id AS batch_run_id, b.alert_budget
        FROM case_reports c
        JOIN transactions t ON t.tx_id = c.tx_id
        LEFT JOIN batch_runs b ON b.id = c.batch_run_id
        WHERE c.id = :case_id
        """),
        {"case_id": case_id},
    ).one_or_none()

    if row is None:
        return None

    return {
        "case_id": int(row.case_id),
        "tx_id": int(row.tx_id),
        "timestep": int(row.timestep),
        # The ground-truth label. Present because this is a research dataset
        # and an evaluator will want it; the investigation pipeline never
        # reads it, and no evidence item is derived from it.
        "label": row.label,
        "risk_score": float(row.risk_score),
        "model_version": row.model_version,
        "queue_rank": int(row.queue_rank) if row.queue_rank is not None else None,
        "graph_score": float(row.graph_score) if row.graph_score is not None else None,
        "status": row.status,
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "confidence_version": row.confidence_version,
        "narrative": row.narrative,
        "narrative_source": row.narrative_source,
        "typology_assessment": row.typology_assessment,
        "recommended_action": row.recommended_action,
        "investigation_meta": row.investigation_meta,
        "error": row.error,
        "batch_run_id": int(row.batch_run_id) if row.batch_run_id is not None else None,
        "alert_budget": float(row.alert_budget) if row.alert_budget is not None else None,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "neighbourhood": _profile(session, int(row.tx_id)).to_dict(),
        "evidence": list_evidence(session, case_id),
    }


def _profile(session: Session, tx_id: int) -> NeighbourhoodProfile:
    return neighbourhood_profile(session, tx_id)


def list_evidence(session: Session, case_id: int) -> list[dict[str, Any]]:
    """Observed evidence for a case, strongest contribution first.

    The graph model's own score is excluded: it reaches the case page as
    `graph_score`, one of the three signals, not as a finding in the evidence
    list. Its row is still in the table.
    """
    rows = session.execute(
        text("""
        SELECT e.id, e.kind, e.summary, e.strength, e.weight,
               e.neighbour_tx_id, e.typology_reference_id, e.details, e.created_at,
               nt.timestep AS neighbour_timestep
        FROM evidence_items e
        LEFT JOIN transactions nt ON nt.tx_id = e.neighbour_tx_id
        WHERE e.case_report_id = :case_id
          AND e.kind = ANY(:observed_kinds)
        ORDER BY (e.strength * e.weight) DESC, e.id ASC
        """),
        {"case_id": case_id, "observed_kinds": _OBSERVED},
    ).all()

    return [
        {
            "id": int(row.id),
            "kind": row.kind,
            "summary": row.summary,
            "strength": float(row.strength),
            "weight": float(row.weight),
            "contribution": round(float(row.strength) * float(row.weight), 6),
            "neighbour_tx_id": (
                int(row.neighbour_tx_id) if row.neighbour_tx_id is not None else None
            ),
            "neighbour_timestep": (
                int(row.neighbour_timestep) if row.neighbour_timestep is not None else None
            ),
            "typology_reference_id": (
                int(row.typology_reference_id) if row.typology_reference_id is not None else None
            ),
            "details": row.details,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]


def get_case_by_tx(session: Session, tx_id: int) -> dict[str, Any] | None:
    case_id = session.execute(
        text("SELECT id FROM case_reports WHERE tx_id = :tx"), {"tx": tx_id}
    ).scalar_one_or_none()
    return get_case(session, int(case_id)) if case_id is not None else None


def get_batch_run(session: Session, timestep: int) -> dict[str, Any] | None:
    """Replay progress for one time step.

    Read from `batch_runs`, not from Celery's result backend: the database is
    the source of truth for job state, so status survives a worker restart.
    """
    row = session.execute(
        text("""
        SELECT b.*, (SELECT count(*) FROM case_reports c
                      JOIN transactions t ON t.tx_id = c.tx_id
                      WHERE t.timestep = b.timestep) AS cases
        FROM batch_runs b WHERE b.timestep = :timestep
        """),
        {"timestep": timestep},
    ).one_or_none()

    if row is None:
        return None

    return {
        "batch_run_id": int(row.id),
        "timestep": int(row.timestep),
        "status": row.status,
        "model_version": row.model_version,
        "alert_budget": float(row.alert_budget) if row.alert_budget is not None else None,
        "scored_count": int(row.scored_count),
        "queued_count": int(row.queued_count),
        "investigated_count": int(row.investigated_count),
        "failed_count": int(row.failed_count),
        "cases": int(row.cases),
        "error": row.error,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


def list_batch_runs(session: Session) -> list[dict[str, Any]]:
    rows = (
        session.execute(text("SELECT timestep FROM batch_runs ORDER BY timestep")).scalars().all()
    )
    return [get_batch_run(session, int(ts)) for ts in rows]
