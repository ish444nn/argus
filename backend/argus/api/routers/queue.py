"""Risk queue, case detail and evidence endpoints.

Thin: every handler validates its inputs and calls one service function. The
query logic lives in `argus.services.queue`, which imports no ML libraries, so
these routes work in the light API image.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from argus.api.deps import SessionDep
from argus.api.schemas import CaseDetail, EvidenceItemOut, QueueEntryOut, QueuePage
from argus.db.enums import CaseStatus, Decision, QueueTier
from argus.services import queue as queue_service

router = APIRouter(prefix="/api", tags=["queue"])


@router.get("/queue", response_model=QueuePage)
def list_queue(
    session: SessionDep,
    timestep: int | None = None,
    status: CaseStatus | None = None,
    queue_tier: QueueTier | None = None,
    decision: Decision | None = None,
    undecided_only: bool = False,
    sort_by: Literal["risk_score", "queue_rank", "created_at", "graph_score"] = "risk_score",
    descending: bool = True,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> QueuePage:
    entries, total = queue_service.list_queue(
        session,
        timestep=timestep,
        status=status.value if status else None,
        queue_tier=queue_tier.value if queue_tier else None,
        decision=decision.value if decision else None,
        undecided_only=undecided_only,
        sort_by=sort_by,
        descending=descending,
        limit=limit,
        offset=offset,
    )
    return QueuePage(
        total=total,
        limit=limit,
        offset=offset,
        items=[QueueEntryOut(**entry.to_dict()) for entry in entries],
    )


@router.get("/cases/{case_id}", response_model=CaseDetail)
def get_case(case_id: int, session: SessionDep) -> CaseDetail:
    case = queue_service.get_case(session, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"no case {case_id}")
    return CaseDetail(**case)


@router.get("/transactions/{tx_id}/case", response_model=CaseDetail)
def get_case_for_transaction(tx_id: int, session: SessionDep) -> CaseDetail:
    """The case for a transaction, if it reached the queue."""
    case = queue_service.get_case_by_tx(session, tx_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"transaction {tx_id} has no case")
    return CaseDetail(**case)


@router.get("/cases/{case_id}/evidence", response_model=list[EvidenceItemOut])
def list_evidence(case_id: int, session: SessionDep) -> list[EvidenceItemOut]:
    if queue_service.get_case(session, case_id) is None:
        raise HTTPException(status_code=404, detail=f"no case {case_id}")
    return [EvidenceItemOut(**item) for item in queue_service.list_evidence(session, case_id)]
