"""Operational overview, neighbourhood graph and analyst decisions.

Thin: each handler validates and calls one service function.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from argus.api.deps import SessionDep
from argus.api.schemas import NeighbourhoodGraph, ReviewIn, ReviewOut
from argus.db.enums import Decision
from argus.services import overview as overview_service
from argus.services import queue as queue_service
from argus.services import review as review_service

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def get_overview(session: SessionDep) -> dict:
    """Counted from the database. No figure here is estimated or invented."""
    return overview_service.operations(session)


@router.get("/transactions/{tx_id}/neighbourhood", response_model=NeighbourhoodGraph)
def get_neighbourhood(tx_id: int, session: SessionDep) -> NeighbourhoodGraph:
    return NeighbourhoodGraph(**overview_service.neighbourhood(session, tx_id))


@router.get("/cases/{case_id}/reviews", response_model=list[ReviewOut])
def list_reviews(case_id: int, session: SessionDep) -> list[ReviewOut]:
    if queue_service.get_case(session, case_id) is None:
        raise HTTPException(status_code=404, detail=f"no case {case_id}")
    return [ReviewOut(**row) for row in review_service.history(session, case_id)]


@router.post("/cases/{case_id}/reviews", response_model=ReviewOut, status_code=201)
def create_review(case_id: int, body: ReviewIn, session: SessionDep) -> ReviewOut:
    """Record an analyst decision.

    Append-only: re-deciding a case adds to its history rather than replacing
    it, so what was concluded and when is never lost.
    """
    allowed = [member.value for member in Decision]
    if body.decision not in allowed:
        raise HTTPException(status_code=422, detail=f"decision must be one of {allowed}")
    try:
        result = review_service.record(session, case_id, body.decision, body.note)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ReviewOut(**result)
