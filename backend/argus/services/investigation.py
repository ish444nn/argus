"""Running an investigation and reporting its outcome.

A thin layer over `argus.agent.graph`: it marks the case as under
investigation, runs the workflow, and turns the resulting state into something
a job result or an API response can carry.

The status transitions matter for the UI. A case goes
`queued -> investigating -> ready`, or `queued -> investigating -> failed`, and
the transition is committed before the slow part starts so a caller polling the
case can tell the difference between "not started" and "in progress".
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.db.enums import CaseStatus, EvidenceKind

log = logging.getLogger(__name__)


def _set_status(session: Session, case_id: int, status: str, error: str | None = None) -> None:
    session.execute(
        text(
            "UPDATE case_reports SET status = :status, error = :error, "
            "updated_at = now() WHERE id = :case_id"
        ),
        {"case_id": case_id, "status": status, "error": error},
    )
    session.commit()


def investigate(session: Session, case_id: int) -> dict[str, Any]:
    """Investigate one case. Safe to re-run."""
    from argus.agent.graph import investigate as run_graph

    exists = session.execute(
        text("SELECT 1 FROM case_reports WHERE id = :case_id"), {"case_id": case_id}
    ).scalar_one_or_none()
    if exists is None:
        raise ValueError(f"no case {case_id}")

    _set_status(session, case_id, CaseStatus.INVESTIGATING.value)

    try:
        state = run_graph(session, case_id)
    except Exception as exc:
        session.rollback()
        _set_status(session, case_id, CaseStatus.FAILED.value, error=str(exc)[:2000])
        log.exception("investigation of case %s failed", case_id)
        raise

    if state.error:
        _set_status(session, case_id, CaseStatus.FAILED.value, error=state.error)
        return {"case_id": case_id, "status": CaseStatus.FAILED.value, "error": state.error}

    narrative = state.generated.narrative
    return {
        "case_id": case_id,
        "status": CaseStatus.READY.value,
        "confidence": state.confidence,
        "confidence_version": state.confidence_version,
        "typology_assessment": narrative.typology_assessment if narrative else None,
        "recommended_action": narrative.recommended_action if narrative else None,
        "provider": state.generated.provider,
        "model": state.generated.model,
        "used_fallback": state.generated.used_fallback,
        "attempts": state.generated.attempts,
        "retrieved_sources": [chunk.typology_id for chunk in state.retrieved.chunks],
        "evidence_items": len(state.deterministic.evidence) if state.deterministic else 0,
    }


def list_cited_sources(session: Session, case_id: int) -> list[dict[str, Any]]:
    """The typology passages this case's report cites.

    Read from `evidence_items` joined to `typology_references`, so a citation
    in a report is always resolvable to the row it came from rather than to a
    string in the narrative.
    """
    rows = session.execute(
        text("""
        SELECT e.id AS evidence_id, e.summary, e.strength, e.details,
               r.id AS reference_id, r.typology_id, r.title, r.publisher,
               r.source_url, r.document, r.year, r.section_heading, r.text,
               r.patterns
        FROM evidence_items e
        JOIN typology_references r ON r.id = e.typology_reference_id
        WHERE e.case_report_id = :case_id AND e.kind = :kind
        ORDER BY e.strength DESC, e.id
        """),
        {"case_id": case_id, "kind": EvidenceKind.TYPOLOGY_REFERENCE.value},
    ).all()

    return [
        {
            "evidence_id": int(row.evidence_id),
            "reference_id": int(row.reference_id),
            "typology_id": row.typology_id,
            "title": row.title,
            "publisher": row.publisher,
            "source_url": row.source_url,
            "document": row.document,
            "year": int(row.year) if row.year is not None else None,
            "section_heading": row.section_heading,
            "text": row.text,
            "patterns": list(row.patterns),
            "similarity": float(row.strength),
            "retrieved_for": (row.details or {}).get("retrieved_for", []),
        }
        for row in rows
    ]
