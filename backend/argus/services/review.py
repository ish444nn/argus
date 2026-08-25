"""Analyst decisions.

The PRD's success criterion is that an analyst can move from the queue to a
recorded decision without leaving the dashboard, so this is deliberately small:
record a decision against a case, and read back what has been recorded.

Decisions are append-only. A case can be revisited and re-decided, and the
history is kept -- overwriting would destroy the record of what an analyst
concluded and when, which is the one thing a review log exists to preserve.
The queue reads the most recent decision.

There is no sign-in yet. Decisions are attributed to the seeded demo analyst;
authentication is the outstanding piece of the PRD's analyst flow.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from argus.db.enums import Decision
from argus.db.models import Review, User

log = logging.getLogger(__name__)

DEMO_ANALYST_EMAIL = "analyst@argus.local"
DEMO_ANALYST_NAME = "Demo Analyst"


def ensure_demo_analyst(session: Session) -> User:
    """The account decisions are attributed to until sign-in exists.

    The password hash is a placeholder, not a usable credential: nothing
    authenticates against it, and a real hash would imply a login path that
    has not been built.
    """
    user = session.scalar(select(User).where(User.email == DEMO_ANALYST_EMAIL))
    if user is None:
        user = User(
            email=DEMO_ANALYST_EMAIL,
            display_name=DEMO_ANALYST_NAME,
            password_hash="!no-login-yet",
            is_active=True,
        )
        session.add(user)
        session.commit()
        log.info("created the demo analyst account")
    return user


def record(
    session: Session, case_id: int, decision: str, note: str | None = None
) -> dict[str, Any]:
    """Record one decision against a case."""
    if decision not in {member.value for member in Decision}:
        raise ValueError(f"unknown decision {decision!r}")

    exists = session.execute(
        text("SELECT 1 FROM case_reports WHERE id = :case_id"), {"case_id": case_id}
    ).scalar_one_or_none()
    if exists is None:
        raise LookupError(f"no case {case_id}")

    analyst = ensure_demo_analyst(session)
    review = Review(
        case_report_id=case_id,
        user_id=analyst.id,
        decision=decision,
        note=(note or None),
    )
    session.add(review)
    session.commit()

    log.info("case %s recorded as %s", case_id, decision)
    return {
        "review_id": review.id,
        "case_id": case_id,
        "decision": decision,
        "note": review.note,
        "analyst": analyst.display_name,
        "created_at": review.created_at.isoformat(),
    }


def history(session: Session, case_id: int) -> list[dict[str, Any]]:
    """Every decision recorded against a case, newest first."""
    rows = session.execute(
        text("""
        SELECT r.id, r.decision, r.note, r.created_at, u.display_name
        FROM reviews r
        JOIN users u ON u.id = r.user_id
        WHERE r.case_report_id = :case_id
        ORDER BY r.created_at DESC, r.id DESC
        """),
        {"case_id": case_id},
    ).all()

    return [
        {
            "review_id": int(row.id),
            "decision": row.decision,
            "note": row.note,
            "analyst": row.display_name,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
