"""Operational aggregates for the overview screen.

Every figure here is counted from the database. Nothing is estimated,
extrapolated or invented -- if a number cannot be derived from what the
pipeline actually wrote, it does not appear on the screen.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.agent.evidence import OBSERVED_KINDS
from argus.core.config import get_settings

# Score bands for the risk distribution. Wider at the bottom because that is
# where almost everything sits -- an even split would render as one bar.
RISK_BANDS = [
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 0.95, "0.8-0.95"),
    (0.95, 0.99, "0.95-0.99"),
    (0.99, 1.01, "0.99-1.0"),
]


def operations(session: Session, budget: float | None = None) -> dict[str, Any]:
    """Everything the overview shows, in one round trip per section.

    `budget` lets the screen ask "what would the queue look like at 2%?"
    without re-scoring anything. It changes only which transactions the
    distribution counts as alerted -- the model, its scores and the stored
    queue are untouched.
    """
    settings = get_settings()
    budget = settings.alert_budget if budget is None else budget

    batches = session.execute(
        text("""
        SELECT count(*) AS runs,
               coalesce(sum(scored_count), 0) AS scored,
               coalesce(sum(queued_count), 0) AS queued,
               max(timestep) AS latest_timestep,
               count(*) FILTER (WHERE status = 'running') AS running,
               count(*) FILTER (WHERE status = 'failed') AS failed
        FROM batch_runs
        """)
    ).one()

    cases = session.execute(
        text("""
        SELECT count(*) AS total,
               count(*) FILTER (WHERE status = 'queued') AS queued,
               count(*) FILTER (WHERE status = 'investigating') AS investigating,
               count(*) FILTER (WHERE status = 'ready') AS ready,
               count(*) FILTER (WHERE status = 'failed') AS failed,
               count(*) FILTER (WHERE narrative_source = 'llm') AS model_written,
               count(*) FILTER (WHERE narrative_source = 'template') AS rule_written
        FROM case_reports
        """)
    ).one()

    # Awaiting review = has a report, no decision recorded yet. This is the
    # number an analyst actually acts on, so it drives the overview.
    awaiting = session.execute(
        text("""
        SELECT count(*) FROM case_reports c
        WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.case_report_id = c.id)
        """)
    ).scalar_one()

    decisions = dict(
        session.execute(
            text("""
            SELECT latest.decision, count(*)
            FROM case_reports c
            JOIN LATERAL (
                SELECT r.decision FROM reviews r
                WHERE r.case_report_id = c.id
                ORDER BY r.created_at DESC LIMIT 1
            ) latest ON TRUE
            GROUP BY latest.decision
            """)
        ).all()
    )

    typologies = dict(
        session.execute(
            text("""
            SELECT typology_assessment, count(*) FROM case_reports
            WHERE typology_assessment IS NOT NULL
            GROUP BY typology_assessment ORDER BY 2 DESC
            """)
        ).all()
    )

    # Observed evidence only. The graph model's own score is a signal reported
    # per case, not a category of finding, so counting it here would put a
    # model's opinion in a chart of what the system found out.
    evidence = dict(
        session.execute(
            text(
                "SELECT kind, count(*) FROM evidence_items "
                "WHERE kind = ANY(:observed_kinds) "
                "GROUP BY kind ORDER BY 2 DESC"
            ),
            {"observed_kinds": list(OBSERVED_KINDS)},
        ).all()
    )

    return {
        "alert_budget": budget,
        "default_alert_budget": settings.alert_budget,
        "llm_provider": settings.llm_provider,
        "replay_range": [settings.replay_min_timestep, settings.replay_max_timestep],
        "batches": {
            "runs": int(batches.runs),
            "latest_timestep": (
                int(batches.latest_timestep) if batches.latest_timestep is not None else None
            ),
            "scored": int(batches.scored),
            "queued": int(batches.queued),
            "running": int(batches.running),
            "failed": int(batches.failed),
            # The realised alert rate across everything replayed. Shown next to
            # the configured budget so drift is visible rather than assumed.
            "realised_alert_rate": (
                round(int(batches.queued) / int(batches.scored), 5) if batches.scored else None
            ),
        },
        "cases": {
            "total": int(cases.total),
            "queued": int(cases.queued),
            "investigating": int(cases.investigating),
            "ready": int(cases.ready),
            "failed": int(cases.failed),
            "model_written": int(cases.model_written),
            "rule_written": int(cases.rule_written),
            "awaiting_review": int(awaiting),
        },
        "decisions": {key: int(value) for key, value in decisions.items()},
        "typologies": {key: int(value) for key, value in typologies.items()},
        "evidence": {key: int(value) for key, value in evidence.items()},
        "risk_distribution": risk_distribution(session, budget),
        "budget_preview": budget_preview(session, budget),
        "corpus": corpus_summary(session),
    }


def risk_distribution(session: Session, budget: float | None = None) -> list[dict[str, Any]]:
    """Score distribution across everything scored, with the alerted slice.

    Deliberately over `risk_scores` rather than `case_reports`: the queue is
    by construction the top slice, so a distribution of queued cases puts
    every row in the highest band and shows nothing. Plotting the whole scored
    population, with the alerted portion marked, is what makes the size of the
    cut legible -- and makes the honest answer visible to the obvious
    question, "what about the high-scoring transactions below the line?".

    `alerted` is what the current stored queue contains. `would_alert` is what
    the chosen budget *would* select, recomputed per batch by rank, so moving
    the control shows the cut moving without touching a single stored row.
    """
    budget = get_settings().alert_budget if budget is None else budget
    params: dict[str, Any] = {f"a{i}": band[0] for i, band in enumerate(RISK_BANDS)}
    params["budget"] = budget

    rows = session.execute(
        text("""
        WITH ranked AS (
            SELECT r.tx_id, r.score, t.timestep,
                   (c.id IS NOT NULL) AS queued,
                   -- Rank within the batch, because the budget is applied per
                   -- batch. A global rank would answer a different question.
                   row_number() OVER (
                       PARTITION BY t.timestep ORDER BY r.score DESC, r.tx_id
                   ) AS rank_in_batch,
                   count(*) OVER (PARTITION BY t.timestep) AS batch_size
            FROM risk_scores r
            JOIN transactions t ON t.tx_id = r.tx_id
            LEFT JOIN case_reports c ON c.tx_id = r.tx_id
        ),
        marked AS (
            SELECT score, queued,
                   rank_in_batch <= ceil(batch_size * :budget) AS would_alert
            FROM ranked
        )
        SELECT
          count(*) FILTER (WHERE score >= :a0 AND score < :a1) AS b0,
          count(*) FILTER (WHERE score >= :a1 AND score < :a2) AS b1,
          count(*) FILTER (WHERE score >= :a2 AND score < :a3) AS b2,
          count(*) FILTER (WHERE score >= :a3 AND score < :a4) AS b3,
          count(*) FILTER (WHERE score >= :a4) AS b4,
          count(*) FILTER (WHERE score >= :a0 AND score < :a1 AND queued) AS q0,
          count(*) FILTER (WHERE score >= :a1 AND score < :a2 AND queued) AS q1,
          count(*) FILTER (WHERE score >= :a2 AND score < :a3 AND queued) AS q2,
          count(*) FILTER (WHERE score >= :a3 AND score < :a4 AND queued) AS q3,
          count(*) FILTER (WHERE score >= :a4 AND queued) AS q4,
          count(*) FILTER (WHERE score >= :a0 AND score < :a1 AND would_alert) AS w0,
          count(*) FILTER (WHERE score >= :a1 AND score < :a2 AND would_alert) AS w1,
          count(*) FILTER (WHERE score >= :a2 AND score < :a3 AND would_alert) AS w2,
          count(*) FILTER (WHERE score >= :a3 AND score < :a4 AND would_alert) AS w3,
          count(*) FILTER (WHERE score >= :a4 AND would_alert) AS w4
        FROM marked
        """),
        params,
    ).one()

    return [
        {
            "band": RISK_BANDS[i][2],
            "count": int(getattr(rows, f"b{i}")),
            "alerted": int(getattr(rows, f"q{i}")),
            "would_alert": int(getattr(rows, f"w{i}")),
        }
        for i in range(len(RISK_BANDS))
    ]


def budget_preview(session: Session, budget: float) -> dict[str, Any]:
    """How many transactions this budget would select, and what it would miss.

    The "what about the rest?" number is deliberately prominent. A reviewer
    seeing 894 transactions scoring above 0.99 and only 275 alerts should be
    told plainly that the remainder are scored, stored and unreviewed -- that
    is what a capacity constraint means, and hiding it would be dishonest.
    """
    row = session.execute(
        text("""
        WITH ranked AS (
            SELECT r.score, t.timestep,
                   row_number() OVER (
                       PARTITION BY t.timestep ORDER BY r.score DESC, r.tx_id
                   ) AS rank_in_batch,
                   count(*) OVER (PARTITION BY t.timestep) AS batch_size
            FROM risk_scores r
            JOIN transactions t ON t.tx_id = r.tx_id
        )
        SELECT count(*) AS scored,
               count(*) FILTER (WHERE rank_in_batch <= ceil(batch_size * :budget)) AS selected,
               count(*) FILTER (WHERE score >= 0.99) AS high_scoring,
               count(*) FILTER (
                   WHERE score >= 0.99 AND rank_in_batch > ceil(batch_size * :budget)
               ) AS high_scoring_unselected
        FROM ranked
        """),
        {"budget": budget},
    ).one()

    return {
        "budget": budget,
        "scored": int(row.scored),
        "selected": int(row.selected),
        "high_scoring": int(row.high_scoring),
        "high_scoring_unselected": int(row.high_scoring_unselected),
    }


def corpus_summary(session: Session) -> dict[str, Any]:
    row = session.execute(
        text("""
        SELECT count(*) AS chunks,
               count(DISTINCT typology_id) AS sources,
               count(DISTINCT publisher) AS publishers,
               min(embedding_model) AS embedding_model
        FROM typology_references
        """)
    ).one()
    return {
        "chunks": int(row.chunks),
        "sources": int(row.sources),
        "publishers": int(row.publishers),
        "embedding_model": row.embedding_model,
    }


def neighbourhood(session: Session, tx_id: int, limit: int = 24) -> dict[str, Any]:
    """The one-hop ego network around a transaction.

    Capped, because the point is a readable sketch of the transaction's
    position rather than a complete rendering. `truncated` says when the cap
    bit, so the UI can say so instead of quietly showing a partial picture.
    """
    rows = session.execute(
        text("""
        SELECT e.src_tx_id, e.dst_tx_id, t.timestep, t.label,
               t.in_degree, t.out_degree,
               r.score AS risk_score,
               (c.id IS NOT NULL) AS flagged
        FROM (
            SELECT src_tx_id, dst_tx_id FROM edges WHERE src_tx_id = :tx
            UNION
            SELECT src_tx_id, dst_tx_id FROM edges WHERE dst_tx_id = :tx
        ) e
        JOIN transactions t
          ON t.tx_id = CASE WHEN e.src_tx_id = :tx THEN e.dst_tx_id ELSE e.src_tx_id END
        LEFT JOIN case_reports c ON c.tx_id = t.tx_id
        LEFT JOIN LATERAL (
            SELECT score FROM risk_scores rs
            WHERE rs.tx_id = t.tx_id ORDER BY created_at DESC LIMIT 1
        ) r ON TRUE
        ORDER BY r.score DESC NULLS LAST
        LIMIT :limit
        """),
        {"tx": tx_id, "limit": limit + 1},
    ).all()

    truncated = len(rows) > limit
    rows = rows[:limit]

    total = session.execute(
        text("SELECT in_degree + out_degree FROM transactions WHERE tx_id = :tx"),
        {"tx": tx_id},
    ).scalar_one_or_none()

    return {
        "tx_id": tx_id,
        "total_degree": int(total) if total is not None else 0,
        "truncated": truncated,
        "neighbours": [
            {
                "tx_id": int(row.dst_tx_id if row.src_tx_id == tx_id else row.src_tx_id),
                "direction": "out" if row.src_tx_id == tx_id else "in",
                "timestep": int(row.timestep),
                "label": row.label,
                "in_degree": int(row.in_degree),
                "out_degree": int(row.out_degree),
                "risk_score": float(row.risk_score) if row.risk_score is not None else None,
                "flagged": bool(row.flagged),
            }
            for row in rows
        ],
    }
