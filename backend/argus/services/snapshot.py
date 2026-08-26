"""Exporting a demo snapshot for the hosted database.

The hosted deployment serves the read side of Argus: the queue, cases,
evidence, investigations and decisions. It does not score batches, because a
background worker is not free. So the results of a local replay are exported
here and loaded into the hosted database once.

Two things shape what goes in.

**Size.** The full local database is about 384 MB, and Supabase's free tier
allows 500 MB. Almost all of the bulk is `transactions.features` -- 203,769
rows of 165 floats, about 168 MB -- and the API never reads it. Only the
worker scores transactions, and the worker is not hosted. The column is
therefore exported as NULL, which the schema already allows precisely for
this case.

**Scope.** Only the replayed range and what it touches. Earlier time steps are
training data; they exist locally as historical graph context and the hosted
API has no use for them, apart from the transactions the similarity evidence
cites, which must come along or the citations dangle.

The output is plain SQL that `psql` can apply. Nothing here is clever: the
point is that a person can read the file before loading it into a database
they cannot easily undo.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Order matters: parents before children, because the load runs inside one
# transaction with foreign keys enforced.
TABLE_ORDER = [
    "users",
    "transactions",
    "edges",
    "batch_runs",
    "risk_scores",
    "transaction_embeddings",
    "typology_references",
    "case_reports",
    "evidence_items",
    "reviews",
]


def _quote(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        # Postgres array literal. Used for `patterns`; `features` is always
        # NULL in a snapshot and never reaches here.
        inner = ",".join(str(item).replace("'", "''") for item in value)
        return "'{" + inner + "}'"
    if isinstance(value, dict):
        import json

        return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"
    return "'" + str(value).replace("'", "''") + "'"


def _rows(session: Session, sql: str, params: dict) -> Iterator[dict]:
    for row in session.execute(text(sql), params).mappings():
        yield dict(row)


def _insert(table: str, rows: list[dict]) -> Iterator[str]:
    if not rows:
        return
    columns = list(rows[0])
    yield f"-- {table}: {len(rows)} row(s)"
    for row in rows:
        values = ", ".join(_quote(row[column]) for column in columns)
        yield f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values});"
    yield ""


def export(session: Session, min_timestep: int = 35) -> str:
    """Build the seed SQL for the hosted database."""
    params = {"ts": min_timestep}
    lines: list[str] = [
        "-- Argus demo snapshot.",
        "--",
        "-- Apply to a database that has already been migrated:",
        '--   psql "$DATABASE_URL" -f demo-snapshot.sql',
        "--",
        "-- `transactions.features` is NULL throughout: the 166-feature arrays",
        "-- are ~168 MB and only the worker reads them, and the worker is not",
        "-- hosted. Everything the API serves is present.",
        "",
        "BEGIN;",
        "",
    ]

    # Transactions in the replayed range, plus any earlier transaction a piece
    # of evidence points at -- similarity cites the training period, and a
    # citation whose target is missing is worse than no citation.
    transactions = list(
        _rows(
            session,
            """
            SELECT tx_id, timestep, label,
                   NULL::real[] AS features,
                   in_degree, out_degree, created_at
            FROM transactions
            WHERE timestep >= :ts
               OR tx_id IN (
                    SELECT neighbour_tx_id FROM evidence_items
                    WHERE neighbour_tx_id IS NOT NULL
               )
            ORDER BY tx_id
            """,
            params,
        )
    )
    kept = {row["tx_id"] for row in transactions}

    tables: dict[str, list[dict]] = {
        "users": list(_rows(session, "SELECT * FROM users ORDER BY id", {})),
        "transactions": transactions,
        # Only edges whose endpoints both survived the filter.
        "edges": list(
            _rows(
                session,
                """
                SELECT e.* FROM edges e
                JOIN transactions s ON s.tx_id = e.src_tx_id
                JOIN transactions d ON d.tx_id = e.dst_tx_id
                WHERE s.timestep >= :ts AND d.timestep >= :ts
                ORDER BY e.src_tx_id, e.dst_tx_id
                """,
                params,
            )
        ),
        "batch_runs": list(_rows(session, "SELECT * FROM batch_runs ORDER BY id", {})),
        "risk_scores": list(
            _rows(
                session,
                """
                SELECT r.* FROM risk_scores r
                JOIN transactions t ON t.tx_id = r.tx_id
                WHERE t.timestep >= :ts ORDER BY r.id
                """,
                params,
            )
        ),
        # Embeddings for everything kept: the queued cases, and the training
        # transactions their similarity evidence cites.
        "transaction_embeddings": list(
            _rows(
                session,
                "SELECT * FROM transaction_embeddings WHERE tx_id = ANY(:ids) ORDER BY tx_id",
                {"ids": sorted(kept)},
            )
        ),
        "typology_references": list(
            _rows(session, "SELECT * FROM typology_references ORDER BY id", {})
        ),
        "case_reports": list(_rows(session, "SELECT * FROM case_reports ORDER BY id", {})),
        "evidence_items": list(_rows(session, "SELECT * FROM evidence_items ORDER BY id", {})),
        "reviews": list(_rows(session, "SELECT * FROM reviews ORDER BY id", {})),
    }

    for table in TABLE_ORDER:
        lines.extend(_insert(table, tables[table]))

    # Sequences would otherwise still be at 1, and the first insert on the
    # hosted database would collide with a seeded id.
    lines.append("-- Advance sequences past the seeded ids.")
    for table in (
        "users",
        "batch_runs",
        "risk_scores",
        "typology_references",
        "case_reports",
        "evidence_items",
        "reviews",
    ):
        lines.append(
            f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
            f"coalesce((SELECT max(id) FROM {table}), 1));"
        )

    lines += ["", "COMMIT;"]

    counts = {name: len(rows) for name, rows in tables.items()}
    log.info("snapshot: %s", counts)
    return "\n".join(lines)


def summarise(session: Session, min_timestep: int = 35) -> dict[str, int]:
    """Row counts a snapshot would contain, without building it."""
    return {
        "transactions": session.execute(
            text(
                "SELECT count(*) FROM transactions WHERE timestep >= :ts "
                "OR tx_id IN (SELECT neighbour_tx_id FROM evidence_items "
                "WHERE neighbour_tx_id IS NOT NULL)"
            ),
            {"ts": min_timestep},
        ).scalar_one(),
        "case_reports": session.execute(text("SELECT count(*) FROM case_reports")).scalar_one(),
        "evidence_items": session.execute(text("SELECT count(*) FROM evidence_items")).scalar_one(),
        "typology_references": session.execute(
            text("SELECT count(*) FROM typology_references")
        ).scalar_one(),
    }
