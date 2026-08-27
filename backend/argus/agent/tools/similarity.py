"""Structural similarity over GraphSAGE embeddings.

The one tool that reaches across time. Elliptic's graph has zero
cross-time-step edges, so a transaction's neighbours are always in its own
batch and adjacency can never connect it to labelled history. Similarity in
embedding space is not adjacency, so it can: given a queued transaction, this
finds transactions from the training range that the graph model represents the
same way.

The reference pool
------------------
Training-range time steps (1-29) **and** a known label. Validation and test
labels are excluded, because citing them would mean quoting an answer the
system is not supposed to have. Analyst-confirmed cases join the pool as they
accumulate; that clause is written but returns nothing until reviews exist.

The HNSW filtering trap
-----------------------
pgvector's HNSW index post-filters: it walks the graph for approximate
neighbours first, then applies `WHERE`. The reference pool is ~13% of the
table, so a default query walks 40 candidates, discards almost all of them,
and returns **zero rows** -- reproduced against real data, not
hypothetical.

`hnsw.iterative_scan = relaxed_order` (pgvector 0.8+) makes the scan continue
until it has enough rows that survive the filter. It is set per-session on the
connection this query runs on, and `verify_against_exact` checks the results
still match a brute-force scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.agent.evidence import EvidenceDraft
from argus.db.enums import EvidenceKind, Label
from argus.ml.splits import TRAIN_TIMESTEPS

DEFAULT_K = 5
# Above this cosine distance the "similar" claim stops being worth making.
MAX_COSINE_DISTANCE = 0.5
# Candidate list size. Larger costs time but reduces the chance the filtered
# scan comes up short.
EF_SEARCH = 200

REFERENCE_POOL_SQL = """
    t.timestep <= :train_max
    AND t.label <> :unknown
"""


@dataclass(frozen=True)
class SimilarTransaction:
    tx_id: int
    timestep: int
    label: str
    cosine_distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - self.cosine_distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "timestep": self.timestep,
            "label": self.label,
            "cosine_distance": round(self.cosine_distance, 6),
            "similarity": round(self.similarity, 6),
        }


def _enable_iterative_scan(session: Session) -> None:
    """Belt and braces.

    Every connection already carries these settings (see
    `argus.db.session.PGVECTOR_OPTIONS`), because forgetting them returns an
    empty result rather than a slow one. Re-asserting them here means this
    query is still correct on a session built some other way, and `SET LOCAL`
    reverts at the end of the transaction so nothing leaks out.
    """
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    session.execute(text(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}"))


def find_similar(
    session: Session,
    tx_id: int,
    k: int = DEFAULT_K,
    max_distance: float = MAX_COSINE_DISTANCE,
    *,
    exact: bool = False,
) -> list[SimilarTransaction]:
    """Nearest reference-pool transactions to `tx_id` in embedding space.

    Excludes the query transaction itself. With `exact`, the index is bypassed
    for a brute-force scan -- slow, but the ground truth the approximate path
    is checked against.
    """
    if exact:
        # `enable_indexscan` alone is not enough: the planner can still reach
        # the HNSW index through a bitmap scan.
        #
        # These are reset in the `finally` below rather than left to `SET
        # LOCAL`'s transaction scope. Releasing a savepoint does not undo a
        # SET LOCAL, so without the reset every later query on this session
        # would silently run without index scans -- which is exactly the bug
        # that made an earlier version of the regression test pass by
        # accident.
        session.execute(text("SET LOCAL enable_indexscan = off"))
        session.execute(text("SET LOCAL enable_bitmapscan = off"))
    else:
        _enable_iterative_scan(session)

    try:
        rows = session.execute(
            text(f"""
            WITH query AS (
                SELECT embedding FROM transaction_embeddings WHERE tx_id = :tx
            )
            SELECT t.tx_id, t.timestep, t.label,
                   e.embedding <=> (SELECT embedding FROM query) AS distance
            FROM transaction_embeddings e
            JOIN transactions t ON t.tx_id = e.tx_id
            WHERE {REFERENCE_POOL_SQL}
              AND t.tx_id <> :tx
            ORDER BY e.embedding <=> (SELECT embedding FROM query)
            LIMIT :k
            """),
            {
                "tx": tx_id,
                "k": k,
                "train_max": max(TRAIN_TIMESTEPS),
                "unknown": Label.UNKNOWN.value,
            },
        ).all()
    finally:
        if exact:
            session.execute(text("RESET enable_indexscan"))
            session.execute(text("RESET enable_bitmapscan"))

    return [
        SimilarTransaction(
            tx_id=int(row.tx_id),
            timestep=int(row.timestep),
            label=row.label,
            cosine_distance=float(row.distance),
        )
        for row in rows
        if row.distance is not None and float(row.distance) <= max_distance
    ]


def similarity_evidence(session: Session, tx_id: int, k: int = DEFAULT_K) -> list[EvidenceDraft]:
    """Turn similar historical transactions into citable evidence.

    Only illicit matches become evidence: a resemblance to a licit transaction
    is not a reason to escalate. The licit matches are still counted in the
    details, because "three of its five nearest historical neighbours were
    licit" is context an analyst should see.
    """
    matches = find_similar(session, tx_id, k=k)
    if not matches:
        return []

    illicit = [m for m in matches if m.label == Label.ILLICIT.value]
    if not illicit:
        return []

    return [
        EvidenceDraft(
            kind=EvidenceKind.STRUCTURAL_SIMILARITY,
            summary=(
                f"Network behaviour closely resembles transaction {match.tx_id} "
                f"(time step {match.timestep}), confirmed illicit in the training "
                f"period, at cosine similarity {match.similarity:.3f}."
            ),
            # A cosine distance of 0 is a perfect match; scale so that
            # distance 0 -> 1.0 and distance MAX -> 0.0.
            strength=max(0.0, 1.0 - match.cosine_distance / MAX_COSINE_DISTANCE),
            neighbour_tx_id=match.tx_id,
            details={
                "cosine_distance": round(match.cosine_distance, 6),
                "similarity": round(match.similarity, 6),
                "reference_timestep": match.timestep,
                "reference_label": match.label,
                "reference_pool": f"labelled transactions, time steps 1-{max(TRAIN_TIMESTEPS)}",
                "neighbours_considered": len(matches),
                "illicit_among_considered": len(illicit),
            },
        )
        for match in illicit
    ]


def verify_against_exact(session: Session, tx_ids: list[int], k: int = DEFAULT_K) -> dict[str, Any]:
    """Compare the indexed path against a brute-force scan.

    Returns per-sample overlap. HNSW is approximate, so perfect agreement is
    not required -- but a systematic shortfall means the filtered query is
    silently returning too little, which is the failure mode this guards.
    """
    samples = []
    for tx_id in tx_ids:
        approx = find_similar(session, tx_id, k=k)
        truth = find_similar(session, tx_id, k=k, exact=True)
        approx_ids = {m.tx_id for m in approx}
        truth_ids = {m.tx_id for m in truth}
        samples.append(
            {
                "tx_id": tx_id,
                "approx_count": len(approx),
                "exact_count": len(truth),
                "overlap": len(approx_ids & truth_ids),
                "recall": (len(approx_ids & truth_ids) / len(truth_ids)) if truth_ids else 1.0,
            }
        )

    recalls = [s["recall"] for s in samples]
    return {
        "samples": samples,
        "mean_recall": sum(recalls) / len(recalls) if recalls else 1.0,
        "empty_approximate_results": sum(1 for s in samples if s["approx_count"] == 0),
    }
