"""Deterministic graph evidence.

Every function here is a SQL query over the transaction graph. No language
model is involved, and nothing is inferred that the data cannot show.

What the dataset permits
------------------------
Elliptic's features are anonymised: there are no amounts, no wall-clock times
and no addresses. The PRD's original heuristics (round-amount structuring,
rapid pass-through, burst frequency) are therefore not computable, and were
redefined as network shapes that are:

    fan_out          one in, many out -- the shape structuring makes
    fan_in           many in, one out -- the shape funnelling makes
    layering_chain   a run of one-in/one-out hops -- the shape layering makes
    dense_cluster    neighbours that transact with each other
    relay_chain      pass-through, but only when it sits inside a chain

Thresholds are set from the graph's own distribution, measured over all
203,769 transactions rather than guessed:

    in-degree   median 1, p95 2,  p99 9,  max 284
    out-degree  median 1, p95 2,  p99 5,  max 472

That distribution is why bare pass-through is not evidence on its own: 74,310
transactions (36%) are exactly one-in/one-out, so a rule firing on that shape
alone would flag a third of the network and mean nothing. `relay_chain` fires
only when the pass-through is part of a run of them, which is both rarer and
the thing layering actually looks like.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.agent.evidence import EvidenceDraft
from argus.db.enums import EvidenceKind, Label

# Fire thresholds, and the value at which strength saturates at 1.0.
FAN_OUT_MIN = 5
FAN_OUT_SATURATION = 40
FAN_IN_MIN = 5
FAN_IN_SATURATION = 40
CHAIN_MIN = 3
CHAIN_SATURATION = 8
CHAIN_MAX_DEPTH = 8
CLUSTER_MIN_NEIGHBOURS = 3

# Measured base rates, carried into evidence details so an analyst can see how
# selective a fired heuristic actually is.
BASE_RATES = {
    "fan_out": 0.0106,
    "fan_in": 0.0190,
    "relay_chain": 0.0,  # populated per batch; the bare shape is 0.365
    "bare_relay": 0.3647,
}


def _scale(value: float, floor: float, saturation: float) -> float:
    """Map a raw count onto 0..1 between its firing threshold and saturation."""
    if saturation <= floor:
        return 1.0
    return float(min(1.0, max(0.0, (value - floor) / (saturation - floor))))


@dataclass(frozen=True)
class NeighbourhoodProfile:
    """The structural facts about one transaction.

    Returned live by the case endpoint rather than persisted as evidence: it
    is the raw material the heuristics are computed from, not a signal in its
    own right.
    """

    tx_id: int
    timestep: int
    in_degree: int
    out_degree: int
    total_degree: int
    neighbour_count: int
    same_batch_neighbours: int
    flagged_neighbours: int
    neighbour_mean_risk: float | None
    chain_length: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def neighbourhood_profile(session: Session, tx_id: int) -> NeighbourhoodProfile:
    """Degrees, neighbour counts and same-batch statistics for one transaction.

    `neighbour_count` is distinct neighbours, which is smaller than
    total degree whenever a pair transacts in both directions.
    """
    row = session.execute(
        text("""
        WITH neighbours AS (
            SELECT dst_tx_id AS nb FROM edges WHERE src_tx_id = :tx
            UNION
            SELECT src_tx_id FROM edges WHERE dst_tx_id = :tx
        )
        SELECT
            t.timestep,
            t.in_degree,
            t.out_degree,
            (SELECT count(*) FROM neighbours) AS neighbour_count,
            (SELECT count(*) FROM neighbours n
                JOIN transactions nt ON nt.tx_id = n.nb
                WHERE nt.timestep = t.timestep) AS same_batch,
            (SELECT count(*) FROM neighbours n
                JOIN case_reports c ON c.tx_id = n.nb) AS flagged,
            (SELECT avg(r.score) FROM neighbours n
                JOIN risk_scores r ON r.tx_id = n.nb) AS mean_risk
        FROM transactions t WHERE t.tx_id = :tx
        """),
        {"tx": tx_id},
    ).one()

    return NeighbourhoodProfile(
        tx_id=tx_id,
        timestep=int(row.timestep),
        in_degree=int(row.in_degree),
        out_degree=int(row.out_degree),
        total_degree=int(row.in_degree) + int(row.out_degree),
        neighbour_count=int(row.neighbour_count),
        same_batch_neighbours=int(row.same_batch),
        flagged_neighbours=int(row.flagged),
        neighbour_mean_risk=float(row.mean_risk) if row.mean_risk is not None else None,
        chain_length=chain_length(session, tx_id),
    )


def chain_length(session: Session, tx_id: int, max_depth: int = CHAIN_MAX_DEPTH) -> int:
    """Length of the pass-through run this transaction sits in.

    Walks forwards and backwards while every hop is one-in/one-out, capped at
    `max_depth` in each direction so a pathological graph cannot make this
    expensive. Returns 1 for a transaction that is not itself pass-through.
    """
    row = session.execute(
        text("""
        WITH RECURSIVE chainable AS (
            SELECT tx_id FROM transactions WHERE in_degree <= 1 AND out_degree <= 1
        ),
        forward(cur, depth) AS (
            SELECT tx_id, 1 FROM chainable WHERE tx_id = :tx
            UNION ALL
            SELECT e.dst_tx_id, f.depth + 1
            FROM forward f
            JOIN edges e ON e.src_tx_id = f.cur
            JOIN chainable c ON c.tx_id = e.dst_tx_id
            WHERE f.depth < :max_depth
        ),
        backward(cur, depth) AS (
            SELECT tx_id, 1 FROM chainable WHERE tx_id = :tx
            UNION ALL
            SELECT e.src_tx_id, b.depth + 1
            FROM backward b
            JOIN edges e ON e.dst_tx_id = b.cur
            JOIN chainable c ON c.tx_id = e.src_tx_id
            WHERE b.depth < :max_depth
        )
        SELECT
            coalesce((SELECT max(depth) FROM forward), 0) AS fwd,
            coalesce((SELECT max(depth) FROM backward), 0) AS bwd
        """),
        {"tx": tx_id, "max_depth": max_depth},
    ).one()

    if not row.fwd and not row.bwd:
        return 1  # not a pass-through transaction
    # Both directions count the transaction itself, hence the -1.
    return int(row.fwd) + int(row.bwd) - 1


def _cluster_density(session: Session, tx_id: int) -> tuple[int, int, float]:
    """Edges among this transaction's neighbours, over the possible number."""
    row = session.execute(
        text("""
        WITH neighbours AS (
            SELECT dst_tx_id AS nb FROM edges WHERE src_tx_id = :tx
            UNION
            SELECT src_tx_id FROM edges WHERE dst_tx_id = :tx
        )
        SELECT
            (SELECT count(*) FROM neighbours) AS n,
            (SELECT count(*) FROM edges e
                WHERE e.src_tx_id IN (SELECT nb FROM neighbours)
                  AND e.dst_tx_id IN (SELECT nb FROM neighbours)) AS interlinks
        """),
        {"tx": tx_id},
    ).one()

    n = int(row.n)
    links = int(row.interlinks)
    possible = n * (n - 1)  # directed pairs
    return n, links, (links / possible if possible else 0.0)


def structural_heuristics(session: Session, tx_id: int) -> list[EvidenceDraft]:
    """Run every heuristic, returning drafts only for the ones that fire."""
    profile = neighbourhood_profile(session, tx_id)
    drafts: list[EvidenceDraft] = []

    if profile.out_degree >= FAN_OUT_MIN and profile.in_degree <= 2:
        drafts.append(
            EvidenceDraft(
                kind=EvidenceKind.HEURISTIC,
                summary=(
                    f"Fan-out: splits into {profile.out_degree} outgoing transactions "
                    f"from {profile.in_degree} incoming. Consistent with structuring."
                ),
                strength=_scale(profile.out_degree, FAN_OUT_MIN, FAN_OUT_SATURATION),
                details={
                    "heuristic": "fan_out",
                    "pattern": "structuring",
                    "out_degree": profile.out_degree,
                    "in_degree": profile.in_degree,
                    "threshold": FAN_OUT_MIN,
                    "base_rate": BASE_RATES["fan_out"],
                },
            )
        )

    if profile.in_degree >= FAN_IN_MIN and profile.out_degree <= 2:
        drafts.append(
            EvidenceDraft(
                kind=EvidenceKind.HEURISTIC,
                summary=(
                    f"Fan-in: consolidates {profile.in_degree} incoming transactions "
                    f"into {profile.out_degree} outgoing. Consistent with funnelling."
                ),
                strength=_scale(profile.in_degree, FAN_IN_MIN, FAN_IN_SATURATION),
                details={
                    "heuristic": "fan_in",
                    "pattern": "funnelling",
                    "in_degree": profile.in_degree,
                    "out_degree": profile.out_degree,
                    "threshold": FAN_IN_MIN,
                    "base_rate": BASE_RATES["fan_in"],
                },
            )
        )

    if profile.chain_length >= CHAIN_MIN:
        is_relay = profile.in_degree <= 1 and profile.out_degree <= 1
        drafts.append(
            EvidenceDraft(
                kind=EvidenceKind.HEURISTIC,
                summary=(
                    f"Layering chain: sits in a run of {profile.chain_length} "
                    f"consecutive pass-through transactions."
                ),
                strength=_scale(profile.chain_length, CHAIN_MIN, CHAIN_SATURATION),
                details={
                    "heuristic": "relay_chain" if is_relay else "layering_chain",
                    "pattern": "layering",
                    "chain_length": profile.chain_length,
                    "threshold": CHAIN_MIN,
                    "max_depth_searched": CHAIN_MAX_DEPTH,
                    # A bare one-in/one-out shape covers 36% of the graph, so
                    # it is only meaningful as part of a chain.
                    "bare_relay_base_rate": BASE_RATES["bare_relay"],
                },
            )
        )

    neighbours, interlinks, density = _cluster_density(session, tx_id)
    if neighbours >= CLUSTER_MIN_NEIGHBOURS and interlinks > 0:
        drafts.append(
            EvidenceDraft(
                kind=EvidenceKind.HEURISTIC,
                summary=(
                    f"Dense cluster: {interlinks} transactions among its "
                    f"{neighbours} counterparties also transact with each other."
                ),
                strength=min(1.0, density * 2),
                details={
                    "heuristic": "dense_cluster",
                    "pattern": "layering",
                    "neighbours": neighbours,
                    "interlinks": interlinks,
                    "density": round(density, 4),
                },
            )
        )

    return drafts


def flagged_neighbours(
    session: Session, tx_id: int, timestep: int, limit: int = 10
) -> list[EvidenceDraft]:
    """Neighbours the system already has reason to distrust.

    Three sources, all of which the system legitimately holds at the time of
    the batch:

    1. neighbours an analyst has confirmed illicit;
    2. neighbours the model has already flagged into the queue;
    3. neighbours labelled illicit in a *strictly earlier* time step.

    The third is the historical-intelligence channel. On Elliptic it always
    returns nothing, because the graph has zero cross-time-step edges -- but
    the query is written correctly so the logic holds for any dataset, and its
    emptiness is exactly why structural similarity exists.

    What this never does is read the ground-truth label of an unreviewed
    transaction in the current batch. That would be reading the answer.
    """
    rows = session.execute(
        text("""
        WITH neighbours AS (
            SELECT dst_tx_id AS nb, 'sends_to' AS direction FROM edges WHERE src_tx_id = :tx
            UNION
            SELECT src_tx_id, 'receives_from' FROM edges WHERE dst_tx_id = :tx
        )
        SELECT
            n.nb, n.direction, t.timestep, t.label,
            c.id AS case_id, c.risk_score,
            EXISTS (
                SELECT 1 FROM reviews r
                WHERE r.case_report_id = c.id AND r.decision = 'confirmed'
            ) AS analyst_confirmed
        FROM neighbours n
        JOIN transactions t ON t.tx_id = n.nb
        LEFT JOIN case_reports c ON c.tx_id = n.nb
        WHERE c.id IS NOT NULL
           OR (t.label = :illicit AND t.timestep < :timestep)
        ORDER BY c.risk_score DESC NULLS LAST
        LIMIT :limit
        """),
        {
            "tx": tx_id,
            "timestep": timestep,
            "illicit": Label.ILLICIT.value,
            "limit": limit,
        },
    ).all()

    drafts: list[EvidenceDraft] = []
    for row in rows:
        confirmed = bool(row.analyst_confirmed)
        historical = row.case_id is None
        if confirmed:
            kind = EvidenceKind.CONFIRMED_NEIGHBOUR
            summary = (
                f"Connected transaction {row.nb} ({row.direction}) was confirmed "
                f"illicit by an analyst."
            )
            strength = 1.0
        elif historical:
            kind = EvidenceKind.CONFIRMED_NEIGHBOUR
            summary = (
                f"Connected transaction {row.nb} ({row.direction}) is known illicit "
                f"from time step {row.timestep}, before this batch."
            )
            strength = 1.0
        else:
            kind = EvidenceKind.FLAGGED_NEIGHBOUR
            summary = (
                f"Connected transaction {row.nb} ({row.direction}) was itself flagged "
                f"by the model at risk {row.risk_score:.3f}."
            )
            strength = float(row.risk_score)

        drafts.append(
            EvidenceDraft(
                kind=kind,
                summary=summary,
                strength=strength,
                neighbour_tx_id=int(row.nb),
                details={
                    "direction": row.direction,
                    "neighbour_timestep": int(row.timestep),
                    "source": (
                        "analyst_review"
                        if confirmed
                        else "historical_label"
                        if historical
                        else "model_flagged"
                    ),
                    "neighbour_risk_score": (
                        float(row.risk_score) if row.risk_score is not None else None
                    ),
                },
            )
        )
    return drafts
