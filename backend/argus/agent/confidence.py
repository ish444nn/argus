"""Deterministic confidence.

A pure function of assembled evidence. The language model never sets it, never
sees it, and is explicitly told not to state one of its own -- a model's
self-reported certainty is not evidence of anything, and having two numbers
compete would leave the analyst deciding which to believe.

    confidence = min(1, sum(strength x weight))

Weights are per evidence kind, fixed in `argus.agent.evidence`, and versioned
as a set: a stored confidence is always interpretable against the weights that
produced it.

Typology references carry weight 0 by design. A retrieved passage explains why
a signal matters; it is not itself a signal, and letting citations raise
confidence would mean the system grew more certain the more it read.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.agent.evidence import WEIGHTS_VERSION
from argus.agent.state import EvidenceRecord
from argus.db.enums import QueueTier

# Reports at or above this reach the analyst's primary queue. Everything else
# is retained in the secondary queue -- held, not discarded, so no finding
# disappears silently.
PRIMARY_QUEUE_THRESHOLD = 0.35


@dataclass(frozen=True)
class ConfidenceResult:
    value: float
    version: str
    queue_tier: str
    contributions: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "confidence": self.value,
            "version": self.version,
            "queue_tier": self.queue_tier,
            "contributions": self.contributions,
        }


def _combine(strengths: list[float]) -> float:
    """Combine several items of the same kind into one score in 0..1.

    Noisy-OR: the chance that at least one of them is a genuine signal. Summing
    them instead would treat five near-identical similarity matches as five
    independent findings, which saturated the first real run at 1.0 for
    essentially every case and made the number useless.

    Five matches to the same cluster of known-illicit transactions are one
    observation, held more firmly than a single match would be. This says that:
    additional items of a kind raise its score with diminishing returns and can
    never take it past the weight assigned to the kind.
    """
    remaining = 1.0
    for strength in strengths:
        remaining *= 1.0 - max(0.0, min(1.0, strength))
    return 1.0 - remaining


def compute(evidence: list[EvidenceRecord]) -> ConfidenceResult:
    """Confidence, its version, and the tier it implies.

    Each kind contributes at most its weight, however many items it has, so
    confidence rises by finding *different* kinds of support rather than more
    of the same. That is the behaviour worth having: corroboration across a
    heuristic, a neighbour and a similarity match is a stronger case than five
    similarity matches, and the arithmetic should say so.
    """
    by_kind: dict[str, list[float]] = {}
    weights: dict[str, float] = {}
    for item in evidence:
        by_kind.setdefault(item.kind, []).append(item.strength)
        weights[item.kind] = item.weight

    contributions = {
        kind: round(weights[kind] * _combine(strengths), 6) for kind, strengths in by_kind.items()
    }

    total = min(1.0, max(0.0, sum(contributions.values())))
    tier = (
        QueueTier.PRIMARY.value if total >= PRIMARY_QUEUE_THRESHOLD else QueueTier.SECONDARY.value
    )
    return ConfidenceResult(
        value=round(total, 4),
        version=f"{WEIGHTS_VERSION}-noisyor+t{PRIMARY_QUEUE_THRESHOLD}",
        queue_tier=tier,
        contributions=contributions,
    )
