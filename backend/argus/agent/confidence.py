"""Evidence confidence.

**Not a model score.** It answers a different question from the two model
outputs, and keeping the three apart is the point:

    Risk score          XGBoost's probability. Ranks the batch and decides
                        which transactions become alerts.
    Graph score         GraphSAGE's probability. An independent second
                        opinion. Decides nothing.
    Evidence confidence (this) How strongly the deterministic evidence Argus
                        actually gathered supports the case. Decides nothing.

It is a pure function of the evidence items on a case, so it exists as soon as
replay has gathered evidence -- no language model, no retrieval, no
investigation required. Running an investigation adds a narrative and
citations; it does not change this number unless the underlying evidence
changed.

What counts, and what does not
------------------------------
`graph_model_corroboration` carries weight **zero**. That item records the raw
GraphSAGE probability, and folding a model's own score into "how much evidence
is there" would make the two indistinguishable: a case with no evidence at all
would still score highly because a model said so. The graph score is shown
beside the confidence, never inside it.

Structural similarity is different and does count. It is a *measurement* made
using GraphSAGE embeddings -- this transaction sits near specific,
named, historically-confirmed illicit transactions -- not the model's opinion
about this transaction. That distinction is easy to lose and worth holding.

`typology_reference` also weighs zero: a retrieved passage explains why a
signal matters, it is not a signal, and a system that grew more confident the
more it read would be measuring its own reading.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.agent.evidence import EVIDENCE_WEIGHTS, WEIGHTS_VERSION
from argus.agent.state import EvidenceRecord

# Kinds excluded from the calculation, and why. Kept explicit rather than
# implied by a zero weight, so the reason survives a future weight edit.
NON_CONTRIBUTING = {
    "graph_model_corroboration": "a model's own score, not evidence",
    "typology_reference": "explains a signal; is not one",
}


@dataclass(frozen=True)
class ConfidenceResult:
    value: float
    version: str
    contributions: dict[str, float]
    #: Kinds present on the case that deliberately contributed nothing.
    excluded: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "confidence": self.value,
            "version": self.version,
            "contributions": self.contributions,
            "excluded": self.excluded,
        }


def _combine(strengths: list[float]) -> float:
    """Combine several items of the same kind into one score in 0..1.

    Noisy-OR: the chance that at least one of them is a genuine signal.
    Summing them instead would treat five near-identical similarity matches as
    five independent findings, which saturated the first real run at 1.0 for
    essentially every case and made the number useless.

    Five matches to the same cluster of known-illicit transactions are one
    observation held more firmly, not five findings. This says that:
    additional items of a kind raise its score with diminishing returns and
    can never take it past the weight assigned to the kind.
    """
    remaining = 1.0
    for strength in strengths:
        remaining *= 1.0 - max(0.0, min(1.0, strength))
    return 1.0 - remaining


def compute(evidence: list[EvidenceRecord]) -> ConfidenceResult:
    """Confidence from the deterministic evidence on a case.

    Each kind contributes at most its weight, however many items it has, so
    confidence rises by finding *different* kinds of support rather than more
    of the same. Corroboration across a heuristic, a neighbour and a
    similarity match is a stronger case than five similarity matches, and the
    arithmetic should say so.
    """
    by_kind: dict[str, list[float]] = {}
    for item in evidence:
        by_kind.setdefault(item.kind, []).append(item.strength)

    contributions: dict[str, float] = {}
    excluded: dict[str, float] = {}
    for kind, strengths in by_kind.items():
        weight = EVIDENCE_WEIGHTS.get(kind, 0.0)
        combined = round(weight * _combine(strengths), 6)
        if kind in NON_CONTRIBUTING or weight == 0.0:
            # Recorded so the UI can say "present, and deliberately not
            # counted" rather than leaving the reader to wonder.
            excluded[kind] = 0.0
        else:
            contributions[kind] = combined

    total = min(1.0, max(0.0, sum(contributions.values())))
    return ConfidenceResult(
        value=round(total, 4),
        version=f"{WEIGHTS_VERSION}-noisyor",
        contributions=contributions,
        excluded=excluded,
    )
