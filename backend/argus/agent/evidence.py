"""The evidence contract.

Every fact a case report rests on is an `EvidenceDraft` before it is a row.
The tools in `argus.agent.tools` return drafts; `persist` turns them into
`evidence_items`. Phase 4's investigation graph produces drafts from the same
tools and adds retrieval on top, so there is one evidence pipeline rather than
two.

Two fields carry the weight:

`strength`  how strong this particular signal is, normalised to 0..1. A
            fan-out of 40 is stronger evidence than a fan-out of 5.
`weight`    how much this *kind* of signal contributes to the deterministic
            confidence score. Fixed per kind, versioned, never inferred.

Confidence is computed from these in Phase 4. Nothing here asks a language
model for anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from argus.db.enums import EvidenceKind
from argus.db.models import EvidenceItem

# Per-kind contribution to confidence. Versioned as a set: changing any value
# means a new version, so a stored confidence always means the same thing.
EVIDENCE_WEIGHTS: dict[str, float] = {
    EvidenceKind.CONFIRMED_NEIGHBOUR: 0.40,
    EvidenceKind.STRUCTURAL_SIMILARITY: 0.25,
    EvidenceKind.FLAGGED_NEIGHBOUR: 0.20,
    EvidenceKind.HEURISTIC: 0.15,
    # Zero on purpose. This item records GraphSAGE's own probability, and
    # folding a model's score into "how much evidence is there" would collapse
    # two things the product keeps apart: a case with no evidence at all would
    # still score highly because a model said so. The graph score is displayed
    # beside the confidence, never inside it.
    #
    # Structural similarity above is the opposite case and does count: it is a
    # measurement made *using* the embeddings -- this transaction sits near
    # these named, historically-confirmed illicit ones -- not the model's
    # opinion about this transaction.
    EvidenceKind.GRAPH_MODEL_CORROBORATION: 0.0,
    # Explains a signal; is not one.
    EvidenceKind.TYPOLOGY_REFERENCE: 0.0,
}
WEIGHTS_VERSION = "w1"


@dataclass
class EvidenceDraft:
    """One fact, before it has an id."""

    kind: EvidenceKind
    summary: str
    strength: float
    # Provenance. A neighbour or similarity claim points at the transaction it
    # came from; a typology claim points at the corpus chunk. Free text is
    # never the only record of where a fact came from.
    neighbour_tx_id: int | None = None
    typology_reference_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in 0..1, got {self.strength}")
        if self.kind not in EVIDENCE_WEIGHTS:
            raise ValueError(f"no weight defined for evidence kind {self.kind}")

    @property
    def weight(self) -> float:
        return EVIDENCE_WEIGHTS[self.kind]

    def to_row(self, case_report_id: int) -> EvidenceItem:
        return EvidenceItem(
            case_report_id=case_report_id,
            kind=self.kind.value,
            summary=self.summary,
            strength=self.strength,
            weight=self.weight,
            neighbour_tx_id=self.neighbour_tx_id,
            typology_reference_id=self.typology_reference_id,
            details=self.details or None,
        )


def persist(
    session: Session,
    case_report_id: int,
    drafts: list[EvidenceDraft],
    *,
    replace: bool = True,
    kinds: Sequence[EvidenceKind] | None = None,
) -> int:
    """Write drafts as evidence rows.

    `replace` clears existing evidence first, because evidence is derived and
    a re-run should restate it rather than accumulate duplicates.

    `kinds` scopes that clearing to the kinds the caller actually produces. A
    caller should only ever delete what it is about to replace: replay
    regenerates deterministic evidence, and without a scope it also deleted the
    typology citations an investigation had written, leaving a stored narrative
    citing sources that no longer existed.
    """
    if replace:
        statement = delete(EvidenceItem).where(EvidenceItem.case_report_id == case_report_id)
        if kinds is not None:
            statement = statement.where(EvidenceItem.kind.in_([kind.value for kind in kinds]))
        session.execute(statement)
    session.add_all(draft.to_row(case_report_id) for draft in drafts)
    return len(drafts)
