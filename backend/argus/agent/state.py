"""Investigation state.

Three compartments, and the separation is the point:

`DeterministicEvidence`  what Argus measured -- scores, ranks, graph
                         statistics, heuristics, similarity matches. Frozen.
`RetrievedKnowledge`     what was retrieved from the typology corpus, with the
                         provenance needed to cite it. Frozen.
`GeneratedAssessment`    what the language model wrote.

The first two are frozen dataclasses, so a generation step physically cannot
overwrite a measurement. That is a stronger guarantee than a convention, and it
is the structural reason a case report's facts survive whatever the model says
about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from argus.agent.schemas import Narrative


@dataclass(frozen=True)
class EvidenceRecord:
    """One persisted evidence item, as the workflow sees it."""

    id: int
    kind: str
    summary: str
    strength: float
    weight: float
    neighbour_tx_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def contribution(self) -> float:
        return self.strength * self.weight


@dataclass(frozen=True)
class RetrievedChunk:
    """One typology passage, with everything needed to cite it."""

    reference_id: int
    typology_id: str
    title: str
    publisher: str
    source_url: str
    document: str | None
    year: int | None
    section_heading: str
    text: str
    patterns: list[str]
    similarity: float

    def citation(self) -> str:
        parts = [self.publisher]
        if self.document:
            parts.append(f'"{self.document}"')
        if self.year:
            parts.append(str(self.year))
        return ", ".join(parts)


@dataclass(frozen=True)
class DeterministicEvidence:
    """Measured facts. Nothing downstream may change these."""

    case_id: int
    tx_id: int
    timestep: int
    risk_score: float
    model_version: str
    queue_rank: int | None
    graph_score: float | None
    in_degree: int
    out_degree: int
    neighbour_count: int
    chain_length: int
    same_batch_neighbours: int
    flagged_neighbours: int
    evidence: list[EvidenceRecord] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[EvidenceRecord]:
        return [item for item in self.evidence if item.kind == kind]

    @property
    def evidence_ids(self) -> set[int]:
        return {item.id for item in self.evidence}


@dataclass(frozen=True)
class RetrievedKnowledge:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    patterns: list[str] = field(default_factory=list)

    @property
    def source_ids(self) -> set[str]:
        """Citation keys the model is permitted to use."""
        return {chunk.typology_id for chunk in self.chunks}


@dataclass
class GeneratedAssessment:
    """Model output. Mutable, because validation may replace it."""

    narrative: Narrative | None = None
    provider: str = ""
    model: str = ""
    attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)
    # Kept apart from validation errors: a failed call and an unsupported
    # citation are different problems, and clearing one must not hide the
    # other from the stored generation metadata.
    provider_errors: list[str] = field(default_factory=list)
    used_fallback: bool = False


@dataclass
class InvestigationState:
    """What flows between the graph's nodes."""

    case_id: int
    deterministic: DeterministicEvidence | None = None
    retrieved: RetrievedKnowledge = field(default_factory=RetrievedKnowledge)
    generated: GeneratedAssessment = field(default_factory=GeneratedAssessment)
    confidence: float | None = None
    confidence_version: str = ""
    queue_tier: str | None = None
    error: str | None = None
