"""API response models."""

from typing import Literal

from pydantic import BaseModel


class DependencyStatus(BaseModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Overall readiness plus a per-dependency breakdown.

    Always returned with HTTP 200 so the dashboard can render which dependency
    is down instead of just seeing a failed request.
    """

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    dependencies: dict[str, DependencyStatus]


class TaskDispatched(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    task_id: str
    state: str
    result: dict | None = None


# --------------------------------------------------------------------------
# Queue, cases and evidence
# --------------------------------------------------------------------------


class QueueEntryOut(BaseModel):
    case_id: int
    tx_id: int
    timestep: int
    risk_score: float
    queue_rank: int | None
    graph_score: float | None
    status: str
    queue_tier: str | None
    confidence: float | None
    evidence_count: int
    latest_decision: str | None
    created_at: str


class QueuePage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[QueueEntryOut]


class EvidenceItemOut(BaseModel):
    id: int
    kind: str
    summary: str
    strength: float
    weight: float
    # strength x weight -- what this item will contribute to the deterministic
    # confidence score in Phase 4.
    contribution: float
    neighbour_tx_id: int | None
    neighbour_timestep: int | None
    typology_reference_id: int | None
    details: dict | None
    created_at: str


class NeighbourhoodOut(BaseModel):
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


class CaseDetail(BaseModel):
    case_id: int
    tx_id: int
    timestep: int
    label: str
    risk_score: float
    model_version: str
    queue_rank: int | None
    graph_score: float | None
    status: str
    queue_tier: str | None
    confidence: float | None
    confidence_version: str | None
    narrative: str | None
    narrative_source: str | None
    typology_assessment: str | None
    recommended_action: str | None
    investigation_meta: dict | None
    error: str | None
    batch_run_id: int | None
    alert_budget: float | None
    created_at: str
    updated_at: str
    neighbourhood: NeighbourhoodOut
    evidence: list[EvidenceItemOut]


class ReplayDispatched(BaseModel):
    task_id: str
    timestep: int
    alert_budget: float
    status_url: str


class BatchRunOut(BaseModel):
    batch_run_id: int
    timestep: int
    status: str
    model_version: str | None
    alert_budget: float | None
    scored_count: int
    queued_count: int
    investigated_count: int
    failed_count: int
    cases: int
    error: str | None
    started_at: str | None
    finished_at: str | None


class CitedSource(BaseModel):
    """A typology passage a report cites, resolved back to its corpus row."""

    evidence_id: int
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
    retrieved_for: list[str]


class InvestigationDispatched(BaseModel):
    task_id: str
    case_id: int
    provider: str
    status_url: str


class NeighbourOut(BaseModel):
    tx_id: int
    direction: str
    timestep: int
    label: str
    in_degree: int
    out_degree: int
    risk_score: float | None
    flagged: bool


class NeighbourhoodGraph(BaseModel):
    tx_id: int
    total_degree: int
    truncated: bool
    neighbours: list[NeighbourOut]


class ReviewIn(BaseModel):
    decision: str
    note: str | None = None


class ReviewOut(BaseModel):
    review_id: int
    decision: str
    note: str | None
    analyst: str
    created_at: str
