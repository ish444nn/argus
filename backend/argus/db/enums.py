"""Enumerated column values.

These are stored as VARCHAR with a CHECK constraint rather than as native
PostgreSQL ENUM types. Adding a value later (evidence kinds will grow in
Phase 4) is then a simple constraint replacement instead of an ALTER TYPE
dance, and the allowed values stay defined in one place in Python.
"""

from enum import StrEnum


class Label(StrEnum):
    ILLICIT = "illicit"
    LICIT = "licit"
    UNKNOWN = "unknown"


class CaseStatus(StrEnum):
    """Lifecycle of a case report, not the analyst's verdict."""

    QUEUED = "queued"
    INVESTIGATING = "investigating"
    READY = "ready"
    FAILED = "failed"


class QueueTier(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class EvidenceKind(StrEnum):
    FLAGGED_NEIGHBOUR = "flagged_neighbour"
    CONFIRMED_NEIGHBOUR = "confirmed_neighbour"
    HEURISTIC = "heuristic"
    TYPOLOGY_REFERENCE = "typology_reference"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    GRAPH_MODEL_CORROBORATION = "graph_model_corroboration"


class NarrativeSource(StrEnum):
    LLM = "llm"
    TEMPLATE = "template"


class Decision(StrEnum):
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"


class BatchStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def values(enum_cls: type[StrEnum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_cls)
