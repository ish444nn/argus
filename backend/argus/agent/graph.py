"""The investigation workflow.

    load_case -> collect_evidence -> build_query -> retrieve
              -> synthesize -> validate -> [retry once] -> persist

A fixed sequence with one conditional edge, not a free-running agent. Six of
the seven nodes are deterministic database work; the model is called once, at
`synthesize`, with evidence that has already been gathered. It cannot decide
what to look up, cannot look up more, and cannot revisit a step.

That is a deliberate trade. A tool-calling loop would look more autonomous and
would be worse here: non-reproducible, more expensive, harder to test, and it
would give the model the opportunity to introduce facts the system never
found. The interesting engineering is in what the model is *prevented* from
doing.

The one conditional edge is `validate`: a response citing ids it was not given
goes back to `synthesize` once, with the specific errors appended, and then
falls back to the rule-built narrative. The fallback is not an error path --
it is what runs whenever there is no API key, so it is exercised constantly.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, StateGraph
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from argus.agent import confidence as confidence_module
from argus.agent import retrieval, validation
from argus.agent.evidence import EvidenceDraft
from argus.agent.llm import PROMPT_VERSION, LLMProvider, get_provider
from argus.agent.prompts import build_prompt, build_template_narrative
from argus.agent.state import (
    DeterministicEvidence,
    EvidenceRecord,
    InvestigationState,
    RetrievedKnowledge,
)
from argus.agent.tools.graph_tools import neighbourhood_profile
from argus.db.enums import CaseStatus, EvidenceKind, NarrativeSource

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


class InvestigationRunner:
    """Holds the session and provider the nodes need.

    LangGraph state carries data between nodes; connections and clients are
    not data, so they live here rather than being threaded through the state.
    """

    def __init__(
        self,
        session: Session,
        provider: LLMProvider | None = None,
        embedder: Any | None = None,
    ):
        self.session = session
        self.provider = provider or get_provider()
        self.embedder = embedder
        self._prompt: str | None = None

    # ---------------------------------------------------------------- nodes

    def load_case(self, state: InvestigationState) -> InvestigationState:
        """Read the case and its measured facts. Nothing here is negotiable."""
        row = self.session.execute(
            sql("""
            SELECT c.id, c.tx_id, c.risk_score, c.model_version, c.queue_rank,
                   c.graph_score, t.timestep
            FROM case_reports c
            JOIN transactions t ON t.tx_id = c.tx_id
            WHERE c.id = :case_id
            """),
            {"case_id": state.case_id},
        ).one_or_none()

        if row is None:
            state.error = f"no case {state.case_id}"
            return state

        profile = neighbourhood_profile(self.session, int(row.tx_id))
        state.deterministic = DeterministicEvidence(
            case_id=int(row.id),
            tx_id=int(row.tx_id),
            timestep=int(row.timestep),
            risk_score=float(row.risk_score),
            model_version=row.model_version,
            queue_rank=int(row.queue_rank) if row.queue_rank is not None else None,
            graph_score=float(row.graph_score) if row.graph_score is not None else None,
            in_degree=profile.in_degree,
            out_degree=profile.out_degree,
            neighbour_count=profile.neighbour_count,
            chain_length=profile.chain_length,
            same_batch_neighbours=profile.same_batch_neighbours,
            flagged_neighbours=profile.flagged_neighbours,
            evidence=[],
        )
        return state

    def collect_evidence(self, state: InvestigationState) -> InvestigationState:
        """Load the evidence Phase 3 already persisted.

        Read, not regenerated. The deterministic evidence is the system's
        record of what it found, and an investigation that recomputed it could
        quietly disagree with the rows an analyst is looking at.
        """
        if state.deterministic is None:
            return state

        rows = self.session.execute(
            sql("""
            SELECT id, kind, summary, strength, weight, neighbour_tx_id, details
            FROM evidence_items
            WHERE case_report_id = :case_id AND kind <> :typology
            ORDER BY (strength * weight) DESC, id
            """),
            {
                "case_id": state.case_id,
                "typology": EvidenceKind.TYPOLOGY_REFERENCE.value,
            },
        ).all()

        records = [
            EvidenceRecord(
                id=int(row.id),
                kind=row.kind,
                summary=row.summary,
                strength=float(row.strength),
                weight=float(row.weight),
                neighbour_tx_id=(
                    int(row.neighbour_tx_id) if row.neighbour_tx_id is not None else None
                ),
                details=row.details or {},
            )
            for row in rows
        ]
        # Frozen dataclass: replaced wholesale rather than mutated.
        state.deterministic = DeterministicEvidence(
            **{**state.deterministic.__dict__, "evidence": records}
        )
        return state

    def build_query(self, state: InvestigationState) -> InvestigationState:
        if state.deterministic is None:
            return state
        query = retrieval.build_query(state.deterministic.evidence)
        state.retrieved = RetrievedKnowledge(query=query.text, patterns=query.patterns)
        return state

    def retrieve(self, state: InvestigationState) -> InvestigationState:
        if state.deterministic is None:
            return state
        query = retrieval.RetrievalQuery(
            text=state.retrieved.query, patterns=state.retrieved.patterns
        )
        state.retrieved = retrieval.retrieve(self.session, query, embedder=self.embedder)
        return state

    def synthesize(self, state: InvestigationState) -> InvestigationState:
        """The single language-model call."""
        if state.deterministic is None:
            return state

        state.generated.provider = self.provider.name
        state.generated.model = self.provider.model
        state.generated.attempts += 1

        prompt = build_prompt(state.deterministic, state.retrieved)
        if state.generated.validation_errors:
            prompt += (
                "\n\n"
                + validation.ValidationResult(
                    ok=False, errors=state.generated.validation_errors
                ).as_feedback()
            )
        self._prompt = prompt

        try:
            narrative = self.provider.narrate(prompt)
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("provider %s failed: %s", self.provider.name, exc)
            state.generated.provider_errors.append(f"provider error: {exc}")
            narrative = None

        if narrative is None:
            # No credentials, or the call failed. The template is the answer,
            # not a degraded one.
            state.generated.narrative = build_template_narrative(
                state.deterministic, state.retrieved
            )
            state.generated.used_fallback = True
        else:
            state.generated.narrative = narrative
        return state

    def validate_output(self, state: InvestigationState) -> InvestigationState:
        if state.generated.narrative is None or state.deterministic is None:
            return state
        if state.generated.used_fallback:
            # Built from the evidence by rule; there is nothing to check.
            state.generated.validation_errors = []
            return state

        result = validation.validate(
            state.generated.narrative,
            allowed_evidence_ids=state.deterministic.evidence_ids,
            allowed_source_ids=state.retrieved.source_ids,
        )
        state.generated.validation_errors = list(result.errors)
        if not result.ok:
            log.warning(
                "citation validation failed on attempt %d: %s",
                state.generated.attempts,
                "; ".join(result.errors),
            )
        return state

    def should_retry(self, state: InvestigationState) -> str:
        """The one conditional edge."""
        if not state.generated.validation_errors:
            return "persist"
        if state.generated.attempts >= MAX_ATTEMPTS:
            return "fallback"
        return "synthesize"

    def fallback(self, state: InvestigationState) -> InvestigationState:
        """Replace an unsupportable response with the rule-built one."""
        if state.deterministic is None:
            return state
        log.warning(
            "falling back to the template narrative for case %s after %d attempt(s)",
            state.case_id,
            state.generated.attempts,
        )
        state.generated.narrative = build_template_narrative(state.deterministic, state.retrieved)
        state.generated.used_fallback = True
        return state

    def persist(self, state: InvestigationState) -> InvestigationState:
        """Write the report. Deterministic evidence is left untouched."""
        if state.deterministic is None or state.generated.narrative is None:
            return state

        narrative = state.generated.narrative
        # One definition of the number: `score_confidence` reads the persisted
        # evidence and calls the same function replay does. The investigation
        # adds a narrative and citations; it does not get its own arithmetic,
        # and re-running it cannot move the confidence unless the evidence
        # itself moved.
        result = confidence_module.compute(state.deterministic.evidence)
        state.confidence = result.value
        state.confidence_version = result.version

        # Only typology evidence is replaced. Phase 3's measurements stay.
        self.session.execute(
            sql("""
            DELETE FROM evidence_items
            WHERE case_report_id = :case_id AND kind = :kind
            """),
            {"case_id": state.case_id, "kind": EvidenceKind.TYPOLOGY_REFERENCE.value},
        )

        cited = {source for claim in narrative.claims for source in claim.source_ids} or set(
            state.retrieved.source_ids
        )
        for chunk in state.retrieved.chunks:
            if chunk.typology_id not in cited:
                continue
            draft = EvidenceDraft(
                kind=EvidenceKind.TYPOLOGY_REFERENCE,
                summary=(f"{chunk.title} -- {chunk.section_heading} ({chunk.citation()})"),
                # Retrieval similarity, recorded but weighted zero: a citation
                # explains a signal, it is not one.
                strength=max(0.0, min(1.0, chunk.similarity)),
                typology_reference_id=chunk.reference_id,
                details={
                    "typology_id": chunk.typology_id,
                    "section": chunk.section_heading,
                    "publisher": chunk.publisher,
                    "source_url": chunk.source_url,
                    "document": chunk.document,
                    "year": chunk.year,
                    "similarity": round(chunk.similarity, 6),
                    "retrieved_for": state.retrieved.patterns,
                    "quoted_text": chunk.text,
                },
            )
            self.session.add(draft.to_row(state.case_id))

        self.session.execute(
            sql("""
            UPDATE case_reports SET
                narrative = :narrative,
                narrative_source = :source,
                typology_assessment = :typology,
                recommended_action = :action,
                confidence = :confidence,
                confidence_version = :confidence_version,
                status = :status,
                investigation_meta = CAST(:meta AS jsonb),
                error = NULL,
                updated_at = now()
            WHERE id = :case_id
            """),
            {
                "case_id": state.case_id,
                "narrative": _render(narrative),
                "source": (
                    NarrativeSource.TEMPLATE.value
                    if state.generated.used_fallback
                    else NarrativeSource.LLM.value
                ),
                "typology": narrative.typology_assessment,
                "action": narrative.recommended_action,
                "confidence": result.value,
                "confidence_version": result.version,
                "status": CaseStatus.READY.value,
                "meta": _meta_json(state, result),
            },
        )
        self.session.commit()
        return state


def _render(narrative) -> str:
    """The narrative as stored text.

    Claims are kept as separate lines with their citations attached, so the
    stored form is still checkable rather than flattened into prose.
    """
    lines = [narrative.summary.strip(), ""]
    if narrative.typology_rationale.strip():
        lines += [narrative.typology_rationale.strip(), ""]
    for claim in narrative.claims:
        refs = []
        if claim.evidence_ids:
            refs.append("evidence " + ", ".join(str(i) for i in claim.evidence_ids))
        if claim.source_ids:
            refs.append("sources " + ", ".join(claim.source_ids))
        suffix = f" [{'; '.join(refs)}]" if refs else ""
        lines.append(f"- {claim.text.strip()}{suffix}")
    return "\n".join(lines).strip()


def _meta_json(state: InvestigationState, result) -> str:
    import json

    narrative = state.generated.narrative
    return json.dumps(
        {
            "provider": state.generated.provider,
            "model": state.generated.model,
            "prompt_version": PROMPT_VERSION,
            "attempts": state.generated.attempts,
            "used_fallback": state.generated.used_fallback,
            "validation_errors": state.generated.validation_errors,
            "provider_errors": state.generated.provider_errors,
            "retrieval_patterns": state.retrieved.patterns,
            "retrieved_sources": [chunk.typology_id for chunk in state.retrieved.chunks],
            "confidence_contributions": result.contributions,
            # What was on the case but deliberately weighed nothing. Recorded
            # because "present and not counted" and "absent" are different
            # facts, and only one of them is a reason to distrust the score.
            "confidence_excluded": result.excluded,
            "claim_count": len(narrative.claims) if narrative else 0,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )


def build_graph(runner: InvestigationRunner):
    """Wire the nodes. Explicit transitions, one conditional edge."""
    graph = StateGraph(InvestigationState)

    graph.add_node("load_case", runner.load_case)
    graph.add_node("collect_evidence", runner.collect_evidence)
    graph.add_node("build_query", runner.build_query)
    graph.add_node("retrieve", runner.retrieve)
    graph.add_node("synthesize", runner.synthesize)
    graph.add_node("validate", runner.validate_output)
    graph.add_node("fallback", runner.fallback)
    graph.add_node("persist", runner.persist)

    graph.set_entry_point("load_case")
    graph.add_edge("load_case", "collect_evidence")
    graph.add_edge("collect_evidence", "build_query")
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate",
        runner.should_retry,
        {"synthesize": "synthesize", "fallback": "fallback", "persist": "persist"},
    )
    graph.add_edge("fallback", "persist")
    graph.add_edge("persist", END)
    return graph.compile()


def investigate(
    session: Session,
    case_id: int,
    provider: LLMProvider | None = None,
    embedder: Any | None = None,
) -> InvestigationState:
    """Run the workflow for one case."""
    runner = InvestigationRunner(session, provider=provider, embedder=embedder)
    compiled = build_graph(runner)
    result = compiled.invoke(InvestigationState(case_id=case_id))
    # LangGraph returns the state as a mapping when the schema is a dataclass.
    if isinstance(result, dict):
        state = InvestigationState(case_id=case_id)
        for key, value in result.items():
            setattr(state, key, value)
        return state
    return result
