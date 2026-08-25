"""The LangGraph workflow, end to end, against the live stack.

The provider is always a fake. Nothing here calls a real model: the tests must
run offline, and a test whose result depends on what a language model felt like
saying that day is not a test.
"""

from __future__ import annotations

import dataclasses

import pytest
from sqlalchemy import text

from argus.agent import corpus, retrieval
from argus.agent.graph import MAX_ATTEMPTS, InvestigationRunner, investigate
from argus.agent.llm import StubProvider
from argus.agent.schemas import Claim, Narrative
from argus.agent.state import InvestigationState
from argus.db.enums import CaseStatus, EvidenceKind

pytestmark = pytest.mark.integration


class ScriptedProvider:
    """Returns whatever the test hands it, in order."""

    name = "scripted"
    model = "test"

    def __init__(self, *responses: Narrative | None):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def narrate(self, prompt: str) -> Narrative | None:
        self.prompts.append(prompt)
        if not self.responses:
            return None
        return self.responses.pop(0)


class ExplodingProvider:
    name = "exploding"
    model = "test"

    def narrate(self, prompt: str) -> Narrative | None:
        raise RuntimeError("model unavailable")


@pytest.fixture(scope="module")
def db():
    from argus.db.session import SessionLocal

    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database not reachable: {exc}")
    yield session
    session.close()


@pytest.fixture(scope="module")
def seeded(db):
    """A corpus and at least one case to investigate."""
    cases = db.execute(text("SELECT count(*) FROM case_reports")).scalar_one()
    if not cases:
        pytest.skip("no cases; run a replay first")
    corpus.ingest(db)
    return db.execute(
        text("SELECT id FROM case_reports ORDER BY risk_score DESC LIMIT 1")
    ).scalar_one()


# --------------------------------------------------------------------------
# Retrieval against stored vectors
# --------------------------------------------------------------------------


def test_corpus_is_stored_with_embeddings(db, seeded):
    row = db.execute(
        text("""
        SELECT count(*) AS chunks,
               count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded,
               count(DISTINCT embedding_model) AS models,
               count(DISTINCT typology_id) AS sources
        FROM typology_references
        """)
    ).one()

    assert row.chunks > 0
    assert row.embedded == row.chunks
    assert row.models == 1, "the corpus must be embedded by a single model"
    assert row.sources >= 10


def test_stored_vectors_have_the_configured_dimension(db, seeded):
    dims = (
        db.execute(text("SELECT DISTINCT vector_dims(embedding) FROM typology_references"))
        .scalars()
        .all()
    )
    assert dims == [768]


def test_retrieval_only_returns_chunks_tagged_with_the_query_pattern(db, seeded):
    """Filter before rank: the guarantee that a fan-out is never explained by
    citing the funnelling note."""
    for pattern in ("structuring", "funnelling", "layering", "network_association"):
        result = retrieval.retrieve(
            db, retrieval.RetrievalQuery(text=pattern, patterns=[pattern]), k=5
        )
        assert result.chunks, f"nothing retrieved for {pattern}"
        for chunk in result.chunks:
            assert pattern in chunk.patterns


def test_retrieval_preserves_provenance(db, seeded):
    result = retrieval.retrieve(
        db, retrieval.RetrievalQuery(text="structuring", patterns=["structuring"]), k=3
    )
    for chunk in result.chunks:
        assert chunk.reference_id > 0
        assert chunk.publisher and chunk.source_url.startswith("http")
        assert chunk.citation()


def test_retrieval_returns_one_chunk_per_source(db, seeded):
    """Four results should be four documents, not four sections of one."""
    result = retrieval.retrieve(
        db, retrieval.RetrievalQuery(text="layering", patterns=["layering"]), k=4
    )
    ids = [chunk.typology_id for chunk in result.chunks]
    assert len(ids) == len(set(ids))


def test_retrieval_is_deterministic(db, seeded):
    query = retrieval.RetrievalQuery(text="layering", patterns=["layering"])
    first = retrieval.retrieve(db, query, k=4)
    second = retrieval.retrieve(db, query, k=4)
    assert [c.reference_id for c in first.chunks] == [c.reference_id for c in second.chunks]


def test_no_patterns_retrieves_nothing(db, seeded):
    result = retrieval.retrieve(db, retrieval.RetrievalQuery(text="anything", patterns=[]))
    assert result.chunks == []


def test_mismatched_embedding_spaces_are_refused(db, seeded):
    """Corpus embedded by one model and queried by another returns confident
    nonsense rather than an error, so it has to be an error."""

    class OtherEmbedder:
        model_name = "some-other-model"
        dimension = 768

        def embed(self, texts):
            return [[0.0] * 768 for _ in texts]

    with pytest.raises(ValueError, match="re-ingest"):
        retrieval.retrieve(
            db,
            retrieval.RetrievalQuery(text="x", patterns=["layering"]),
            embedder=OtherEmbedder(),
        )


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


def test_workflow_runs_end_to_end_with_the_stub(db, seeded):
    state = investigate(db, seeded, provider=StubProvider())

    assert state.error is None
    assert state.deterministic is not None
    assert state.generated.narrative is not None
    assert state.confidence is not None
    assert state.queue_tier in {"primary", "secondary"}
    # No key means the rule-built narrative, and that is a normal outcome.
    assert state.generated.used_fallback


def test_state_transitions_run_in_order(db, seeded):
    """Each node populates its own compartment and leaves earlier ones alone."""
    runner = InvestigationRunner(db, provider=StubProvider())
    state = InvestigationState(case_id=seeded)

    state = runner.load_case(state)
    assert state.deterministic is not None and not state.deterministic.evidence

    state = runner.collect_evidence(state)
    assert state.deterministic.evidence

    state = runner.build_query(state)
    assert state.retrieved.patterns and not state.retrieved.chunks

    state = runner.retrieve(state)
    assert state.retrieved.chunks

    state = runner.synthesize(state)
    assert state.generated.narrative is not None


def test_a_missing_case_stops_the_workflow(db, seeded):
    state = investigate(db, 99_999_999, provider=StubProvider())
    assert state.error is not None


def test_deterministic_evidence_is_immutable(db, seeded):
    """A frozen dataclass, so a generation step physically cannot rewrite a
    measurement."""
    runner = InvestigationRunner(db, provider=StubProvider())
    state = runner.collect_evidence(runner.load_case(InvestigationState(case_id=seeded)))

    # Frozen dataclasses raise FrozenInstanceError on assignment.
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.deterministic.risk_score = 0.0


def test_the_model_never_sets_confidence(db, seeded):
    """Two different narratives, identical evidence: confidence must not move."""
    allowed = _evidence_ids(db, seeded)
    sources = _source_ids(db, seeded)

    first = investigate(db, seeded, provider=ScriptedProvider(_valid(allowed, sources, "layering")))
    second = investigate(
        db,
        seeded,
        provider=ScriptedProvider(_valid(allowed, sources, "no_clear_typology")),
    )
    assert first.confidence == second.confidence


def test_a_valid_model_response_is_kept(db, seeded):
    allowed = _evidence_ids(db, seeded)
    sources = _source_ids(db, seeded)

    state = investigate(db, seeded, provider=ScriptedProvider(_valid(allowed, sources)))

    assert not state.generated.used_fallback
    assert state.generated.attempts == 1
    assert not state.generated.validation_errors


def test_an_invalid_citation_is_retried_then_falls_back(db, seeded):
    """The bounded retry. A response that cites what it was not given never
    reaches the database."""
    bad = Narrative(
        summary="Invented.",
        typology_assessment="structuring",
        typology_rationale="From [source fatf-2019-invented].",
        claims=[Claim(text="x", evidence_ids=[987654], source_ids=["fatf-2019-invented"])],
        recommended_action="escalate",
    )
    provider = ScriptedProvider(bad, bad)

    state = investigate(db, seeded, provider=provider)

    assert provider.prompts, "the provider was never called"
    assert state.generated.attempts == MAX_ATTEMPTS
    assert state.generated.used_fallback
    # The retry prompt must carry the specific complaint.
    assert "rejected" in provider.prompts[-1]

    stored = db.execute(
        text("SELECT narrative, narrative_source FROM case_reports WHERE id = :c"),
        {"c": seeded},
    ).one()
    assert stored.narrative_source == "template"
    assert "fatf-2019-invented" not in (stored.narrative or "")


def test_a_provider_failure_falls_back_rather_than_crashing(db, seeded):
    state = investigate(db, seeded, provider=ExplodingProvider())

    assert state.generated.used_fallback
    assert state.generated.narrative is not None
    assert any("provider error" in e for e in state.generated.provider_errors)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_the_report_is_persisted(db, seeded):
    investigate(db, seeded, provider=StubProvider())

    row = db.execute(
        text("""
        SELECT narrative, narrative_source, typology_assessment, recommended_action,
               confidence, confidence_version, queue_tier, status, investigation_meta
        FROM case_reports WHERE id = :c
        """),
        {"c": seeded},
    ).one()

    assert row.narrative
    assert row.narrative_source in {"llm", "template"}
    assert row.typology_assessment
    assert row.recommended_action
    assert row.confidence is not None
    assert row.confidence_version
    assert row.queue_tier in {"primary", "secondary"}
    assert row.status == CaseStatus.READY.value
    assert row.investigation_meta["prompt_version"]
    assert "generated_at" in row.investigation_meta


def test_deterministic_evidence_survives_the_investigation(db, seeded):
    """The load-bearing guarantee: an LLM workflow must not disturb what the
    system measured."""
    before = db.execute(
        text("""
        SELECT id, kind, summary, strength, weight FROM evidence_items
        WHERE case_report_id = :c AND kind <> :typology ORDER BY id
        """),
        {"c": seeded, "typology": EvidenceKind.TYPOLOGY_REFERENCE.value},
    ).all()

    investigate(db, seeded, provider=StubProvider())

    after = db.execute(
        text("""
        SELECT id, kind, summary, strength, weight FROM evidence_items
        WHERE case_report_id = :c AND kind <> :typology ORDER BY id
        """),
        {"c": seeded, "typology": EvidenceKind.TYPOLOGY_REFERENCE.value},
    ).all()

    assert before == after


def test_rerunning_does_not_duplicate_typology_evidence(db, seeded):
    investigate(db, seeded, provider=StubProvider())
    first = _typology_count(db, seeded)
    investigate(db, seeded, provider=StubProvider())
    assert _typology_count(db, seeded) == first


def test_persisted_citations_resolve_to_real_corpus_rows(db, seeded):
    """No stored citation may point at something that is not in the corpus."""
    investigate(db, seeded, provider=StubProvider())

    dangling = db.execute(
        text("""
        SELECT count(*) FROM evidence_items e
        LEFT JOIN typology_references r ON r.id = e.typology_reference_id
        WHERE e.case_report_id = :c AND e.kind = :kind AND r.id IS NULL
        """),
        {"c": seeded, "kind": EvidenceKind.TYPOLOGY_REFERENCE.value},
    ).scalar_one()
    assert dangling == 0


def test_typology_evidence_is_stored_with_zero_weight(db, seeded):
    investigate(db, seeded, provider=StubProvider())

    weights = (
        db.execute(
            text(
                "SELECT DISTINCT weight FROM evidence_items WHERE case_report_id = :c AND kind = :k"
            ),
            {"c": seeded, "k": EvidenceKind.TYPOLOGY_REFERENCE.value},
        )
        .scalars()
        .all()
    )
    assert weights in ([], [0.0])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _evidence_ids(session, case_id: int) -> list[int]:
    return (
        session.execute(
            text("SELECT id FROM evidence_items WHERE case_report_id = :c AND kind <> :k"),
            {"c": case_id, "k": EvidenceKind.TYPOLOGY_REFERENCE.value},
        )
        .scalars()
        .all()
    )


def _source_ids(session, case_id: int) -> list[str]:
    from argus.agent.graph import InvestigationRunner

    runner = InvestigationRunner(session, provider=StubProvider())
    state = runner.retrieve(
        runner.build_query(
            runner.collect_evidence(runner.load_case(InvestigationState(case_id=case_id)))
        )
    )
    return [chunk.typology_id for chunk in state.retrieved.chunks]


def _typology_count(session, case_id: int) -> int:
    return session.execute(
        text("SELECT count(*) FROM evidence_items WHERE case_report_id = :c AND kind = :k"),
        {"c": case_id, "k": EvidenceKind.TYPOLOGY_REFERENCE.value},
    ).scalar_one()


def _valid(evidence_ids, source_ids, typology="layering") -> Narrative:
    return Narrative(
        summary="Consistent with a known pattern.",
        typology_assessment=typology,
        typology_rationale=(
            f"See [source {source_ids[0]}]."
            if source_ids and typology != "no_clear_typology"
            else ""
        ),
        claims=[
            Claim(
                text="The transaction resembles previously confirmed activity.",
                evidence_ids=list(evidence_ids[:2]),
                source_ids=list(source_ids[:1]),
            )
        ],
        recommended_action="monitor",
    )
