"""Citation validation, deterministic confidence, and the query builder.

No database, no network. These are the rules that decide what the system is
willing to publish, so they are tested in isolation.
"""

from __future__ import annotations

from argus.agent import confidence, retrieval, validation
from argus.agent.evidence import EVIDENCE_WEIGHTS
from argus.agent.schemas import Claim, Narrative
from argus.agent.state import EvidenceRecord
from argus.db.enums import EvidenceKind


def make_narrative(**overrides) -> Narrative:
    defaults = {
        "summary": "A summary.",
        "typology_assessment": "layering",
        "typology_rationale": "Because of [source layering-chains].",
        "claims": [Claim(text="A claim.", evidence_ids=[1], source_ids=["layering-chains"])],
        "recommended_action": "monitor",
    }
    return Narrative(**{**defaults, **overrides})


def record(kind: str, strength: float, item_id: int = 1) -> EvidenceRecord:
    return EvidenceRecord(
        id=item_id,
        kind=kind,
        summary="x",
        strength=strength,
        weight=EVIDENCE_WEIGHTS[kind],
    )


# --------------------------------------------------------------------------
# Citation validation
# --------------------------------------------------------------------------


def test_a_well_cited_narrative_passes():
    result = validation.validate(make_narrative(), {1}, {"layering-chains"})
    assert result.ok, result.errors


def test_an_invented_evidence_id_is_rejected():
    narrative = make_narrative(
        claims=[Claim(text="x", evidence_ids=[999], source_ids=["layering-chains"])]
    )
    result = validation.validate(narrative, {1, 2}, {"layering-chains"})

    assert not result.ok
    assert any("999" in error for error in result.errors)


def test_an_invented_source_is_rejected():
    """The failure the corpus exists to prevent."""
    narrative = make_narrative(
        claims=[Claim(text="x", evidence_ids=[1], source_ids=["fatf-report-2019"])]
    )
    result = validation.validate(narrative, {1}, {"layering-chains"})

    assert not result.ok
    assert any("fatf-report-2019" in error for error in result.errors)


def test_a_claim_citing_nothing_is_rejected():
    narrative = make_narrative(claims=[Claim(text="Trust me.", evidence_ids=[], source_ids=[])])
    result = validation.validate(narrative, {1}, {"layering-chains"})

    assert not result.ok
    assert any("cites nothing" in error for error in result.errors)


def test_a_typology_asserted_without_a_source_is_rejected():
    """Naming a typology from the model's own knowledge is the substitution
    the whole retrieval layer exists to prevent."""
    narrative = make_narrative(
        typology_assessment="structuring",
        typology_rationale="It looks like structuring to me.",
        claims=[Claim(text="x", evidence_ids=[1], source_ids=[])],
    )
    result = validation.validate(narrative, {1}, {"layering-chains"})

    assert not result.ok
    assert any("without citing" in error for error in result.errors)


def test_no_clear_typology_needs_no_source():
    narrative = make_narrative(
        typology_assessment="no_clear_typology",
        typology_rationale="",
        claims=[Claim(text="x", evidence_ids=[1], source_ids=[])],
    )
    assert validation.validate(narrative, {1}, set()).ok


def test_a_rationale_citation_satisfies_the_typology_rule():
    narrative = make_narrative(
        typology_rationale="See [source layering-chains].",
        claims=[Claim(text="x", evidence_ids=[1], source_ids=[])],
    )
    assert validation.validate(narrative, {1}, {"layering-chains"}).ok


def test_an_off_list_typology_is_rejected():
    result = validation.validate(
        make_narrative(typology_assessment="terrorism_financing"), {1}, {"layering-chains"}
    )
    assert not result.ok


def test_an_off_list_action_is_rejected():
    result = validation.validate(
        make_narrative(recommended_action="freeze_account"), {1}, {"layering-chains"}
    )
    assert not result.ok


def test_errors_become_prompt_feedback_for_the_retry():
    result = validation.validate(
        make_narrative(claims=[Claim(text="x", evidence_ids=[42])]), {1}, {"layering-chains"}
    )
    feedback = result.as_feedback()
    assert "rejected" in feedback and "42" in feedback


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------


def test_confidence_is_zero_without_evidence():
    """No evidence, no confidence -- and no error."""
    result = confidence.compute([])
    assert result.value == 0.0
    assert result.contributions == {}


def test_one_kind_can_never_exceed_its_weight():
    """The fix for the first real run, where five near-identical similarity
    matches summed past 1.0 and made confidence meaningless."""
    many = [record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.99, i) for i in range(20)]
    result = confidence.compute(many)

    assert result.value <= EVIDENCE_WEIGHTS[EvidenceKind.STRUCTURAL_SIMILARITY] + 1e-9


def test_more_items_of_one_kind_help_but_with_diminishing_returns():
    one = confidence.compute([record(EvidenceKind.HEURISTIC, 0.5, 1)]).value
    two = confidence.compute(
        [record(EvidenceKind.HEURISTIC, 0.5, 1), record(EvidenceKind.HEURISTIC, 0.5, 2)]
    ).value
    three = confidence.compute([record(EvidenceKind.HEURISTIC, 0.5, i) for i in range(3)]).value

    assert one < two < three
    assert (two - one) > (three - two)


def test_diverse_evidence_beats_repetition():
    """The property worth having: corroboration across kinds should win."""
    repeated = confidence.compute(
        [record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, i) for i in range(5)]
    )
    diverse = confidence.compute(
        [
            record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1),
            record(EvidenceKind.HEURISTIC, 0.9, 2),
            # A contributing third kind: graph corroboration weighs nothing,
            # so using it here would not demonstrate diversity at all.
            record(EvidenceKind.FLAGGED_NEIGHBOUR, 0.9, 3),
        ]
    )
    assert diverse.value > repeated.value


def test_the_graph_score_does_not_move_evidence_confidence():
    """The load-bearing separation.

    `graph_model_corroboration` records GraphSAGE's own probability. Folding a
    model's score into "how much evidence is there" would make the two
    indistinguishable, so it weighs nothing -- whatever its value.
    """
    evidence = [record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, i) for i in range(3)]

    baseline = confidence.compute(evidence).value
    for graph_score in (0.0, 0.5, 0.99, 1.0):
        with_graph = confidence.compute(
            [*evidence, record(EvidenceKind.GRAPH_MODEL_CORROBORATION, graph_score, 99)]
        )
        assert with_graph.value == baseline, f"graph score {graph_score} moved confidence"
        assert "graph_model_corroboration" in with_graph.excluded


def test_a_graph_score_alone_is_not_evidence():
    """A confident model with nothing behind it must score zero."""
    result = confidence.compute([record(EvidenceKind.GRAPH_MODEL_CORROBORATION, 1.0, 1)])
    assert result.value == 0.0


def test_similarity_still_counts_even_though_it_uses_the_same_model():
    """The distinction that is easy to lose.

    Structural similarity is a measurement made *using* GraphSAGE embeddings --
    this transaction sits near these named, historically-confirmed illicit ones
    -- not the model's opinion about this transaction. It counts.
    """
    result = confidence.compute([record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1)])
    assert result.value > 0
    assert "structural_similarity" in result.contributions


def test_typology_references_never_raise_confidence():
    """Reading more sources must not make the system more certain."""
    base = confidence.compute([record(EvidenceKind.HEURISTIC, 0.8, 1)])
    with_sources = confidence.compute(
        [
            record(EvidenceKind.HEURISTIC, 0.8, 1),
            record(EvidenceKind.TYPOLOGY_REFERENCE, 1.0, 2),
            record(EvidenceKind.TYPOLOGY_REFERENCE, 1.0, 3),
        ]
    )
    assert with_sources.value == base.value


def test_confidence_is_bounded():
    everything = [record(kind, 1.0, i) for i, kind in enumerate(EVIDENCE_WEIGHTS)]
    assert 0.0 <= confidence.compute(everything).value <= 1.0


def test_confidence_reports_what_it_excluded():
    """Present-but-not-counted is different from absent, and the UI says so."""
    result = confidence.compute(
        [
            record(EvidenceKind.CONFIRMED_NEIGHBOUR, 1.0, 1),
            record(EvidenceKind.GRAPH_MODEL_CORROBORATION, 0.9, 2),
            record(EvidenceKind.TYPOLOGY_REFERENCE, 0.8, 3),
        ]
    )
    assert set(result.contributions) == {"confirmed_neighbour"}
    assert set(result.excluded) == {"graph_model_corroboration", "typology_reference"}


def test_confidence_is_deterministic():
    evidence = [
        record(EvidenceKind.HEURISTIC, 0.4, 1),
        record(EvidenceKind.FLAGGED_NEIGHBOUR, 0.7, 2),
    ]
    assert confidence.compute(evidence) == confidence.compute(list(reversed(evidence)))


def test_version_string_changes_with_the_scheme():
    """A stored confidence must be interpretable against how it was made."""
    version = confidence.compute([]).version
    # No threshold in the version any more: the tier it used to gate is gone,
    # and confidence no longer decides anything.
    assert "noisyor" in version
    assert "t0." not in version


# --------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------


def test_query_maps_heuristics_to_their_patterns():
    evidence = [
        EvidenceRecord(
            id=1,
            kind="heuristic",
            summary="",
            strength=0.5,
            weight=0.15,
            details={"heuristic": "fan_out"},
        ),
    ]
    query = retrieval.build_query(evidence)
    assert "structuring" in query.patterns


def test_query_covers_similarity_and_corroboration():
    """The two that matter most: the queue rarely fires a heuristic."""
    evidence = [
        record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1),
        record(EvidenceKind.GRAPH_MODEL_CORROBORATION, 0.9, 2),
    ]
    query = retrieval.build_query(evidence)

    assert "behavioural_similarity" in query.patterns
    assert "model_risk_scoring" in query.patterns


def test_query_is_deterministic():
    evidence = [record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1)]
    assert retrieval.build_query(evidence) == retrieval.build_query(evidence)


def test_no_evidence_produces_no_patterns():
    query = retrieval.build_query([])
    assert query.patterns == []


# --------------------------------------------------------------------------
# What a narrative may cite
# --------------------------------------------------------------------------


def deterministic(*records) -> object:
    from argus.agent.state import DeterministicEvidence

    return DeterministicEvidence(
        case_id=1,
        tx_id=2,
        timestep=35,
        risk_score=0.99,
        model_version="xgb-all166",
        queue_rank=1,
        graph_score=0.98,
        in_degree=1,
        out_degree=1,
        neighbour_count=2,
        chain_length=1,
        same_batch_neighbours=2,
        flagged_neighbours=0,
        evidence=list(records),
    )


def test_a_narrative_may_not_cite_the_second_opinion():
    """The graph score is a signal, not a finding.

    It is reported next to the risk score and the confidence, and the case
    page's evidence list does not contain it -- so a claim citing its id would
    point the reader at something they cannot find. Holding it out of the
    citable set is what keeps the report traceable.
    """
    similarity = record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1)
    graph = record(EvidenceKind.GRAPH_MODEL_CORROBORATION, 0.98, 2)
    state = deterministic(similarity, graph)

    assert state.evidence_ids == {1}
    assert [item.id for item in state.observed] == [1]
    # ...but the row is still there, because typology retrieval keys off it.
    assert len(state.evidence) == 2
    assert "model_risk_scoring" in retrieval.build_query(state.evidence).patterns


def test_the_rule_built_narrative_cites_only_what_the_page_shows():
    from argus.agent.prompts import build_template_narrative
    from argus.agent.state import RetrievedKnowledge

    state = deterministic(
        record(EvidenceKind.STRUCTURAL_SIMILARITY, 0.9, 1),
        record(EvidenceKind.GRAPH_MODEL_CORROBORATION, 0.98, 2),
    )
    narrative = build_template_narrative(state, RetrievedKnowledge())

    cited = {i for claim in narrative.claims for i in claim.evidence_ids}
    assert cited <= state.evidence_ids, "the template cited an id the page does not list"
    assert 2 not in cited
