"""Alert-budget selection and the evidence contract.

Pure unit tests: no database, no model files.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.agent.evidence import EVIDENCE_WEIGHTS, EvidenceDraft
from argus.db.enums import EvidenceKind
from argus.ml.scoring import select_alerts

# --------------------------------------------------------------------------
# Alert budget
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "budget", "expected"),
    [
        (5507, 0.01, 56),  # the real time step 35
        (6393, 0.01, 64),
        (1000, 0.01, 10),
        (1000, 0.05, 50),
        (101, 0.01, 2),  # ceil, not round: 1.01 -> 2
        (100, 0.01, 1),
        (1, 0.01, 1),  # never selects zero
        (10, 1.0, 10),
    ],
)
def test_budget_selects_exactly_ceil_of_the_batch(n, budget, expected):
    scores = np.linspace(0, 1, n)
    assert select_alerts(scores, budget).size == expected


def test_selection_takes_the_highest_scores_in_order():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    selected = select_alerts(scores, budget=0.6)  # ceil(3.0) = 3

    np.testing.assert_array_equal(selected, [1, 3, 2])


def test_selection_is_not_a_threshold_when_scores_are_tied():
    """Every score identical: a threshold would alert all of them."""
    scores = np.full(1000, 0.97)
    selected = select_alerts(scores, budget=0.01)

    assert selected.size == 10
    assert np.sum(scores >= scores.min()) == 1000


def test_ties_break_deterministically():
    scores = np.tile([0.5, 0.9], 500)
    first = select_alerts(scores, 0.02)
    second = select_alerts(scores, 0.02)
    np.testing.assert_array_equal(first, second)


def test_empty_batch_selects_nothing():
    assert select_alerts(np.array([]), 0.01).size == 0


@pytest.mark.parametrize("budget", [0.0, -0.1, 1.5])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="alert budget"):
        select_alerts(np.array([0.5, 0.6]), budget)


# --------------------------------------------------------------------------
# Evidence contract
# --------------------------------------------------------------------------


def test_every_evidence_kind_has_a_weight():
    """A kind without a weight would silently contribute nothing later."""
    for kind in EvidenceKind:
        assert kind in EVIDENCE_WEIGHTS


def test_draft_rejects_out_of_range_strength():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="strength"):
            EvidenceDraft(kind=EvidenceKind.HEURISTIC, summary="x", strength=bad)


def test_draft_weight_comes_from_the_kind_not_the_caller():
    draft = EvidenceDraft(kind=EvidenceKind.CONFIRMED_NEIGHBOUR, summary="x", strength=0.5)
    assert draft.weight == EVIDENCE_WEIGHTS[EvidenceKind.CONFIRMED_NEIGHBOUR]


def test_typology_references_carry_no_confidence_weight():
    """A retrieved passage explains a signal; it is not itself a signal."""
    assert EVIDENCE_WEIGHTS[EvidenceKind.TYPOLOGY_REFERENCE] == 0.0


def test_confirmed_neighbour_outweighs_a_merely_flagged_one():
    assert (
        EVIDENCE_WEIGHTS[EvidenceKind.CONFIRMED_NEIGHBOUR]
        > EVIDENCE_WEIGHTS[EvidenceKind.FLAGGED_NEIGHBOUR]
    )


def test_draft_maps_provenance_onto_the_row():
    draft = EvidenceDraft(
        kind=EvidenceKind.STRUCTURAL_SIMILARITY,
        summary="resembles 42",
        strength=0.8,
        neighbour_tx_id=42,
        details={"cosine_distance": 0.1},
    )
    row = draft.to_row(case_report_id=7)

    assert row.case_report_id == 7
    assert row.kind == EvidenceKind.STRUCTURAL_SIMILARITY.value
    assert row.neighbour_tx_id == 42
    assert row.details == {"cosine_distance": 0.1}


# --------------------------------------------------------------------------
# Artifact locations
# --------------------------------------------------------------------------


def test_models_dir_is_found_by_walking_up_not_by_fixed_depth(tmp_path, monkeypatch):
    """Counting parent directories breaks in the container.

    On the host the package lives at `<repo>/backend/argus/ml/`, in the image
    at `/app/argus/ml/` with models bind-mounted at `/app/models`. A fixed
    depth resolves one of those to `/models` and the replay job cannot find
    its manifest.
    """
    from argus.ml import registry

    override = tmp_path / "elsewhere"
    override.mkdir()
    monkeypatch.setenv("ARGUS_MODELS_DIR", str(override))
    assert registry._default_models_dir() == override

    monkeypatch.delenv("ARGUS_MODELS_DIR")
    resolved = registry._default_models_dir()
    assert resolved.name == "models"
    assert resolved.is_dir(), f"{resolved} should exist in this checkout"


def test_model_dir_helper_respects_an_explicit_root(tmp_path):
    from argus.ml import registry

    assert registry.model_dir("xgb-all166", tmp_path) == tmp_path / "xgb-all166"
