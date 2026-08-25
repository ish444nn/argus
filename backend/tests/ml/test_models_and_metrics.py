"""Model output shapes, determinism, and the alert-budget metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from argus.ml import evaluate, features, gnn, graph, splits
from argus.ml.evaluate import ALERT_BUDGET

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_unknown_nodes_are_excluded_from_precision_and_recall():
    # Both `unknown` rows score above the threshold. If they counted as
    # negatives they would show up as two false positives and halve precision.
    scores = np.array([0.9, 0.8, 0.1, 0.7])
    labels = np.array(["illicit", "unknown", "licit", "unknown"])

    metrics = evaluate.evaluate(scores, labels, threshold=0.65)

    assert metrics.n_scored == 4
    assert metrics.n_labelled == 2
    assert metrics.n_illicit == 1
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.precision == 1.0


def test_unknown_nodes_still_consume_alert_budget():
    """An analyst queue does not know which rows carry ground truth."""
    scores = np.array([0.9, 0.95, 0.1, 0.2])
    labels = np.array(["illicit", "unknown", "licit", "licit"])

    metrics = evaluate.evaluate(scores, labels, threshold=0.5)

    assert metrics.alert_rate == pytest.approx(0.5)  # 2 of 4 rows alerted
    assert metrics.true_positives == 1


def test_alerts_at_budget_selects_exactly_k_even_with_ties():
    """Boosted probabilities saturate; a threshold would over-alert here."""
    scores = np.full(1000, 0.99)
    alerted = evaluate.alerts_at_budget(scores, budget=0.01)

    assert alerted.sum() == 10
    # A naive threshold comparison would alert all 1000.
    assert np.sum(scores >= evaluate.threshold_for_budget(scores, 0.01)) == 1000


def test_alerts_at_budget_picks_the_highest_scores():
    scores = np.arange(100, dtype=float)
    alerted = evaluate.alerts_at_budget(scores, budget=0.05)

    assert alerted.sum() == 5
    np.testing.assert_array_equal(np.nonzero(alerted)[0], [95, 96, 97, 98, 99])


def test_alerts_at_budget_is_deterministic():
    scores = np.tile([0.5, 0.9], 500)
    first = evaluate.alerts_at_budget(scores, budget=0.02)
    second = evaluate.alerts_at_budget(scores, budget=0.02)
    np.testing.assert_array_equal(first, second)


def test_budget_rejects_nonsensical_values():
    with pytest.raises(ValueError):
        evaluate.threshold_for_budget(np.array([0.5]), budget=0.0)
    with pytest.raises(ValueError):
        evaluate.threshold_for_budget(np.array([0.5]), budget=1.5)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate.evaluate(np.array([0.1, 0.2]), np.array(["licit"]), threshold=0.5)


def test_auprc_is_nan_when_only_one_class_is_present():
    metrics = evaluate.evaluate(np.array([0.9, 0.8]), np.array(["licit", "licit"]), threshold=0.5)
    assert np.isnan(metrics.auprc)


def test_report_split_measures_both_ways():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    labels = np.where(rng.random(1000) < 0.1, "illicit", "licit")

    report = evaluate.report_split(scores, labels, fixed_threshold=0.999)

    assert report.at_budget.alert_rate == pytest.approx(ALERT_BUDGET, abs=1e-3)
    # A punishing frozen threshold alerts far less than the budget allows.
    assert report.at_fixed_threshold.alert_rate < report.at_budget.alert_rate


# --------------------------------------------------------------------------
# GraphSAGE
# --------------------------------------------------------------------------


def _train_graph(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    return graph.build_split_graph(synthetic, x, masks.train)[0]


def test_graphsage_emits_64_dimensional_embeddings(synthetic):
    data = _train_graph(synthetic)
    model = gnn.GraphSAGE(in_channels=data.x.shape[1])

    scores, embeddings = gnn.predict(model, data)

    assert embeddings.shape == (data.num_nodes, 64)
    assert embeddings.shape[1] == gnn.EMBEDDING_DIM
    assert scores.shape == (data.num_nodes,)


def test_embedding_dim_matches_the_pgvector_column():
    from argus.db.models import TX_EMBEDDING_DIM

    assert gnn.EMBEDDING_DIM == TX_EMBEDDING_DIM == 64


def test_graph_risk_scores_are_probabilities(synthetic):
    data = _train_graph(synthetic)
    model = gnn.GraphSAGE(in_channels=data.x.shape[1])

    scores, _ = gnn.predict(model, data)

    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_prediction_is_deterministic_in_eval_mode(synthetic):
    """Dropout must be off, or embeddings would differ between runs."""
    data = _train_graph(synthetic)
    gnn.set_seed(0)
    model = gnn.GraphSAGE(in_channels=data.x.shape[1])

    first_scores, first_embeddings = gnn.predict(model, data)
    second_scores, second_embeddings = gnn.predict(model, data)

    np.testing.assert_array_equal(first_scores, second_scores)
    np.testing.assert_array_equal(first_embeddings, second_embeddings)


def test_same_seed_produces_the_same_initial_weights(synthetic):
    data = _train_graph(synthetic)
    gnn.set_seed(123)
    a = gnn.GraphSAGE(in_channels=data.x.shape[1])
    gnn.set_seed(123)
    b = gnn.GraphSAGE(in_channels=data.x.shape[1])

    for left, right in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right)


def test_positive_weight_is_the_licit_to_illicit_ratio(synthetic):
    data = _train_graph(synthetic)
    weight = gnn.positive_weight(data)

    y = data.y[data.supervised_mask]
    expected = float(len(y) - y.sum()) / float(y.sum())
    assert float(weight) == pytest.approx(expected)


def test_training_runs_and_improves_on_the_training_objective(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    scaler = features.fit_scaler(x, masks.train)
    scaled = scaler.transform(x)

    train_graph, _ = graph.build_split_graph(synthetic, scaled, masks.train)
    val_graph, _ = graph.build_split_graph(synthetic, scaled, masks.val)

    config = gnn.TrainingConfig(epochs=20, patience=100)
    model, history = gnn.train(train_graph, val_graph, config)

    assert history["best_epoch"] > 0
    assert history["history"], "expected at least one evaluation checkpoint"
    _, embeddings = gnn.predict(model, train_graph)
    assert embeddings.shape[1] == 64
