"""Temporal split boundaries and the leakage guards that protect them.

These are the tests worth having. Everything else in the ML pipeline produces
a number that is obviously wrong when it breaks; a leak produces a number that
looks *better* than it should, which is exactly the failure nobody notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.ml import baseline, features, graph, splits
from argus.ml.splits import Split


def test_splits_are_contiguous_and_ordered():
    train, val, test = (
        splits.TRAIN_TIMESTEPS,
        splits.VAL_TIMESTEPS,
        splits.TEST_TIMESTEPS,
    )
    assert train.stop == val.start, "validation must begin where training ends"
    assert val.stop == test.start, "test must begin where validation ends"
    assert train.start == 1 and test.stop - 1 == 49


def test_no_timestep_appears_in_two_splits():
    seen = [
        set(splits.TRAIN_TIMESTEPS),
        set(splits.VAL_TIMESTEPS),
        set(splits.TEST_TIMESTEPS),
    ]
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            assert not seen[a] & seen[b]
    assert set().union(*seen) == set(range(1, 50))


def test_masks_partition_every_node_exactly_once(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    total = masks.train.astype(int) + masks.val.astype(int) + masks.test.astype(int)
    assert np.all(total == 1)


def test_build_masks_rejects_an_incomplete_partition():
    """A time step outside 1..49 must fail loudly, not fall through a gap."""
    with pytest.raises(ValueError, match="partition"):
        splits.build_masks(np.array([1, 50, 30], dtype=np.int16))


def test_each_split_mask_selects_only_its_own_timesteps(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    for split in Split:
        allowed = set(splits.timesteps_for(split))
        assert set(synthetic.timesteps[masks[split]].tolist()) <= allowed


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_scaler_uses_training_rows_only(synthetic):
    """The fixture inflates post-training features 50x.

    A scaler fitted on everything would inherit that scale; one fitted
    correctly matches the training rows exactly.
    """
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)

    fitted = features.fit_scaler(x, masks.train)
    expected = features.Scaler.fit(x[masks.train])

    np.testing.assert_allclose(fitted.mean, expected.mean, rtol=1e-6)
    np.testing.assert_allclose(fitted.scale, expected.scale, rtol=1e-6)

    naive = features.Scaler.fit(x)
    assert not np.allclose(fitted.scale, naive.scale), (
        "fixture should make an all-rows fit visibly different"
    )


def test_timestep_is_not_reachable_as_a_feature(synthetic):
    """The stored feature matrix has 165 columns; the time step is not one."""
    assert synthetic.features.shape[1] == 165
    for variant in features.VARIANTS:
        selected = features.select(synthetic.features, variant)
        for column in range(selected.shape[1]):
            assert not np.array_equal(
                selected[:, column], synthetic.timesteps.astype(selected.dtype)
            ), f"{variant} column {column} reproduces the time step"


def test_training_graph_contains_no_future_nodes(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    _, kept = graph.build_split_graph(synthetic, x, masks.train)

    assert set(synthetic.timesteps[kept].tolist()) <= set(splits.TRAIN_TIMESTEPS)


def test_unknown_nodes_are_neighbours_but_never_supervision_targets(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    train_graph, kept = graph.build_split_graph(synthetic, x, masks.train)

    supervised = train_graph.supervised_mask.numpy()
    assert (synthetic.labels[kept][supervised] != "unknown").all()
    # ...but they are still present in the graph, carrying features.
    assert (synthetic.labels[kept] == "unknown").any()
    assert train_graph.num_nodes == len(kept)


def test_poisoning_future_labels_cannot_change_training_supervision(synthetic):
    """The load-bearing leakage test.

    Every validation and test label is flipped to `illicit`. If any future
    label reached the training graph -- through the mask, through message
    passing, through the target vector -- the training targets would change.
    They must be byte-identical.
    """
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    clean_graph, clean_kept = graph.build_split_graph(synthetic, x, masks.train)

    poisoned_labels = synthetic.labels.copy()
    poisoned_labels[masks.val | masks.test] = "illicit"
    poisoned = type(synthetic)(
        tx_ids=synthetic.tx_ids,
        timesteps=synthetic.timesteps,
        labels=poisoned_labels,
        features=synthetic.features,
        edge_index=synthetic.edge_index,
    )
    dirty_graph, dirty_kept = graph.build_split_graph(poisoned, x, masks.train)

    np.testing.assert_array_equal(clean_kept, dirty_kept)
    np.testing.assert_array_equal(clean_graph.y.numpy(), dirty_graph.y.numpy())
    np.testing.assert_array_equal(
        clean_graph.supervised_mask.numpy(), dirty_graph.supervised_mask.numpy()
    )
    np.testing.assert_array_equal(clean_graph.edge_index.numpy(), dirty_graph.edge_index.numpy())


def test_poisoning_future_features_cannot_change_a_fitted_baseline(synthetic):
    """Same idea, one layer up: the tabular model must not see future rows."""
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    scaler = features.fit_scaler(x, masks.train)
    labelled = synthetic.labels != "unknown"

    def fit_and_score(feature_matrix: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(feature_matrix)
        config = baseline.BaselineConfig(n_estimators=20, early_stopping_rounds=5)
        model = baseline.train(
            scaled[masks.train & labelled],
            (synthetic.labels[masks.train & labelled] == "illicit").astype(np.int8),
            scaled[masks.val & labelled],
            (synthetic.labels[masks.val & labelled] == "illicit").astype(np.int8),
            config,
        )
        return baseline.predict(model, scaled[masks.train])

    before = fit_and_score(synthetic.features)

    poisoned = synthetic.features.copy()
    poisoned[masks.test] = 999.0
    after = fit_and_score(poisoned)

    np.testing.assert_allclose(before, after, rtol=1e-6)


def test_reference_pool_for_similarity_excludes_non_training_timesteps(synthetic):
    """The embedding reference pool is train-range labelled nodes only.

    Phase 4 searches this pool for structurally similar transactions. Letting
    a validation or test node in would mean citing an answer we are not
    supposed to have yet.
    """
    masks = splits.build_masks(synthetic.timesteps)
    pool = masks.train & (synthetic.labels != "unknown")

    assert pool.any()
    assert not (pool & masks.val).any()
    assert not (pool & masks.test).any()
    assert set(synthetic.timesteps[pool].tolist()) <= set(splits.TRAIN_TIMESTEPS)
