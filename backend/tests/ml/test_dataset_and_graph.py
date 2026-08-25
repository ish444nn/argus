"""Dataset parsing, id mapping and graph construction."""

from __future__ import annotations

import numpy as np
import pytest

from argus.ml import dataset, features, graph, splits


def test_documented_column_counts_are_self_consistent():
    assert dataset.RAW_COLUMNS == 1 + 1 + dataset.N_FEATURES  # txId + timestep + features
    assert dataset.N_LOCAL_FEATURES + dataset.N_AGGREGATED_FEATURES == dataset.N_FEATURES
    assert dataset.N_AGGREGATED_FEATURES == 72


def test_feature_variant_dimensions():
    assert features.VARIANT_DIMS[features.LOCAL94] == 93
    assert features.VARIANT_DIMS[features.ALL166] == 165


def test_variant_selection_shapes(synthetic):
    for variant, expected in features.VARIANT_DIMS.items():
        assert features.select(synthetic.features, variant).shape == (
            synthetic.n_nodes,
            expected,
        )


def test_local_variant_is_a_prefix_of_the_full_variant(synthetic):
    local = features.select(synthetic.features, features.LOCAL94)
    full = features.select(synthetic.features, features.ALL166)
    np.testing.assert_array_equal(local, full[:, : local.shape[1]])


def test_unknown_variant_is_rejected(synthetic):
    with pytest.raises(ValueError, match="unknown feature variant"):
        features.select(synthetic.features, "all999")


def test_index_of_maps_transaction_ids_to_rows(synthetic):
    for row in (0, 5, synthetic.n_nodes - 1):
        assert synthetic.index_of(int(synthetic.tx_ids[row])) == row
    with pytest.raises(KeyError):
        synthetic.index_of(-1)


def test_edge_index_refers_to_rows_not_transaction_ids(synthetic):
    assert synthetic.edge_index.max() < synthetic.n_nodes
    assert synthetic.edge_index.min() >= 0


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


def test_to_undirected_adds_the_reverse_of_every_edge():
    directed = np.array([[0, 1], [1, 2]], dtype=np.int64)
    undirected = graph.to_undirected(directed)

    pairs = {tuple(pair) for pair in undirected.T.tolist()}
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}


def test_to_undirected_does_not_duplicate_a_reciprocated_edge():
    directed = np.array([[0, 1], [1, 0]], dtype=np.int64)
    undirected = graph.to_undirected(directed)
    assert undirected.shape[1] == 2


def test_no_self_loops_are_introduced(synthetic):
    """SAGEConv has its own root weight; a self-loop would double-count."""
    undirected = graph.to_undirected(synthetic.edge_index)
    assert not np.any(undirected[0] == undirected[1])


def test_induced_subgraph_drops_edges_leaving_the_mask():
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    mask = np.array([True, True, False, False])

    sub, kept = graph.induced_subgraph(edge_index, mask)

    np.testing.assert_array_equal(kept, [0, 1])
    np.testing.assert_array_equal(sub, [[0], [1]])


def test_induced_subgraph_renumbers_nodes_from_zero():
    edge_index = np.array([[3, 4], [4, 5]], dtype=np.int64)
    mask = np.array([False, False, False, True, True, True])

    sub, kept = graph.induced_subgraph(edge_index, mask)

    np.testing.assert_array_equal(kept, [3, 4, 5])
    np.testing.assert_array_equal(sub, [[0, 1], [1, 2]])


def test_split_graph_carries_features_labels_and_supervision(synthetic):
    masks = splits.build_masks(synthetic.timesteps)
    x = features.select(synthetic.features, features.ALL166)
    train_graph, kept = graph.build_split_graph(synthetic, x, masks.train)

    assert train_graph.x.shape == (len(kept), 165)
    assert train_graph.y.shape == (len(kept),)
    assert train_graph.supervised_mask.shape == (len(kept),)
    np.testing.assert_array_equal(train_graph.x.numpy(), x[kept])


def test_synthetic_fixture_has_no_cross_timestep_edges(synthetic):
    assert graph.count_cross_timestep_edges(synthetic) == 0


# --------------------------------------------------------------------------
# Against the real dataset
# --------------------------------------------------------------------------


@pytest.mark.dataset
def test_real_dataset_matches_its_documented_shape(real_dataset):
    assert real_dataset.n_nodes == 203_769
    assert real_dataset.n_edges == 234_355
    assert real_dataset.features.shape == (203_769, 165)
    assert set(np.unique(real_dataset.labels)) == {"illicit", "licit", "unknown"}


@pytest.mark.dataset
def test_real_label_counts(real_dataset):
    counts = dict(zip(*np.unique(real_dataset.labels, return_counts=True), strict=True))
    assert counts["illicit"] == 4_545
    assert counts["licit"] == 42_019
    assert counts["unknown"] == 157_205


@pytest.mark.dataset
def test_elliptic_has_zero_cross_timestep_edges(real_dataset):
    """The fact the whole leakage argument rests on.

    If a future dataset revision introduced cross-time-step edges, training
    on the induced training subgraph would start seeing future nodes and this
    test is the thing that catches it.
    """
    assert graph.count_cross_timestep_edges(real_dataset) == 0


@pytest.mark.dataset
def test_real_transaction_ids_are_unique(real_dataset):
    assert len(np.unique(real_dataset.tx_ids)) == real_dataset.n_nodes


@pytest.mark.dataset
def test_parsing_is_deterministic(real_dataset):
    """Two loads of the same files must agree exactly."""
    from argus.ml.cli import PROCESSED_DIR, RAW_DIR

    again = dataset.load_cached(RAW_DIR, PROCESSED_DIR)
    np.testing.assert_array_equal(again.tx_ids, real_dataset.tx_ids)
    np.testing.assert_array_equal(again.labels, real_dataset.labels)
    np.testing.assert_array_equal(again.features, real_dataset.features)
    np.testing.assert_array_equal(again.edge_index, real_dataset.edge_index)
