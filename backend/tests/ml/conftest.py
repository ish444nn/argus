"""Synthetic dataset fixtures.

Almost every ML test runs against a small generated graph rather than the real
Elliptic files, so the suite stays fast and works on a fresh clone that has
not downloaded the 690 MB feature file. Tests that genuinely need the real
data are marked `dataset` and skip themselves when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.ml.dataset import N_FEATURES, EllipticData

RNG_SEED = 7


def make_dataset(
    n_per_timestep: int = 6,
    n_features: int = N_FEATURES,
    seed: int = RNG_SEED,
) -> EllipticData:
    """A miniature Elliptic-shaped graph spanning all 49 time steps.

    Mirrors the two properties the pipeline depends on: edges never cross a
    time step, and roughly three quarters of nodes are `unknown`.
    """
    rng = np.random.default_rng(seed)
    timesteps = np.repeat(np.arange(1, 50, dtype=np.int16), n_per_timestep)
    n_nodes = len(timesteps)
    tx_ids = np.arange(1000, 1000 + n_nodes, dtype=np.int64)

    # Later time steps get a deliberately different feature scale so a scaler
    # fitted on the wrong rows is detectable.
    features = rng.normal(size=(n_nodes, n_features)).astype(np.float32)
    features[timesteps >= 30] *= 50.0

    labels = np.full(n_nodes, "unknown", dtype="<U8")
    for step in range(1, 50):
        idx = np.nonzero(timesteps == step)[0]
        labels[idx[0]] = "illicit"
        labels[idx[1]] = "licit"

    # Chain the nodes within each time step, so every edge is intra-step.
    edges = []
    for step in range(1, 50):
        idx = np.nonzero(timesteps == step)[0]
        for a, b in zip(idx[:-1], idx[1:], strict=True):
            edges.append((a, b))
    edge_index = np.asarray(edges, dtype=np.int64).T

    return EllipticData(
        tx_ids=tx_ids,
        timesteps=timesteps,
        labels=labels,
        features=features,
        edge_index=edge_index,
    )


@pytest.fixture
def synthetic() -> EllipticData:
    return make_dataset()


@pytest.fixture
def real_dataset():
    """The actual Elliptic data, or a skip if it has not been downloaded."""
    from argus.ml.cli import PROCESSED_DIR, RAW_DIR
    from argus.ml.dataset import cache_path, load_cached

    if (
        not cache_path(PROCESSED_DIR).exists()
        and not (RAW_DIR / "elliptic_txs_features.csv").exists()
    ):
        pytest.skip("Elliptic dataset not downloaded (`python -m argus.ml.cli download`)")
    return load_cached(RAW_DIR, PROCESSED_DIR)
