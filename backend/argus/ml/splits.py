"""Temporal splits.

The dataset's time steps are the only ordering we have, so every split is a
contiguous range of them and nothing is ever shuffled. The ranges are fixed
constants, not parameters, because a split you can tune is a split you will
eventually tune against the test set.

    train      time steps  1..29   model fitting, scaler fitting,
                                   embedding reference pool
    validation time steps 30..34   early stopping, hyperparameter choice,
                                   alert-budget threshold selection
    test       time steps 35..49   read exactly once, at the end

What is available at each stage
-------------------------------
Fitting (train): features and labels of train-range transactions, plus the
features -- never the labels -- of `unknown` transactions in that range, which
participate as message-passing neighbours only.

Selection (validation): model scores on validation-range transactions and
their labels. Used to pick the decision threshold and to early-stop. Never
used to update model weights.

Reporting (test): model scores on test-range transactions and their labels,
used only to compute the numbers in the final report.

The published Elliptic benchmarks use a two-way 1..34 / 35..49 split, which
leaves nowhere to choose a threshold except the test set. Carving validation
out of the training range costs five time steps of training data and buys an
honest threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

TRAIN_TIMESTEPS = range(1, 30)  # 1..29
VAL_TIMESTEPS = range(30, 35)  # 30..34
TEST_TIMESTEPS = range(35, 50)  # 35..49


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


_RANGES = {
    Split.TRAIN: TRAIN_TIMESTEPS,
    Split.VAL: VAL_TIMESTEPS,
    Split.TEST: TEST_TIMESTEPS,
}


def timesteps_for(split: Split) -> range:
    return _RANGES[split]


def describe() -> dict[str, str]:
    return {split.value: f"{r.start}-{r.stop - 1}" for split, r in _RANGES.items()}


@dataclass(frozen=True)
class SplitMasks:
    """Boolean node masks, one per split, aligned to the dataset row order."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __getitem__(self, split: Split) -> np.ndarray:
        return {Split.TRAIN: self.train, Split.VAL: self.val, Split.TEST: self.test}[split]


def build_masks(timesteps: np.ndarray) -> SplitMasks:
    """Partition nodes by time step.

    Asserts the partition is exact: every node lands in exactly one split.
    A gap or an overlap here would quietly corrupt every downstream number.
    """
    masks = SplitMasks(
        train=np.isin(timesteps, list(TRAIN_TIMESTEPS)),
        val=np.isin(timesteps, list(VAL_TIMESTEPS)),
        test=np.isin(timesteps, list(TEST_TIMESTEPS)),
    )
    stacked = np.vstack([masks.train, masks.val, masks.test])
    counts = stacked.sum(axis=0)
    if not np.all(counts == 1):
        raise ValueError("temporal splits must partition the nodes exactly once each")
    return masks
