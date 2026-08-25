"""Feature variants and scaling.

Two variants are trained so the model comparison is honest about what the
graph model is actually being asked to beat:

`local94`  the transaction's own attributes only (93 columns after dropping
           the time step). This is a genuine tabular baseline: it knows
           nothing about the network.
`all166`   those plus the dataset's 72 aggregated features (165 columns).
           Those aggregates are one-hop neighbour summaries computed by the
           dataset authors, so a model trained on them **already has one hop
           of graph information**.

`all166` is the bar GraphSAGE has to clear. Reporting `local94` alongside it
shows how much of the graph signal the hand-crafted aggregates already
capture, which is the interesting part of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argus.ml.dataset import N_FEATURES, N_LOCAL_FEATURES

FeatureVariant = str
LOCAL94: FeatureVariant = "local94"
ALL166: FeatureVariant = "all166"
VARIANTS: tuple[FeatureVariant, ...] = (LOCAL94, ALL166)

# Column counts actually fed to a model. The names keep the dataset's own
# terminology; both exclude the time step (see argus.ml.dataset).
VARIANT_DIMS = {LOCAL94: N_LOCAL_FEATURES, ALL166: N_FEATURES}


def select(features: np.ndarray, variant: FeatureVariant) -> np.ndarray:
    if variant == LOCAL94:
        return features[:, :N_LOCAL_FEATURES]
    if variant == ALL166:
        return features
    raise ValueError(f"unknown feature variant {variant!r}")


@dataclass
class Scaler:
    """Standardisation fitted on training rows only.

    Deliberately not scikit-learn's StandardScaler: this is six lines, it
    serialises to plain arrays we can store next to the model, and having it
    here makes the fit-on-train-only rule visible at the call site.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Constant columns would divide by zero; leaving them at 1.0 maps them
        # to a constant 0 after centring, which is what we want.
        scale = np.where(std > 1e-8, std, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, blob: dict[str, list[float]]) -> Scaler:
        return cls(
            mean=np.asarray(blob["mean"], dtype=np.float32),
            scale=np.asarray(blob["scale"], dtype=np.float32),
        )


def fit_scaler(features: np.ndarray, train_mask: np.ndarray) -> Scaler:
    """Fit on training-range rows only.

    Fitting on the whole dataset would leak the distribution of future time
    steps into training -- a small leak, but a real one, and the kind that is
    invisible once it is in the pipeline.
    """
    if not train_mask.any():
        raise ValueError("cannot fit a scaler on an empty training mask")
    return Scaler.fit(features[train_mask])
