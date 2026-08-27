"""Loading the primary scorer and scoring a batch.

Evaluation settled which model that is: `xgb-all166`, because GraphSAGE missed
the pre-registered promotion bar. Nothing here knows that, though -- it reads
the model named by `PRIMARY_MODEL`, loads its manifest, and applies the scaler
the manifest carries. Promoting a different model is a one-line change plus a
retrain, not a rewrite.

This module imports xgboost, so it runs in the worker and never in the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from argus.ml import features, registry

log = logging.getLogger(__name__)

PRIMARY_MODEL = "xgb-all166"


@dataclass
class LoadedModel:
    booster: object
    metadata: registry.ModelMetadata
    scaler: features.Scaler

    @property
    def version(self) -> str:
        return self.metadata.version


def load_primary(name: str = PRIMARY_MODEL, root: Path | None = None) -> LoadedModel:
    """Load the scorer named by its manifest.

    The scaler travels with the model rather than being refitted, so a score
    produced today uses exactly the statistics the model was trained under.
    """
    import xgboost as xgb

    metadata = registry.load_metadata(name, root)
    if metadata.scaler is None:
        raise ValueError(f"model {name} has no scaler in its manifest")

    booster = xgb.XGBClassifier()
    booster.load_model(str(registry.model_dir(name, root) / "model.ubj"))

    log.info("loaded %s (version %s)", name, metadata.version)
    return LoadedModel(
        booster=booster,
        metadata=metadata,
        scaler=features.Scaler.from_dict(metadata.scaler),
    )


def load_batch_features(session: Session, timestep: int) -> tuple[np.ndarray, np.ndarray]:
    """Transaction ids and their raw feature matrix for one time step.

    Ordered by tx_id so a replay of the same batch always processes rows in
    the same sequence, which is what makes tie-breaking reproducible.
    """
    rows = session.execute(
        text("""
        SELECT tx_id, features
        FROM transactions
        WHERE timestep = :timestep AND features IS NOT NULL
        ORDER BY tx_id
        """),
        {"timestep": timestep},
    ).all()

    if not rows:
        raise ValueError(
            f"no transactions with features at time step {timestep} "
            "-- has `argus.ml.cli ingest` been run?"
        )

    tx_ids = np.fromiter((row.tx_id for row in rows), dtype=np.int64, count=len(rows))
    matrix = np.asarray([row.features for row in rows], dtype=np.float32)
    return tx_ids, matrix


def score_batch(model: LoadedModel, raw_features: np.ndarray) -> np.ndarray:
    """Illicit probability per row.

    Unlabelled transactions are scored too: their features are observable and
    an operational queue has to rank them alongside everything else.
    """
    selected = features.select(raw_features, model.metadata.feature_variant)
    if selected.shape[1] != model.metadata.n_features:
        raise ValueError(
            f"feature width {selected.shape[1]} does not match the "
            f"{model.metadata.n_features} the model expects"
        )
    scaled = model.scaler.transform(selected)
    return model.booster.predict_proba(scaled)[:, 1]


def select_alerts(scores: np.ndarray, budget: float) -> np.ndarray:
    """Indices of the batch's alerts, highest score first.

    Selects exactly ceil(budget x n) transactions by rank. Evaluation established
    why this cannot be a stored probability threshold: score distributions
    shift across later time steps, and a threshold frozen on validation alerted
    0.12% of the test range instead of 1%, collapsing recall from 0.374 to
    0.063. Ranking the batch in front of us keeps the budget honest.

    Ties break by position, and `load_batch_features` orders by tx_id, so the
    selection is reproducible.
    """
    if not 0 < budget <= 1:
        raise ValueError(f"alert budget must be in (0, 1], got {budget}")
    n = scores.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    k = max(1, min(n, int(np.ceil(budget * n))))
    return np.argsort(-scores, kind="stable")[:k]
