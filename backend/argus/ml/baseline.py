"""XGBoost: the primary risk scorer.

This is the model that decides the analyst queue. It sees one transaction at
a time -- no graph, no neighbours -- which is exactly what makes it the right
thing for GraphSAGE to be measured against.

Class imbalance is handled with `scale_pos_weight` (the licit-to-illicit
ratio) rather than resampling. Resampling would change the effective data
distribution and, for the graph model we compare against, would mean altering
the graph itself; keeping both models on plain class weighting keeps the
comparison like-for-like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

log = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    early_stopping_rounds: int = 30
    seed: int = 42

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def scale_pos_weight(y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    return n_neg / max(n_pos, 1.0)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: BaselineConfig,
) -> xgb.XGBClassifier:
    """Fit on labelled training rows, early-stop on labelled validation rows.

    Both arrays must already exclude `unknown` nodes: they have no target, so
    they cannot participate in a supervised fit.
    """
    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        scale_pos_weight=scale_pos_weight(y_train),
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=config.seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    log.info(
        "xgboost fitted: best_iteration=%s best_score=%.4f",
        model.best_iteration,
        model.best_score,
    )
    return model


def predict(model: xgb.XGBClassifier, x: np.ndarray) -> np.ndarray:
    """Illicit probability for every row, `unknown`-labelled ones included.

    Scoring unlabelled transactions is not a leak -- their features are
    observable and an operational queue must rank them too. Only *metrics*
    exclude them.
    """
    return model.predict_proba(x)[:, 1]
