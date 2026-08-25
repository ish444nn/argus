"""Training orchestration and the pre-registered model comparison.

Order of operations matters here and is enforced by the structure of
`run_all`: every model is fitted, every threshold is chosen, and every
hyperparameter is frozen using train and validation data only. The test split
is touched once, at the very end, by `_evaluate_test`.

The promotion rule was written down before any test number existed:

    GraphSAGE replaces XGBoost as the primary scorer if and only if its test
    recall at the 1% alert budget exceeds the best XGBoost variant's by at
    least 2 absolute percentage points.

Whichever way it lands is reported. A baseline win is a result, not a
failure -- the scoring pipeline reads whichever model the manifest names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from argus.db.enums import Label
from argus.ml import baseline, evaluate, features, gnn, graph, registry, splits
from argus.ml.dataset import EllipticData
from argus.ml.evaluate import ALERT_BUDGET, PROMOTION_MARGIN, SplitReport
from argus.ml.splits import Split

log = logging.getLogger(__name__)

EXPERIMENT_NAME = "argus-risk-models"


@dataclass
class ModelResult:
    name: str
    model_type: str
    feature_variant: str
    threshold: float
    val: SplitReport
    test: SplitReport
    mlflow_run_id: str | None = None

    @property
    def headline_recall(self) -> float:
        """Test recall at the actual 1% alert budget -- the promotion metric."""
        return self.test.at_budget.recall


def _labelled(labels: np.ndarray) -> np.ndarray:
    return labels != Label.UNKNOWN.value


def _targets(labels: np.ndarray) -> np.ndarray:
    return (labels == Label.ILLICIT.value).astype(np.int8)


def _log_run(
    name: str,
    params: dict,
    val: SplitReport,
    test: SplitReport,
) -> str | None:
    """Record one experiment. MLflow failures must not lose a training run."""
    try:
        import mlflow

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            for split_name, report in (("val", val), ("test", test)):
                for mode, metrics in (
                    ("budget", report.at_budget),
                    ("fixed", report.at_fixed_threshold),
                ):
                    for key, value in metrics.to_dict().items():
                        if isinstance(value, int | float):
                            mlflow.log_metric(f"{split_name}_{mode}_{key}", float(value))
            return run.info.run_id
    except Exception as exc:  # pragma: no cover - tracking is not critical path
        log.warning("MLflow logging failed for %s: %s", name, exc)
        return None


def train_baseline(
    data: EllipticData,
    masks: splits.SplitMasks,
    variant: str,
) -> tuple[ModelResult, object, features.Scaler]:
    """Fit one XGBoost variant and evaluate it on validation, then test."""
    x_all = features.select(data.features, variant)
    scaler = features.fit_scaler(x_all, masks.train)
    x_scaled = scaler.transform(x_all)

    fit_mask = masks.train & _labelled(data.labels)
    val_fit_mask = masks.val & _labelled(data.labels)

    config = baseline.BaselineConfig()
    model = baseline.train(
        x_scaled[fit_mask],
        _targets(data.labels[fit_mask]),
        x_scaled[val_fit_mask],
        _targets(data.labels[val_fit_mask]),
        config,
    )

    # Threshold comes from validation scores over ALL validation transactions,
    # labelled or not -- the alert budget constrains analyst workload.
    val_scores = baseline.predict(model, x_scaled[masks.val])
    threshold = evaluate.threshold_for_budget(val_scores, ALERT_BUDGET)
    val = evaluate.report_split(val_scores, data.labels[masks.val], threshold)

    test_scores = baseline.predict(model, x_scaled[masks.test])
    test = evaluate.report_split(test_scores, data.labels[masks.test], threshold)

    name = f"xgb-{variant}"
    log.info("%s  val @budget: %s", name, val.at_budget.summary())
    log.info("%s test @budget: %s", name, test.at_budget.summary())
    log.info("%s test @fixed : %s", name, test.at_fixed_threshold.summary())

    run_id = _log_run(
        name,
        {
            "model_type": "xgboost",
            "feature_variant": variant,
            "n_features": int(x_all.shape[1]),
            "alert_budget": ALERT_BUDGET,
            "splits": splits.describe(),
            **config.to_dict(),
        },
        val,
        test,
    )
    result = ModelResult(
        name=name,
        model_type="xgboost",
        feature_variant=variant,
        threshold=threshold,
        val=val,
        test=test,
        mlflow_run_id=run_id,
    )
    return result, model, scaler


def train_graphsage(
    data: EllipticData,
    masks: splits.SplitMasks,
    variant: str = features.ALL166,
) -> tuple[ModelResult, gnn.GraphSAGE, features.Scaler, dict]:
    """Fit GraphSAGE on the training subgraph.

    Each split gets its own induced subgraph. Because Elliptic has zero
    cross-time-step edges, that subgraph is simultaneously (a) free of any
    future node and (b) the complete neighbourhood its nodes actually have --
    nothing is withheld at inference.
    """
    x_all = features.select(data.features, variant)
    scaler = features.fit_scaler(x_all, masks.train)
    x_scaled = scaler.transform(x_all)

    train_graph, _ = graph.build_split_graph(data, x_scaled, masks.train)
    val_graph, val_nodes = graph.build_split_graph(data, x_scaled, masks.val)
    test_graph, test_nodes = graph.build_split_graph(data, x_scaled, masks.test)

    config = gnn.TrainingConfig()
    model, history = gnn.train(train_graph, val_graph, config)

    val_scores, _ = gnn.predict(model, val_graph)
    threshold = evaluate.threshold_for_budget(val_scores, ALERT_BUDGET)
    val = evaluate.report_split(val_scores, data.labels[val_nodes], threshold)

    test_scores, _ = gnn.predict(model, test_graph)
    test = evaluate.report_split(test_scores, data.labels[test_nodes], threshold)

    log.info("graphsage  val @budget: %s", val.at_budget.summary())
    log.info("graphsage test @budget: %s", test.at_budget.summary())
    log.info("graphsage test @fixed : %s", test.at_fixed_threshold.summary())

    run_id = _log_run(
        "graphsage",
        {
            "model_type": "graphsage",
            "feature_variant": variant,
            "n_features": int(x_all.shape[1]),
            "alert_budget": ALERT_BUDGET,
            "splits": splits.describe(),
            **config.to_dict(),
        },
        val,
        test,
    )
    result = ModelResult(
        name="graphsage",
        model_type="graphsage",
        feature_variant=variant,
        threshold=threshold,
        val=val,
        test=test,
        mlflow_run_id=run_id,
    )
    return result, model, scaler, history


def embed_all(
    data: EllipticData,
    masks: splits.SplitMasks,
    model: gnn.GraphSAGE,
    scaler: features.Scaler,
    variant: str = features.ALL166,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed every transaction, one split's subgraph at a time.

    Splits are embedded separately so a node's representation is built only
    from neighbours in its own time step -- the same context the model was
    trained under, and the same context available operationally.

    Returns (tx_ids, embeddings, graph_scores) covering all nodes.
    """
    x_scaled = scaler.transform(features.select(data.features, variant))

    tx_ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for split in Split:
        split_graph, nodes = graph.build_split_graph(data, x_scaled, masks[split])
        split_scores, split_embeddings = gnn.predict(model, split_graph)
        tx_ids.append(data.tx_ids[nodes])
        vectors.append(split_embeddings)
        scores.append(split_scores)

    return np.concatenate(tx_ids), np.vstack(vectors), np.concatenate(scores)


def decide_primary(baselines: list[ModelResult], graphsage: ModelResult) -> tuple[str, str]:
    """Apply the pre-registered promotion rule. Returns (winner, rationale)."""
    best = max(baselines, key=lambda r: r.headline_recall)
    lift = graphsage.headline_recall - best.headline_recall
    if lift >= PROMOTION_MARGIN:
        return graphsage.name, (
            f"GraphSAGE recall@{ALERT_BUDGET:.0%} exceeds {best.name} by "
            f"{lift:+.3f}, clearing the pre-registered {PROMOTION_MARGIN:+.2f} margin."
        )
    return best.name, (
        f"GraphSAGE recall@{ALERT_BUDGET:.0%} differs from {best.name} by "
        f"{lift:+.3f}, short of the pre-registered {PROMOTION_MARGIN:+.2f} margin; "
        f"{best.name} remains the primary scorer."
    )


def run_all(
    data: EllipticData,
    models_root: Path | None = None,
    tracking_uri: str | None = None,
) -> dict:
    """Train every model, apply the promotion rule, export artifacts."""
    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except ImportError:
        # `mlflow` lives in the optional `train` extra so the worker image can
        # skip it. Training without it still works; it just goes unrecorded.
        log.warning("mlflow not installed -- experiments will not be tracked")

    masks = splits.build_masks(data.timesteps)
    split_description = splits.describe()

    baseline_results: list[ModelResult] = []
    for variant in features.VARIANTS:
        result, model, scaler = train_baseline(data, masks, variant)
        baseline_results.append(result)
        registry.save_xgboost(
            model,
            registry.ModelMetadata(
                name=result.name,
                model_type="xgboost",
                feature_variant=variant,
                version=f"{result.name}@{(result.mlflow_run_id or 'local')[:8]}",
                n_features=features.VARIANT_DIMS[variant],
                splits=split_description,
                threshold=result.threshold,
                alert_budget=ALERT_BUDGET,
                hyperparameters=baseline.BaselineConfig().to_dict(),
                metrics={
                    "val": result.val.to_dict(),
                    "test": result.test.to_dict(),
                },
                scaler=scaler.to_dict(),
                mlflow_run_id=result.mlflow_run_id,
            ),
            models_root,
        )

    sage_result, sage_model, sage_scaler, history = train_graphsage(data, masks)
    registry.save_torch(
        sage_model.state_dict(),
        registry.ModelMetadata(
            name="graphsage",
            model_type="graphsage",
            feature_variant=sage_result.feature_variant,
            version=f"graphsage@{(sage_result.mlflow_run_id or 'local')[:8]}",
            n_features=features.VARIANT_DIMS[sage_result.feature_variant],
            splits=split_description,
            # GraphSAGE does not gate the queue, so it has no alert threshold.
            threshold=None,
            alert_budget=ALERT_BUDGET,
            hyperparameters=gnn.TrainingConfig().to_dict(),
            metrics={
                "val": sage_result.val.to_dict(),
                "test": sage_result.test.to_dict(),
                "best_epoch": history["best_epoch"],
            },
            scaler=sage_scaler.to_dict(),
            mlflow_run_id=sage_result.mlflow_run_id,
        ),
        models_root,
    )

    winner, rationale = decide_primary(baseline_results, sage_result)
    log.info("primary scorer: %s -- %s", winner, rationale)

    return {
        "baselines": baseline_results,
        "graphsage": sage_result,
        "primary": winner,
        "rationale": rationale,
        "graphsage_model": sage_model,
        "graphsage_scaler": sage_scaler,
        "masks": masks,
    }
