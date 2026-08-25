"""Metrics for the illicit class.

Accuracy is never reported as evidence that a model works. Illicit
transactions are about 10% of labelled nodes and 2% of all nodes, so a model
that predicts "licit" for everything scores 98% accuracy while catching
nothing.

The alert budget
----------------
An analyst team can review a fixed number of cases per batch, so the
operational question is "of the transactions we can afford to look at, how
many illicit ones did we find?".

Each model is therefore reported two ways, because they answer different
questions and the gap between them is itself a finding:

`at_budget`  the split's own scores are ranked and exactly the top 1% are
             alerted. This is what "recall at a fixed 1% alert budget"
             actually means, and it is what a live queue does: rank this
             batch, take the top 1%. Computing it uses only scores -- never
             labels -- so it is not a form of test-set fitting. Selecting a
             fixed *count* rather than applying a threshold matters: boosted
             probabilities saturate near 1.0, and a threshold admits every
             row tied at the cut, which handed one variant 53% more alerts
             than another and made the comparison meaningless.

`at_fixed_threshold`  the numeric threshold chosen on validation, applied
             verbatim. This is what a deployed model does if it is never
             recalibrated, and on Elliptic it degrades badly: score
             distributions shift in the later time steps, so a frozen
             threshold alerts far below budget and recall collapses.

The budget is computed over *all* scored transactions, including
`unknown`-labelled ones, because an analyst's queue does not know which
transactions happen to carry ground truth. Recall and precision are then
measured over labelled transactions only, since those are the only ones whose
correctness we can judge.

Note the ceiling this imposes: the test split has 67,504 transactions and
1,083 labelled illicit ones, so a 1% budget buys 676 alerts and caps
achievable recall at about 0.62. Recall figures should be read against that
ceiling, not against 1.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from argus.db.enums import Label

# Pre-registered before any test-set evaluation. See docs/modeling.md.
ALERT_BUDGET = 0.01
PROMOTION_MARGIN = 0.02  # absolute recall points GraphSAGE must add to be promoted


@dataclass(frozen=True)
class Metrics:
    """All figures refer to the illicit class."""

    n_scored: int
    n_labelled: int
    n_illicit: int
    threshold: float
    alert_rate: float
    precision: float
    recall: float
    f1: float
    auprc: float
    roc_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"precision={self.precision:.3f} recall={self.recall:.3f} "
            f"auprc={self.auprc:.3f} alert_rate={self.alert_rate:.4f} "
            f"threshold={self.threshold:.4f}"
        )


def illicit_mask(labels: np.ndarray) -> np.ndarray:
    return labels == Label.ILLICIT.value


def labelled_mask(labels: np.ndarray) -> np.ndarray:
    """Rows with ground truth. `unknown` nodes are excluded from every metric."""
    return labels != Label.UNKNOWN.value


def threshold_for_budget(scores: np.ndarray, budget: float = ALERT_BUDGET) -> float:
    """Score of the k-th highest transaction, where k = budget x n.

    Rank-based rather than `np.quantile`, because gradient-boosted
    probabilities saturate near 1.0 and produce heavy ties at the top; an
    interpolated quantile lands between tied values and silently alerts two or
    three times the intended budget.

    `scores` must be every score the model produced for the split, labelled or
    not -- the budget constrains analyst workload, not ground truth.
    """
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    if scores.size == 0:
        raise ValueError("cannot pick a threshold from an empty score array")
    n = scores.size
    k = max(1, min(n, int(np.ceil(budget * n))))
    return float(np.partition(scores, n - k)[n - k])


def alerts_at_budget(scores: np.ndarray, budget: float = ALERT_BUDGET) -> np.ndarray:
    """Boolean mask selecting exactly k = budget x n transactions.

    Threshold comparison alone is not good enough for the comparison metric:
    boosted probabilities saturate, so `scores >= threshold` can admit half
    again as many alerts as the budget allows when many rows tie at the cut.
    One model then gets a bigger queue than another and the recall figures
    stop being like-for-like. Ranking and taking exactly k keeps every model
    on the same budget; ties are broken by index, deterministically.
    """
    n = scores.size
    k = max(1, min(n, int(np.ceil(budget * n))))
    alerted = np.zeros(n, dtype=bool)
    # -scores so argsort ascending gives descending score; 'stable' makes the
    # tie order reproducible across runs.
    alerted[np.argsort(-scores, kind="stable")[:k]] = True
    return alerted


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    alerted: np.ndarray | None = None,
) -> Metrics:
    """Score a split.

    Alerts default to `scores >= threshold`. Pass `alerted` to score an
    explicit selection instead (see `alerts_at_budget`); `threshold` is then
    reported for reference only.

    `scores` and `labels` cover every node in the split; `unknown` nodes count
    towards the alert rate but not towards precision, recall or AUPRC.
    """
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be the same length")
    if alerted is None:
        alerted = scores >= threshold
    elif alerted.shape != scores.shape:
        raise ValueError("alerted mask must match the score array")

    known = labelled_mask(labels)
    y_true = illicit_mask(labels[known]).astype(np.int8)
    y_score = scores[known]
    y_pred = alerted[known].astype(np.int8)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # A split with only one class present cannot have a meaningful AUPRC.
    if y_true.min() == y_true.max():
        auprc = float("nan")
        roc_auc = float("nan")
    else:
        auprc = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))

    return Metrics(
        n_scored=int(scores.size),
        n_labelled=int(known.sum()),
        n_illicit=int(y_true.sum()),
        threshold=float(threshold),
        alert_rate=float(alerted.mean()),
        precision=precision,
        recall=recall,
        f1=f1,
        auprc=auprc,
        roc_auc=roc_auc,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


@dataclass(frozen=True)
class SplitReport:
    """One model on one split, measured both ways.

    `at_budget` is the headline number used by the promotion rule.
    `at_fixed_threshold` shows what the same model does when the validation
    threshold is applied without recalibration.
    """

    at_budget: Metrics
    at_fixed_threshold: Metrics

    def to_dict(self) -> dict:
        return {
            "at_budget": self.at_budget.to_dict(),
            "at_fixed_threshold": self.at_fixed_threshold.to_dict(),
        }


def report_split(
    scores: np.ndarray,
    labels: np.ndarray,
    fixed_threshold: float,
    budget: float = ALERT_BUDGET,
) -> SplitReport:
    """Evaluate a split at its own budget and at a frozen threshold."""
    return SplitReport(
        at_budget=evaluate(
            scores,
            labels,
            threshold_for_budget(scores, budget),
            alerted=alerts_at_budget(scores, budget),
        ),
        at_fixed_threshold=evaluate(scores, labels, fixed_threshold),
    )
