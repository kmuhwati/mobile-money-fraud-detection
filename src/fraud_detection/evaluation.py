"""Metrics that match how a fraud team actually operates.

ROC-AUC is reported but not trusted: at a 0.3% base rate it is dominated by
easy negatives. The decision-relevant numbers are precision/recall at the
alert budget the review team can handle, and the share of *money* caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def at_budget(y: np.ndarray, score: np.ndarray, amount: np.ndarray, budget: float) -> dict:
    """Alert on the top ``budget`` share of transactions by score."""
    k = max(int(round(budget * len(y))), 1)
    top = np.argsort(-score, kind="stable")[:k]
    tp = y[top].sum()
    return {
        "alerts": k,
        "precision": float(tp / k),
        "recall": float(tp / max(y.sum(), 1)),
        "value_recall": float(amount[top][y[top] == 1].sum() / max(amount[y == 1].sum(), 1)),
    }


def at_threshold(y, score, amount, threshold: float) -> dict:
    alert = score >= threshold
    tp = (alert & (y == 1)).sum()
    return {
        "alert_rate": float(alert.mean()),
        "precision": float(tp / max(alert.sum(), 1)),
        "recall": float(tp / max(y.sum(), 1)),
        "value_recall": float(amount[alert & (y == 1)].sum() / max(amount[y == 1].sum(), 1)),
    }


def summary(y, score, amount, budget: float) -> dict:
    return {
        "auprc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        **{f"{k}@budget": v for k, v in at_budget(y, score, amount, budget).items()},
    }


def bootstrap_ci(y, score, amount, budget, n_boot=200, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    aps, recs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        aps.append(average_precision_score(y[i], score[i]))
        recs.append(at_budget(y[i], score[i], amount[i], budget)["recall"])
    return {"auprc_lo": float(np.percentile(aps, 2.5)), "auprc_hi": float(np.percentile(aps, 97.5)),
            "recall_lo": float(np.percentile(recs, 2.5)), "recall_hi": float(np.percentile(recs, 97.5))}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins from the reference sample.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [edges[0] - 1, edges[-1] + 1]]))
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def reliability(y, prob, bins=10) -> pd.DataFrame:
    q = pd.qcut(prob, bins, duplicates="drop")
    return pd.DataFrame({"pred": prob, "obs": y, "bin": q}).groupby("bin", observed=True).agg(
        predicted=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size")).reset_index(drop=True)
