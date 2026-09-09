"""Metrics for numerical and categorical variables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance(values, sample: bool = False) -> float:
    """Calculate variance for one numerical variable."""
    series = pd.Series(values).dropna()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("variance requires numerical values.")
    n = len(series)
    if n == 0:
        return float("nan")
    denominator = n - 1 if sample and n > 1 else n
    mean = float(series.sum()) / n
    return float(((series - mean) ** 2).sum() / denominator)


def entropy(values, base: float = 2.0) -> float:
    """Calculate Shannon entropy for a discrete variable."""
    series = pd.Series(values).dropna()
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probabilities = counts / counts.sum()
    logs = np.log(probabilities) / np.log(base)
    return float(-(probabilities * logs).sum())


def auc_score(scores, target, positive_class=None) -> float:
    """Calculate AUC for numerical scores and a binary target.

    The implementation is based on pairwise comparisons between positive and
    negative examples. Ties count as 0.5.
    """
    score_series = pd.Series(scores)
    target_series = pd.Series(target)
    valid = score_series.notna() & target_series.notna()
    score_series = score_series[valid]
    target_series = target_series[valid]
    classes = list(pd.unique(target_series))
    if len(classes) != 2:
        raise ValueError("AUC requires exactly two target classes.")
    positive = positive_class if positive_class is not None else classes[-1]
    positive_scores = score_series[target_series == positive]
    negative_scores = score_series[target_series != positive]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("AUC requires at least one positive and one negative example.")

    wins = 0.0
    total = len(positive_scores) * len(negative_scores)
    for pos in positive_scores:
        wins += float((pos > negative_scores).sum())
        wins += 0.5 * float((pos == negative_scores).sum())
    return wins / total


def attribute_metrics(data: pd.DataFrame, target: str | None = None, positive_class=None) -> pd.DataFrame:
    """Calculate the appropriate metric for each dataset attribute."""
    rows = []
    for column in data.columns:
        if column == target:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            rows.append({"attribute": column, "metric": "variance", "value": variance(data[column])})
            if target is not None:
                rows.append({"attribute": column, "metric": "auc", "value": auc_score(data[column], data[target], positive_class)})
        else:
            rows.append({"attribute": column, "metric": "entropy", "value": entropy(data[column])})
    return pd.DataFrame(rows)
