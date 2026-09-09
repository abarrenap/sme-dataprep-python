"""Metrics for numerical and categorical variables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def variance(values, sample: bool = False) -> float:
    """Calculate the variance of one numerical variable.

    Variance measures how far the values of a numerical variable are spread
    around their mean. A variance close to zero means that most observations are
    very similar, while a larger variance means the variable changes more across
    the dataset.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values. Missing values are
        removed before the calculation.
    sample:
        If `False`, calculate population variance by dividing by `n`. If `True`,
        calculate sample variance by dividing by `n - 1` when at least two
        values are available.

    Returns
    -------
    float
        Variance of the non-missing values. Returns `nan` when there are no
        non-missing values.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    """
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
    """Calculate Shannon entropy for one discrete variable.

    Entropy measures the uncertainty or diversity of a categorical variable. If
    all observations belong to the same category, entropy is zero. If categories
    are evenly distributed, entropy is higher.

    Parameters
    ----------
    values:
        Sequence or pandas Series with categorical or already-discretized
        values. Missing values are ignored.
    base:
        Logarithm base used in the entropy formula. The default `2` expresses
        entropy in bits.

    Returns
    -------
    float
        Shannon entropy calculated as `-sum(p * log(p))`, where `p` is each
        category probability. Returns `0.0` for an empty variable.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        return 0.0
    counts = series.value_counts()
    probabilities = counts / counts.sum()
    logs = np.log(probabilities) / np.log(base)
    return float(-(probabilities * logs).sum())


def auc_score(scores, target, positive_class=None) -> float:
    """Calculate AUC for numerical scores and a binary target.

    AUC, or Area Under the ROC Curve, evaluates how well a numerical attribute
    orders the two classes of a binary target. In this educational
    implementation, every positive example is compared with every negative
    example. A comparison counts as `1` when the positive example has a higher
    score, `0.5` when both scores are tied, and `0` otherwise. The AUC is the
    average result over all positive-negative pairs.

    Parameters
    ----------
    scores:
        Numerical values of the attribute being evaluated.
    target:
        Binary class values with the same length as `scores`.
    positive_class:
        Class value considered positive. If omitted, the second distinct class
        found in `target` is used.

    Returns
    -------
    float
        AUC value between 0 and 1. A value around 0.5 means that the attribute is
        not better than random ordering. A value closer to 1 means higher scores
        are more associated with the positive class.

    Raises
    ------
    ValueError
        If the target does not contain exactly two classes, or if one of the two
        classes has no valid observations.
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
    """Calculate suitable metrics for every attribute in a dataset.

    The function inspects the type of each column and applies the required
    coursework metric. Numerical columns receive variance. If a binary target is
    supplied, numerical columns also receive AUC against that target. Non-
    numerical columns receive entropy.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    target:
        Optional name of the binary target column used for AUC. The target
        column itself is skipped as an attribute.
    positive_class:
        Optional class value treated as the positive class when calculating AUC.

    Returns
    -------
    pandas.DataFrame
        Long-format table with three columns: `attribute`, `metric`, and
        `value`. Each row contains one calculated metric for one attribute.

    Raises
    ------
    ValueError
        If AUC is requested and the target is not binary.
    """
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
