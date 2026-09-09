"""Discretization algorithms for numerical variables and datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_bins(bins: int) -> None:
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")


def discretize_equal_width(values, bins: int = 5, labels=None) -> pd.Series:
    """Discretize one numerical variable using intervals with equal width.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values.
    bins:
        Number of intervals to create.
    labels:
        Optional labels for the resulting intervals.

    Returns
    -------
    pandas.Series
        Discrete interval labels. Missing values remain missing.
    """
    _validate_bins(bins)
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("equal-width discretization requires numerical values.")
    non_missing = series.dropna()
    if non_missing.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    if non_missing.min() == non_missing.max():
        return pd.Series(np.where(series.isna(), pd.NA, "bin_1"), index=series.index)
    labels = labels or [f"bin_{i + 1}" for i in range(bins)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def discretize_equal_frequency(values, bins: int = 5, labels=None) -> pd.Series:
    """Discretize one numerical variable so each bin has similar frequency."""
    _validate_bins(bins)
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("equal-frequency discretization requires numerical values.")
    non_missing = series.dropna()
    if non_missing.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    if non_missing.nunique() == 1:
        return pd.Series(np.where(series.isna(), pd.NA, "bin_1"), index=series.index)
    labels = labels or [f"bin_{i + 1}" for i in range(bins)]
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=min(bins, non_missing.nunique()), labels=labels[: min(bins, non_missing.nunique())])


def discretize_dataset_equal_width(data: pd.DataFrame, bins: int = 5, columns=None) -> pd.DataFrame:
    """Apply equal-width discretization to numerical columns in a dataset."""
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = discretize_equal_width(result[column], bins=bins)
    return result


def discretize_dataset_equal_frequency(data: pd.DataFrame, bins: int = 5, columns=None) -> pd.DataFrame:
    """Apply equal-frequency discretization to numerical columns in a dataset."""
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = discretize_equal_frequency(result[column], bins=bins)
    return result
