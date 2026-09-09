"""Pairwise association measures for dataset attributes."""

from __future__ import annotations

import math

import pandas as pd

from .metrics import entropy


def correlation_pair(x, y) -> float:
    """Calculate Pearson correlation for two numerical variables."""
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    valid = x_series.notna() & y_series.notna()
    x_series = x_series[valid]
    y_series = y_series[valid]
    if len(x_series) == 0:
        return float("nan")
    x_mean = x_series.mean()
    y_mean = y_series.mean()
    numerator = ((x_series - x_mean) * (y_series - y_mean)).sum()
    denominator = math.sqrt(((x_series - x_mean) ** 2).sum() * ((y_series - y_mean) ** 2).sum())
    return float(numerator / denominator) if denominator != 0 else 0.0


def mutual_information(x, y, base: float = 2.0) -> float:
    """Calculate mutual information for two categorical variables."""
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    valid = x_series.notna() & y_series.notna()
    x_series = x_series[valid]
    y_series = y_series[valid]
    if len(x_series) == 0:
        return 0.0
    joint = pd.Series(list(zip(x_series, y_series)))
    return entropy(x_series, base=base) + entropy(y_series, base=base) - entropy(joint, base=base)


def association_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise correlations or mutual information by variable type."""
    columns = list(data.columns)
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for left in columns:
        for right in columns:
            left_numeric = pd.api.types.is_numeric_dtype(data[left])
            right_numeric = pd.api.types.is_numeric_dtype(data[right])
            if left_numeric and right_numeric:
                matrix.loc[left, right] = correlation_pair(data[left], data[right])
            elif not left_numeric and not right_numeric:
                matrix.loc[left, right] = mutual_information(data[left], data[right])
            else:
                matrix.loc[left, right] = float("nan")
    return matrix
