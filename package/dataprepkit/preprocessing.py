"""Normalization, standardization, and metric-based variable filtering."""

from __future__ import annotations

import pandas as pd

from .metrics import attribute_metrics


def normalize_variable(values) -> pd.Series:
    """Scale one numerical variable to the [0, 1] range."""
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("normalization requires numerical values.")
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series(0.0, index=series.index).where(series.notna(), pd.NA)
    return (series - minimum) / (maximum - minimum)


def standardize_variable(values) -> pd.Series:
    """Standardize one numerical variable to mean 0 and standard deviation 1."""
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("standardization requires numerical values.")
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index).where(series.notna(), pd.NA)
    return (series - mean) / std


def normalize_dataset(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Normalize numerical columns in a dataset."""
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = normalize_variable(result[column])
    return result


def standardize_dataset(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Standardize numerical columns in a dataset."""
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = standardize_variable(result[column])
    return result


def filter_variables(
    data: pd.DataFrame,
    metric: str,
    threshold: float,
    operator: str = ">=",
    target: str | None = None,
    positive_class=None,
) -> pd.DataFrame:
    """Filter variables according to a metric threshold."""
    metrics = attribute_metrics(data, target=target, positive_class=positive_class)
    selected_metrics = metrics[metrics["metric"] == metric]
    if operator == ">=":
        keep = selected_metrics[selected_metrics["value"] >= threshold]["attribute"]
    elif operator == ">":
        keep = selected_metrics[selected_metrics["value"] > threshold]["attribute"]
    elif operator == "<=":
        keep = selected_metrics[selected_metrics["value"] <= threshold]["attribute"]
    elif operator == "<":
        keep = selected_metrics[selected_metrics["value"] < threshold]["attribute"]
    else:
        raise ValueError("operator must be one of >=, >, <=, <.")
    columns = list(keep)
    if target is not None and target in data.columns:
        columns.append(target)
    return data.loc[:, columns]
