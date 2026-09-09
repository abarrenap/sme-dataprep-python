"""Normalization, standardization, and metric-based variable filtering."""

from __future__ import annotations

import pandas as pd

from .metrics import attribute_metrics


def normalize_variable(values) -> pd.Series:
    """Normalize one numerical variable to the `[0, 1]` range.

    Normalization changes the scale of a variable without changing the order of
    its values. The minimum non-missing value becomes `0`, the maximum becomes
    `1`, and every other value is placed proportionally between them.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values.

    Returns
    -------
    pandas.Series
        Normalized values. Missing values remain missing. If all non-missing
        values are equal, they are returned as `0.0` because there is no range
        to scale.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    """
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("normalization requires numerical values.")
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)
    if pd.isna(minimum) or pd.isna(maximum) or minimum == maximum:
        return pd.Series(0.0, index=series.index).where(series.notna(), pd.NA)
    return (series - minimum) / (maximum - minimum)


def standardize_variable(values) -> pd.Series:
    """Standardize one numerical variable using z-scores.

    Standardization subtracts the mean and divides by the standard deviation.
    The transformed variable has mean `0` and standard deviation `1`, which
    makes variables measured in different units easier to compare.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values.

    Returns
    -------
    pandas.Series
        Standardized values. Missing values remain missing. If the variable has
        zero standard deviation, all non-missing values are returned as `0.0`.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    """
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("standardization requires numerical values.")
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index).where(series.notna(), pd.NA)
    return (series - mean) / std


def normalize_dataset(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Normalize selected numerical columns in a dataset.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    columns:
        Optional list of column names to normalize. If omitted, all numerical
        columns are normalized.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with selected columns scaled to `[0, 1]`. Non-selected
        columns are left unchanged.
    """
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = normalize_variable(result[column])
    return result


def standardize_dataset(data: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Standardize selected numerical columns in a dataset.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    columns:
        Optional list of column names to standardize. If omitted, all numerical
        columns are standardized.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with selected columns transformed to z-scores. Non-
        selected columns are left unchanged.
    """
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
    """Filter dataset variables according to a metric threshold.

    The function first calls `attribute_metrics`, then keeps only the attributes
    whose selected metric satisfies the comparison rule. This is a simple
    feature-filtering method useful for reducing a dataset before later
    analysis.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    metric:
        Metric used for filtering. Supported values are the metric names
        produced by `attribute_metrics`, such as `variance`, `auc`, or
        `entropy`.
    threshold:
        Numeric threshold used in the comparison.
    operator:
        Comparison operator. Must be one of `>=`, `>`, `<=`, or `<`.
    target:
        Optional binary target column. Required when filtering by AUC.
    positive_class:
        Optional class value treated as positive for AUC.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with the variables that pass the filter. When `target` is
        provided, the target column is kept in the output.

    Raises
    ------
    ValueError
        If `operator` is not supported, or if AUC is requested with a non-binary
        target.
    """
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
