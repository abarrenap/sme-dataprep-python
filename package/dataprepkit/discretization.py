"""Discretization algorithms for numerical variables and datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_bins(bins: int) -> None:
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")


def discretize_equal_width(values, bins: int = 5, labels=None) -> pd.Series:
    """Discretize one numerical variable using equal-width intervals.

    This function transforms a continuous numerical variable into an ordered
    categorical variable. The interval limits are created by splitting the
    distance between the minimum and maximum observed values into `bins`
    intervals with the same width. This is useful when a later step needs
    discrete input or when a continuous variable should be easier to interpret.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values. Missing values are
        ignored when calculating the interval limits and remain missing in the
        output.
    bins:
        Number of intervals to create. Must be a positive integer.
    labels:
        Optional labels for the resulting intervals. If omitted, labels are
        created as `bin_1`, `bin_2`, and so on.

    Returns
    -------
    pandas.Series
        Series with one discrete bin label per input value. If all non-missing
        values are equal, every non-missing value is assigned to `bin_1`.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    ValueError
        If `bins` is not a positive integer.
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
    """Discretize one numerical variable using equal-frequency bins.

    Equal-frequency discretization sorts the observations and places them into
    groups with approximately the same number of rows. Unlike equal-width
    discretization, the numerical size of each interval can be different, but
    the amount of data in each bin is more balanced.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values. Missing values remain
        missing in the output.
    bins:
        Desired number of frequency groups. If there are fewer unique values
        than requested bins, the effective number of bins is reduced.
    labels:
        Optional labels for the bins. Defaults to `bin_1`, `bin_2`, and so on.

    Returns
    -------
    pandas.Series
        Ordered bin labels with roughly similar frequencies.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    ValueError
        If `bins` is not a positive integer.
    """
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
    """Apply equal-width discretization to several dataset columns.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    bins:
        Number of equal-width intervals to create for each transformed column.
    columns:
        Optional list of column names to transform. If omitted, all numerical
        columns in `data` are discretized.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` where the selected numerical columns have been replaced
        by discrete bin labels. Non-selected columns are left unchanged.
    """
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = discretize_equal_width(result[column], bins=bins)
    return result


def discretize_dataset_equal_frequency(data: pd.DataFrame, bins: int = 5, columns=None) -> pd.DataFrame:
    """Apply equal-frequency discretization to several dataset columns.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    bins:
        Desired number of equally frequent groups for each transformed column.
    columns:
        Optional list of column names to transform. If omitted, all numerical
        columns in `data` are discretized.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` where the selected numerical columns have been replaced
        by equal-frequency bin labels. Non-selected columns are left unchanged.
    """
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = discretize_equal_frequency(result[column], bins=bins)
    return result
