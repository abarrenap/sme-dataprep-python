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


def discretize_by_thresholds(values, thresholds, labels=None) -> pd.Series:
    """Discretize one numerical variable using user-defined thresholds.

    Manual-threshold discretization is useful when cut points have a practical
    interpretation. For example, Titanic `Age` could be split into children,
    adults, and older passengers using thresholds chosen by the analyst.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values. Missing values remain
        missing in the output.
    thresholds:
        Ordered or unordered list of numerical cut points. The function sorts
        them and creates intervals `(-inf, threshold_1]`,
        `(threshold_1, threshold_2]`, ..., `(last_threshold, inf)`.
    labels:
        Optional labels for the resulting groups. The number of labels must be
        one more than the number of thresholds. If omitted, labels are created
        as `bin_1`, `bin_2`, and so on.

    Returns
    -------
    pandas.Series
        Ordered categorical bin labels.

    Raises
    ------
    TypeError
        If `values` or `thresholds` are not numerical.
    ValueError
        If `labels` has an invalid length.
    """
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("threshold discretization requires numerical values.")
    cuts = pd.Series(thresholds).dropna().sort_values().tolist()
    if not all(isinstance(value, (int, float, np.integer, np.floating)) for value in cuts):
        raise TypeError("thresholds must be numerical.")
    labels = labels or [f"bin_{i + 1}" for i in range(len(cuts) + 1)]
    if len(labels) != len(cuts) + 1:
        raise ValueError("labels must contain exactly len(thresholds) + 1 values.")
    breaks = [float("-inf")] + [float(value) for value in cuts] + [float("inf")]
    return pd.cut(series, bins=breaks, labels=labels, include_lowest=True, ordered=True)


def discretize_by_standard_deviation(values, sd_thresholds=(-1, 1), labels=None) -> pd.Series:
    """Discretize one numerical variable by distance from its mean.

    The function first converts the variable to z-scores by subtracting the mean
    and dividing by the population standard deviation. It then discretizes those
    z-scores using standard-deviation thresholds. The default creates three
    groups: values more than one standard deviation below the mean, values
    within one standard deviation of the mean, and values more than one standard
    deviation above the mean.

    Parameters
    ----------
    values:
        Sequence or pandas Series with numerical values.
    sd_thresholds:
        Cut points expressed in standard deviations from the mean. The default
        `(-1, 1)` creates low, typical, and high groups.
    labels:
        Optional labels. Must contain one more label than the number of standard
        deviation thresholds. Defaults to `low`, `typical`, and `high` for the
        default thresholds, otherwise `bin_1`, `bin_2`, and so on.

    Returns
    -------
    pandas.Series
        Ordered categorical labels based on distance from the mean. Missing
        values remain missing. If the variable has zero standard deviation, all
        non-missing values are assigned to the middle group when possible.

    Raises
    ------
    TypeError
        If `values` is not numerical.
    ValueError
        If `labels` has an invalid length.
    """
    series = pd.Series(values).copy()
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("standard-deviation discretization requires numerical values.")
    cuts = pd.Series(sd_thresholds).dropna().sort_values().tolist()
    if labels is None and cuts == [-1, 1]:
        labels = ["low", "typical", "high"]
    labels = labels or [f"bin_{i + 1}" for i in range(len(cuts) + 1)]
    if len(labels) != len(cuts) + 1:
        raise ValueError("labels must contain exactly len(sd_thresholds) + 1 values.")
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        middle = min(len(labels) // 2, len(labels) - 1)
        return pd.Series(np.where(series.isna(), pd.NA, labels[middle]), index=series.index)
    z_scores = (series - mean) / std
    return discretize_by_thresholds(z_scores, thresholds=cuts, labels=labels)


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


def discretize_dataset_by_thresholds(data: pd.DataFrame, thresholds, columns=None, labels=None) -> pd.DataFrame:
    """Apply manual-threshold discretization to selected dataset columns.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    thresholds:
        Either one list of thresholds used for every selected column, or a
        dictionary mapping column names to their own threshold lists.
    columns:
        Optional columns to transform. If omitted, all numerical columns are
        transformed.
    labels:
        Optional labels shared by all transformed columns.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with selected columns replaced by manual-threshold bins.
    """
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        column_thresholds = thresholds[column] if isinstance(thresholds, dict) else thresholds
        result[column] = discretize_by_thresholds(result[column], thresholds=column_thresholds, labels=labels)
    return result


def discretize_dataset_by_standard_deviation(
    data: pd.DataFrame,
    sd_thresholds=(-1, 1),
    columns=None,
    labels=None,
) -> pd.DataFrame:
    """Apply standard-deviation discretization to selected dataset columns.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    sd_thresholds:
        Cut points expressed in standard deviations from each column mean.
    columns:
        Optional columns to transform. If omitted, all numerical columns are
        transformed.
    labels:
        Optional labels shared by all transformed columns.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with selected columns replaced by standard-deviation
        based bins.
    """
    result = data.copy()
    selected = columns or result.select_dtypes(include="number").columns
    for column in selected:
        result[column] = discretize_by_standard_deviation(result[column], sd_thresholds=sd_thresholds, labels=labels)
    return result
