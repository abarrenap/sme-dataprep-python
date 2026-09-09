"""Pairwise association measures for dataset attributes."""

from __future__ import annotations

import math

import pandas as pd

from .metrics import entropy


def correlation_pair(x, y) -> float:
    """Calculate Pearson correlation between two numerical variables.

    Pearson correlation measures linear association. Values close to `1` mean
    that both variables tend to increase together, values close to `-1` mean
    that one variable tends to decrease when the other increases, and values
    close to `0` indicate little linear association.

    Parameters
    ----------
    x, y:
        Numerical sequences with the same length. Rows with missing values in
        either variable are ignored.

    Returns
    -------
    float
        Pearson correlation coefficient. Returns `nan` when there are no valid
        pairs and `0.0` when one variable has zero variance.
    """
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
    """Calculate mutual information between two categorical variables.

    Mutual information measures how much knowing one variable reduces
    uncertainty about another variable. It is calculated from entropy as
    `MI(X, Y) = H(X) + H(Y) - H(X, Y)`, where `H(X, Y)` is the entropy of the
    joint distribution.

    Parameters
    ----------
    x, y:
        Categorical or discrete sequences with the same length. Rows with
        missing values in either variable are ignored.
    base:
        Logarithm base used by the entropy calculation.

    Returns
    -------
    float
        Mutual information value. A value of `0.0` means the variables do not
        reduce uncertainty about each other in the observed data.
    """
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
    """Calculate a pairwise association matrix for a dataset.

    The function compares all pairs of columns. Numerical-numerical pairs use
    Pearson correlation. Categorical-categorical pairs use mutual information.
    Mixed numerical-categorical pairs are returned as missing values because
    this first coursework version keeps the two required association measures
    separate.

    Parameters
    ----------
    data:
        Input pandas DataFrame containing numerical and/or categorical columns.

    Returns
    -------
    pandas.DataFrame
        Square matrix whose rows and columns are the dataset attributes. Each
        cell contains the association value for that pair of variables.
    """
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
