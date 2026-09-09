"""Dataset management utilities for preprocessing workflows."""

from __future__ import annotations

import pandas as pd


def detect_variable_types(data: pd.DataFrame, high_cardinality_threshold: float = 0.5) -> pd.DataFrame:
    """Classify dataset variables by practical data-analysis type.

    This helper goes beyond the raw pandas dtype. It identifies binary
    variables, numerical variables, categorical variables, possible identifiers,
    and high-cardinality categorical variables. The result is useful before
    deciding which preprocessing or metric function should be applied.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    high_cardinality_threshold:
        Proportion of unique non-missing values above which a text/categorical
        variable is marked as high cardinality. For example, `0.5` means that a
        column with unique values in more than half of the rows is considered
        high cardinality.

    Returns
    -------
    pandas.DataFrame
        Table with one row per variable and columns `variable`, `pandas_dtype`,
        `detected_type`, `missing_count`, `missing_rate`, `unique_count`, and
        `unique_rate`.
    """
    rows = []
    row_count = len(data)
    for column in data.columns:
        series = data[column]
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())
        unique_rate = unique_count / len(non_missing) if len(non_missing) else 0.0
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / row_count if row_count else 0.0

        if unique_count == 2:
            detected = "binary"
        elif pd.api.types.is_numeric_dtype(series):
            detected = "identifier" if unique_count == len(non_missing) and row_count > 0 else "numerical"
        elif unique_rate >= high_cardinality_threshold:
            detected = "high_cardinality_categorical"
        else:
            detected = "categorical"

        rows.append(
            {
                "variable": column,
                "pandas_dtype": str(series.dtype),
                "detected_type": detected,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "unique_rate": unique_rate,
            }
        )
    return pd.DataFrame(rows)


def missing_value_report(data: pd.DataFrame) -> pd.DataFrame:
    """Report missing values for every variable in a dataset.

    Parameters
    ----------
    data:
        Input pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        Table ordered by decreasing missing rate with columns `variable`,
        `missing_count`, and `missing_rate`.
    """
    row_count = len(data)
    report = pd.DataFrame(
        {
            "variable": data.columns,
            "missing_count": [int(data[column].isna().sum()) for column in data.columns],
        }
    )
    report["missing_rate"] = report["missing_count"] / row_count if row_count else 0.0
    return report.sort_values(["missing_rate", "missing_count"], ascending=False).reset_index(drop=True)


def dataset_summary(data: pd.DataFrame) -> dict:
    """Create a compact summary of a dataset.

    The summary combines general dimensions, variable type detection, and
    missing-value information. It is designed to be the first function called
    when exploring a new dataset.

    Parameters
    ----------
    data:
        Input pandas DataFrame.

    Returns
    -------
    dict
        Dictionary with `n_rows`, `n_columns`, `variables`, `missing_values`,
        `numerical_columns`, and `categorical_columns`.
    """
    variable_info = detect_variable_types(data)
    return {
        "n_rows": int(data.shape[0]),
        "n_columns": int(data.shape[1]),
        "variables": variable_info,
        "missing_values": missing_value_report(data),
        "numerical_columns": list(data.select_dtypes(include="number").columns),
        "categorical_columns": list(data.select_dtypes(exclude="number").columns),
    }


def validate_binary_target(data: pd.DataFrame, target: str) -> bool:
    """Validate that a target column exists and has exactly two classes.

    AUC requires a supervised binary target. This function checks that the
    target is present and that the non-missing target values contain exactly two
    distinct classes.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    target:
        Name of the target column.

    Returns
    -------
    bool
        `True` when the target is valid.

    Raises
    ------
    ValueError
        If the target column is missing or does not contain exactly two classes.
    """
    if target not in data.columns:
        raise ValueError(f"target column '{target}' was not found.")
    classes = data[target].dropna().unique()
    if len(classes) != 2:
        raise ValueError("target must contain exactly two non-missing classes.")
    return True


def impute_missing_values(data: pd.DataFrame, strategy: str = "auto", fill_value=None, columns=None) -> pd.DataFrame:
    """Impute missing values in selected dataset columns.

    The function implements simple, transparent imputation rules. Numerical
    columns can be imputed with the mean, median, or a constant. Categorical
    columns can be imputed with the most frequent value (mode) or a constant.
    With `strategy='auto'`, numerical columns use the median and categorical
    columns use the mode.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    strategy:
        Imputation strategy. Supported values are `auto`, `mean`, `median`,
        `mode`, and `constant`.
    fill_value:
        Constant value used when `strategy='constant'`.
    columns:
        Optional list of columns to impute. If omitted, all columns are checked.

    Returns
    -------
    pandas.DataFrame
        Copy of `data` with missing values filled in the selected columns.

    Raises
    ------
    ValueError
        If an unsupported strategy is requested or a constant strategy is used
        without `fill_value`.
    """
    allowed = {"auto", "mean", "median", "mode", "constant"}
    if strategy not in allowed:
        raise ValueError(f"strategy must be one of {sorted(allowed)}.")
    if strategy == "constant" and fill_value is None:
        raise ValueError("fill_value must be provided when strategy='constant'.")

    result = data.copy()
    selected = columns or result.columns
    for column in selected:
        series = result[column]
        if not series.isna().any():
            continue
        column_strategy = strategy
        if strategy == "auto":
            column_strategy = "median" if pd.api.types.is_numeric_dtype(series) else "mode"
        if column_strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("mean imputation can only be applied to numerical columns.")
            replacement = series.mean(skipna=True)
        elif column_strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("median imputation can only be applied to numerical columns.")
            replacement = series.median(skipna=True)
        elif column_strategy == "mode":
            modes = series.dropna().mode()
            replacement = modes.iloc[0] if not modes.empty else fill_value
        else:
            replacement = fill_value
        result[column] = series.fillna(replacement)
    return result
