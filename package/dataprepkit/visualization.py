"""Visualization helpers for AUC values and association matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import attribute_metrics, auc_score


def _roc_curve_points(scores, target, positive_class=None) -> pd.DataFrame:
    score_series = pd.Series(scores)
    target_series = pd.Series(target)
    valid = score_series.notna() & target_series.notna()
    score_series = score_series[valid]
    target_series = target_series[valid]
    classes = list(pd.unique(target_series))
    if len(classes) != 2:
        raise ValueError("ROC curve requires exactly two target classes.")
    positive = positive_class if positive_class is not None else classes[-1]
    positive_count = int((target_series == positive).sum())
    negative_count = int((target_series != positive).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC curve requires at least one positive and one negative example.")

    thresholds = [float("inf")] + sorted(score_series.unique(), reverse=True) + [float("-inf")]
    rows = []
    for threshold in thresholds:
        predicted_positive = score_series >= threshold
        true_positive = int(((target_series == positive) & predicted_positive).sum())
        false_positive = int(((target_series != positive) & predicted_positive).sum())
        rows.append(
            {
                "threshold": threshold,
                "false_positive_rate": false_positive / negative_count,
                "true_positive_rate": true_positive / positive_count,
            }
        )
    return pd.DataFrame(rows)


def plot_roc_curve(scores, target, positive_class=None, ax=None, label: str | None = None):
    """Plot the ROC curve for one numerical attribute.

    The ROC curve is the standard plot behind AUC. Each point is created by
    choosing a threshold for the numerical score and calculating the false
    positive rate and true positive rate. The diagonal line represents random
    ordering. A curve closer to the top-left corner indicates better separation
    between the positive and negative classes.

    Parameters
    ----------
    scores:
        Numerical values of one attribute, such as Titanic `Fare` or `Age`.
    target:
        Binary class values with the same length as `scores`.
    positive_class:
        Class value considered positive. If omitted, the second distinct class
        found in `target` is used.
    ax:
        Optional matplotlib Axes object. If omitted, a new figure and axes are
        created.
    label:
        Optional label for the plotted attribute.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the ROC curve, random baseline, and AUC value.

    Raises
    ------
    ValueError
        If the target is not binary or one class has no valid observations.
    """
    curve = _roc_curve_points(scores, target, positive_class=positive_class)
    auc = auc_score(scores, target, positive_class=positive_class)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    curve_label = label or "attribute"
    ax.plot(
        curve["false_positive_rate"],
        curve["true_positive_rate"],
        marker="o",
        linewidth=2,
        markersize=3,
        label=f"{curve_label} (AUC = {auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random baseline")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    return ax


def plot_roc_curves(data: pd.DataFrame, target: str, columns=None, positive_class=None, ax=None):
    """Plot ROC curves for several numerical attributes.

    This function extends `plot_roc_curve` to multiple variables. It is useful
    when comparing which numerical attributes separate the positive and negative
    classes better. If `columns` is omitted, all numerical columns except the
    target are plotted.

    Parameters
    ----------
    data:
        Input pandas DataFrame containing numerical attributes and a binary
        target column.
    target:
        Name of the binary target column.
    columns:
        Optional list of numerical columns to plot. If omitted, all numerical
        columns except `target` are used.
    positive_class:
        Class value considered positive for AUC and ROC calculation.
    ax:
        Optional matplotlib Axes object. If omitted, a new figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing one ROC curve per selected numerical attribute.
    """
    selected = columns or [column for column in data.select_dtypes(include="number").columns if column != target]
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    for column in selected:
        curve = _roc_curve_points(data[column], data[target], positive_class=positive_class)
        auc = auc_score(data[column], data[target], positive_class=positive_class)
        ax.plot(
            curve["false_positive_rate"],
            curve["true_positive_rate"],
            linewidth=2,
            label=f"{column} (AUC = {auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random baseline")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend()
    return ax


def compare_auc_values(data: pd.DataFrame, target: str, positive_class=None, ax=None):
    """Compare AUC values for all numerical attributes with a barplot.

    This is a summary visualization, not the ROC curve itself. It is useful when
    several numerical variables have been evaluated and we want to compare their
    final AUC values side by side. Use `plot_roc_curve` to draw the standard ROC
    plot for one specific attribute.

    Parameters
    ----------
    data:
        Input pandas DataFrame containing numerical attributes and a binary
        target column.
    target:
        Name of the binary target column.
    positive_class:
        Optional class value treated as positive for AUC.
    ax:
        Optional matplotlib Axes object. If omitted, a new figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing a bar plot ordered from highest to lowest AUC.
    """
    metrics = attribute_metrics(data, target=target, positive_class=positive_class)
    auc_values = metrics[metrics["metric"] == "auc"].sort_values("value", ascending=False)
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.bar(auc_values["attribute"], auc_values["value"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Attribute")
    ax.set_title("AUC by numerical attribute")
    ax.tick_params(axis="x", rotation=45)
    return ax


def _kde_points(values, points: int = 200):
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        raise ValueError("distribution plot requires at least one non-missing numerical value.")
    minimum = float(series.min())
    maximum = float(series.max())
    if minimum == maximum:
        grid = np.linspace(minimum - 0.5, maximum + 0.5, points)
        bandwidth = 1.0
    else:
        grid = np.linspace(minimum, maximum, points)
        std = float(series.std(ddof=0))
        bandwidth = 1.06 * std * (len(series) ** (-1 / 5)) if std > 0 else (maximum - minimum) / 10
        if bandwidth == 0:
            bandwidth = 1.0
    differences = (grid[:, None] - series.to_numpy()[None, :]) / bandwidth
    density = np.exp(-0.5 * differences**2).sum(axis=1)
    density = density / (len(series) * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def plot_variable_distribution(
    data: pd.DataFrame,
    variable: str,
    target: str | None = None,
    by_class: bool = False,
    ax=None,
):
    """Plot a KDE-style distribution curve for a numerical variable.

    The function estimates the variable distribution using a simple Gaussian
    kernel density estimate implemented inside the package. When `by_class` is
    `True`, one density line is drawn for each class of the target variable.

    Parameters
    ----------
    data:
        Input pandas DataFrame.
    variable:
        Name of the numerical variable to plot.
    target:
        Optional class column. Required when `by_class=True`.
    by_class:
        If `False`, draw one overall density curve. If `True`, draw one density
        curve per target class.
    ax:
        Optional matplotlib Axes object. If omitted, a new figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the distribution curve or class-specific curves.

    Raises
    ------
    TypeError
        If `variable` is not numerical.
    ValueError
        If `by_class=True` and no `target` is provided.
    """
    if not pd.api.types.is_numeric_dtype(data[variable]):
        raise TypeError("distribution plot requires a numerical variable.")
    if by_class and target is None:
        raise ValueError("target must be provided when by_class=True.")
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    if by_class:
        for class_value in pd.Series(data[target]).dropna().unique():
            class_values = data.loc[data[target] == class_value, variable]
            grid, density = _kde_points(class_values)
            ax.plot(grid, density, linewidth=2, label=f"{target} = {class_value}")
        ax.legend()
    else:
        grid, density = _kde_points(data[variable])
        ax.plot(grid, density, linewidth=2, label=variable)
    ax.set_xlabel(variable)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {variable}")
    return ax


def plot_variavle_distribution(*args, **kwargs):
    """Alias for `plot_variable_distribution`.

    This keeps accidental calls with the misspelled name working. Prefer the
    correctly spelled `plot_variable_distribution` in new code.
    """
    return plot_variable_distribution(*args, **kwargs)


def plot_auc_values(data: pd.DataFrame, target: str, positive_class=None, ax=None):
    """Backward-compatible alias for `compare_auc_values`.

    Prefer `compare_auc_values` in new code because it describes the plot more
    accurately. A barplot compares final AUC values, while `plot_roc_curve`
    draws the specific ROC curve used to interpret AUC for one attribute.
    """
    return compare_auc_values(data, target=target, positive_class=positive_class, ax=ax)


def plot_association_matrix(matrix: pd.DataFrame, ax=None, cmap: str = "viridis"):
    """Plot an association matrix as a heatmap.

    Parameters
    ----------
    matrix:
        Square DataFrame returned by `association_matrix`.
    ax:
        Optional matplotlib Axes object. If omitted, a new figure and axes are
        created.
    cmap:
        Matplotlib color map used for the heatmap.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the heatmap and color bar.
    """
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    image = ax.imshow(matrix.astype(float), cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title("Association matrix")
    ax.figure.colorbar(image, ax=ax)
    return ax
