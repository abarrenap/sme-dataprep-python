"""Visualization helpers for AUC values and association matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
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
