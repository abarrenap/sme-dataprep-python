"""Visualization helpers for AUC values and association matrices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import attribute_metrics


def plot_auc_values(data: pd.DataFrame, target: str, positive_class=None, ax=None):
    """Plot AUC values for all numerical attributes in a dataset.

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
