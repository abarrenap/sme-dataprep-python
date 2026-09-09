"""Educational data preprocessing tools for SME coursework."""

from .associations import association_matrix, correlation_pair, mutual_information
from .dataset import DataPrepDataset
from .discretization import (
    discretize_by_standard_deviation,
    discretize_by_thresholds,
    discretize_dataset_by_standard_deviation,
    discretize_dataset_by_thresholds,
    discretize_dataset_equal_frequency,
    discretize_dataset_equal_width,
    discretize_equal_frequency,
    discretize_equal_width,
)
from .metrics import attribute_metrics, auc_score, entropy, variance
from .preprocessing import (
    filter_variables,
    normalize_dataset,
    normalize_variable,
    standardize_dataset,
    standardize_variable,
)
from .visualization import (
    compare_auc_values,
    plot_association_matrix,
    plot_auc_values,
    plot_roc_curve,
    plot_roc_curves,
    plot_variable_distribution,
    plot_variavle_distribution,
)

__all__ = [
    "DataPrepDataset",
    "association_matrix",
    "attribute_metrics",
    "auc_score",
    "correlation_pair",
    "discretize_by_standard_deviation",
    "discretize_by_thresholds",
    "discretize_dataset_by_standard_deviation",
    "discretize_dataset_by_thresholds",
    "discretize_dataset_equal_frequency",
    "discretize_dataset_equal_width",
    "discretize_equal_frequency",
    "discretize_equal_width",
    "entropy",
    "filter_variables",
    "mutual_information",
    "normalize_dataset",
    "normalize_variable",
    "compare_auc_values",
    "plot_association_matrix",
    "plot_auc_values",
    "plot_roc_curve",
    "plot_roc_curves",
    "plot_variable_distribution",
    "plot_variavle_distribution",
    "standardize_dataset",
    "standardize_variable",
    "variance",
]
