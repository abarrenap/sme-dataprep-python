"""Dataset object wrapper for coursework preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .associations import association_matrix
from .discretization import discretize_dataset_equal_frequency, discretize_dataset_equal_width
from .metrics import attribute_metrics
from .preprocessing import filter_variables, normalize_dataset, standardize_dataset


@dataclass
class DataPrepDataset:
    """Container for a dataset and its optional supervised target.

    `DataPrepDataset` is a lightweight object that keeps a pandas DataFrame
    together with the name of a target column. It provides method versions of
    the package functions so a user can write `dataset.normalize()` or
    `dataset.metrics()` without passing the DataFrame every time.

    Parameters
    ----------
    data:
        Dataset stored as a pandas DataFrame.
    target:
        Optional name of the target column. It is used by methods that need a
        supervised binary class, such as AUC.
    """

    data: pd.DataFrame
    target: str | None = None

    @classmethod
    def from_csv(cls, path: str, target: str | None = None, **kwargs) -> "DataPrepDataset":
        """Read a CSV file and return a `DataPrepDataset`.

        Parameters
        ----------
        path:
            Path to the CSV file.
        target:
            Optional target column name.
        **kwargs:
            Additional keyword arguments passed to `pandas.read_csv`.

        Returns
        -------
        DataPrepDataset
            Dataset object containing the loaded DataFrame and target name.
        """
        return cls(pd.read_csv(path, **kwargs), target=target)

    def to_csv(self, path: str, **kwargs) -> None:
        """Write the dataset DataFrame to a CSV file.

        Parameters
        ----------
        path:
            Output CSV path.
        **kwargs:
            Additional keyword arguments passed to `pandas.DataFrame.to_csv`.

        Returns
        -------
        None
            The file is written as a side effect.
        """
        self.data.to_csv(path, index=False, **kwargs)

    def metrics(self, positive_class=None) -> pd.DataFrame:
        """Calculate metrics for the dataset attributes.

        Parameters
        ----------
        positive_class:
            Optional class value treated as positive for AUC.

        Returns
        -------
        pandas.DataFrame
            Long-format table with attribute names, metric names, and metric
            values.
        """
        return attribute_metrics(self.data, target=self.target, positive_class=positive_class)

    def normalize(self, columns=None) -> "DataPrepDataset":
        """Return a dataset copy with numerical columns normalized to `[0, 1]`.

        Parameters
        ----------
        columns:
            Optional list of column names to normalize. If omitted, all
            numerical columns are transformed.

        Returns
        -------
        DataPrepDataset
            New dataset object with transformed data and the same target.
        """
        return DataPrepDataset(normalize_dataset(self.data, columns=columns), target=self.target)

    def standardize(self, columns=None) -> "DataPrepDataset":
        """Return a dataset copy with numerical columns standardized.

        Parameters
        ----------
        columns:
            Optional list of column names to standardize. If omitted, all
            numerical columns are transformed.

        Returns
        -------
        DataPrepDataset
            New dataset object with z-score transformed data and the same target.
        """
        return DataPrepDataset(standardize_dataset(self.data, columns=columns), target=self.target)

    def discretize_equal_width(self, bins: int = 5, columns=None) -> "DataPrepDataset":
        """Return a dataset copy discretized with equal-width bins.

        Parameters
        ----------
        bins:
            Number of same-width intervals.
        columns:
            Optional list of columns to discretize. If omitted, all numerical
            columns are transformed.

        Returns
        -------
        DataPrepDataset
            New dataset object with selected columns converted to bin labels.
        """
        return DataPrepDataset(discretize_dataset_equal_width(self.data, bins=bins, columns=columns), target=self.target)

    def discretize_equal_frequency(self, bins: int = 5, columns=None) -> "DataPrepDataset":
        """Return a dataset copy discretized with equal-frequency bins.

        Parameters
        ----------
        bins:
            Desired number of groups with similar row counts.
        columns:
            Optional list of columns to discretize. If omitted, all numerical
            columns are transformed.

        Returns
        -------
        DataPrepDataset
            New dataset object with selected columns converted to bin labels.
        """
        return DataPrepDataset(discretize_dataset_equal_frequency(self.data, bins=bins, columns=columns), target=self.target)

    def filter(self, metric: str, threshold: float, operator: str = ">=", positive_class=None) -> "DataPrepDataset":
        """Return a dataset copy filtered by an attribute metric.

        Parameters
        ----------
        metric:
            Metric name to filter by, such as `variance`, `auc`, or `entropy`.
        threshold:
            Numeric threshold used for the comparison.
        operator:
            Comparison operator: `>=`, `>`, `<=`, or `<`.
        positive_class:
            Optional class value treated as positive when filtering by AUC.

        Returns
        -------
        DataPrepDataset
            New dataset object containing only the selected variables and the
            target column when one is defined.
        """
        return DataPrepDataset(
            filter_variables(
                self.data,
                metric=metric,
                threshold=threshold,
                operator=operator,
                target=self.target,
                positive_class=positive_class,
            ),
            target=self.target,
        )

    def associations(self) -> pd.DataFrame:
        """Calculate pairwise associations between all dataset columns.

        Returns
        -------
        pandas.DataFrame
            Square matrix with Pearson correlations for numerical pairs, mutual
            information for categorical pairs, and missing values for mixed
            pairs.
        """
        return association_matrix(self.data)
