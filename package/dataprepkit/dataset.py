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
    """Small object for managing a pandas DataFrame and an optional target column."""

    data: pd.DataFrame
    target: str | None = None

    @classmethod
    def from_csv(cls, path: str, target: str | None = None, **kwargs) -> "DataPrepDataset":
        """Read a CSV file into a DataPrepDataset."""
        return cls(pd.read_csv(path, **kwargs), target=target)

    def to_csv(self, path: str, **kwargs) -> None:
        """Write the dataset to CSV."""
        self.data.to_csv(path, index=False, **kwargs)

    def metrics(self, positive_class=None) -> pd.DataFrame:
        """Calculate attribute metrics for the dataset."""
        return attribute_metrics(self.data, target=self.target, positive_class=positive_class)

    def normalize(self, columns=None) -> "DataPrepDataset":
        """Return a normalized copy of the dataset."""
        return DataPrepDataset(normalize_dataset(self.data, columns=columns), target=self.target)

    def standardize(self, columns=None) -> "DataPrepDataset":
        """Return a standardized copy of the dataset."""
        return DataPrepDataset(standardize_dataset(self.data, columns=columns), target=self.target)

    def discretize_equal_width(self, bins: int = 5, columns=None) -> "DataPrepDataset":
        """Return a copy discretized with equal-width bins."""
        return DataPrepDataset(discretize_dataset_equal_width(self.data, bins=bins, columns=columns), target=self.target)

    def discretize_equal_frequency(self, bins: int = 5, columns=None) -> "DataPrepDataset":
        """Return a copy discretized with equal-frequency bins."""
        return DataPrepDataset(discretize_dataset_equal_frequency(self.data, bins=bins, columns=columns), target=self.target)

    def filter(self, metric: str, threshold: float, operator: str = ">=", positive_class=None) -> "DataPrepDataset":
        """Return a copy filtered by an attribute metric."""
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
        """Calculate the pairwise association matrix."""
        return association_matrix(self.data)
