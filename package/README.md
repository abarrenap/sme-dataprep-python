# sme-dataprep

Educational data-preprocessing utilities for the Software Matemático y
Estadístico coursework. The package includes dataset management, missing-value
handling, normalization, standardization, discretization, association metrics,
AUC/ROC helpers, and visualizations.

## Installation

Install the package from PyPI:

```bash
python -m pip install sme-dataprep
```

In a Jupyter notebook, you can run:

```python
%pip install sme-dataprep
```

## Quick start

The distribution is named `sme-dataprep`; import its API from `dataprepkit`:

```python
import pandas as pd

from dataprepkit import dataset_summary, normalize_variable

data = pd.DataFrame({"score": [10, 20, 30], "group": ["a", "a", "b"]})

print(dataset_summary(data))
scaled_score = normalize_variable(data["score"])
```

The project source and additional examples are available in the
[GitHub repository](https://github.com/abarrenap/sme-dataprep-python).
