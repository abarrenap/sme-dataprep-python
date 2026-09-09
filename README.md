# SME DataPrep Python

`sme-dataprep-python` contains the Python implementation for the Software Matemático y Estadístico assignment. The project is organized as a small installable package plus a notebook that illustrates the required functions.

## Repository layout

```text
.
├── README.md
├── usage_examples.ipynb
└── package/
    ├── pyproject.toml
    ├── dataprepkit/
    │   ├── __init__.py
    │   ├── associations.py
    │   ├── dataset.py
    │   ├── discretization.py
    │   ├── metrics.py
    │   ├── preprocessing.py
    │   └── visualization.py
```

## Install locally

From this folder:

```bash
cd package
python3 -m pip install -e .
```

## Dataset recommendation

For the final notebook, choose a tabular dataset with:

- Several numerical columns, so variance, AUC, normalization, standardization, correlation, and discretization can be shown.
- Several categorical columns, so entropy and mutual information can be shown.
- One binary target/class column, such as `yes/no`, `0/1`, `approved/rejected`, or `disease/no_disease`, because AUC needs a supervised binary class.
- A manageable size, ideally between 500 and 20,000 rows, so examples run quickly.

Good Kaggle-style examples include customer churn, credit default, Titanic survival, heart disease, bank marketing, or loan approval datasets.

## Publishing later

This package is prepared for future publication to PyPI. Before publishing, update the author metadata in `package/pyproject.toml`, choose a final version number, build the package, and upload it with `twine`.
