# SME DataPrep Python

`sme-dataprep-python` contains the Python implementation for the Software Matemático y Estadístico assignment. The project is organized as a small installable package plus a notebook that illustrates the required functions.

## Repository layout

```text
.
├── README.md
├── data/
│   └── titanic/
│       ├── test.csv
│       └── train.csv
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

## Current example dataset

The first example uses the Titanic dataset in `data/titanic/train.csv`.

- Binary target: `Survived`
- Numerical variables: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical variables: `Pclass`, `Sex`, `Embarked`

The notebook intentionally ignores identifiers and high-cardinality text fields such as `PassengerId`, `Name`, `Ticket`, and `Cabin` because they make the first explanation less clear.

The AUC visualization section includes two different plots:

- `plot_roc_curve`: the standard ROC curve for one numerical attribute.
- `compare_auc_values`: a barplot that compares final AUC values across several numerical attributes.

## Second dataset recommendation

For the later second example, choose another tabular dataset with numerical columns, categorical columns, and one binary target. Good Kaggle-style examples include customer churn, credit default, heart disease, bank marketing, or loan approval datasets.

## Publishing later

This package is prepared for future publication to PyPI. Before publishing, update the author metadata in `package/pyproject.toml`, choose a final version number, build the package, and upload it with `twine`.
