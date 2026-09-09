# SME DataPrep Python

`sme-dataprep-python` contains the Python implementation for the Software Matemático y Estadístico assignment. The project is organized as a small installable package plus a notebook that illustrates the required functions.

## Repository layout

```text
.
├── README.md
├── data/
│   ├── diabetes/
│   │   └── diabetes_prediction_dataset.csv
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
    │   ├── management.py
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

## Example datasets

The notebook uses two datasets:

1. `data/titanic/train.csv` is used for a general preprocessing walkthrough because it has numerical variables, categorical variables, missing values, and a binary target.
2. `data/diabetes/diabetes_prediction_dataset.csv` is used for a compact realistic analysis focused on target validation, missing-value checks, AUC/ROC, distribution plots, and short conclusions.

- Titanic binary target: `Survived`
- Numerical variables: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical variables: `Pclass`, `Sex`, `Embarked`

- Diabetes binary target: `diabetes`
- Main numerical variables: `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`
- Other relevant variables: `hypertension`, `heart_disease`, `gender`, `smoking_history`

The notebook intentionally treats the first dataset as a generic supervised table rather than focusing on its story. The final diabetes section is more analytical and interprets the most relevant outputs.

The AUC visualization section includes two different plots:

- `plot_roc_curve`: the standard ROC curve for one numerical attribute.
- `plot_roc_curves`: several ROC curves in one figure, using selected columns or all numerical columns when `columns=None`.
- `compare_auc_values`: a barplot that compares final AUC values across several numerical attributes.

Additional implemented extras:

- `discretize_by_thresholds`: manual cut points chosen by the analyst.
- `discretize_by_standard_deviation`: groups values by distance from the mean.
- `plot_variable_distribution`: KDE-style distribution plot, optionally split by target class.
- `dataset_summary`: compact overview of rows, columns, types, and missing values.
- `missing_value_report`: missing counts and missing percentages by variable.
- `detect_variable_types`: practical type detection for numerical, categorical, binary, identifier, and high-cardinality variables.
- `validate_binary_target`: checks that a target exists and has exactly two classes.
- `impute_missing_values`: simple missing-value imputation with median/mode or explicit strategies.

## Publishing later

This package is prepared for future publication to PyPI. Before publishing, update the author metadata in `package/pyproject.toml`, choose a final version number, build the package, and upload it with `twine`.
