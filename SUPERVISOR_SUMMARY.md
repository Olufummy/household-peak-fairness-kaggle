# Supervisor explanation

## Two-minute description

The project predicts unusually high household electricity demand one hour ahead. A peak is defined separately for each home as demand above that home's 95th percentile, calculated from its training period only. This makes the target household-specific and prevents future test observations from influencing the label.

The prepared hourly table combines lagged and rolling electricity-demand features, Edinburgh weather variables and cyclical time variables. Homes remain in chronological training, validation and test periods. Load features are divided by the household-specific training threshold so that differences in household scale do not dominate the models.

Three approaches are compared: a simple 24-hour seasonal rule, XGBoost and LightGBM. Each model's classification cut-off is selected by maximising F1 on the validation set. That cut-off is frozen before the test set is evaluated. The reported metrics include ROC-AUC, average precision, F1, sensitivity, specificity and balanced accuracy, which are more informative than raw accuracy for a rare peak class.

Fairness is assessed by comparing sensitivity across equivalised-income, dwelling-type and household-size groups. Sensitivity is central because it measures how often real peaks are detected. For the income comparison, confidence intervals are produced by resampling whole households rather than individual rows, preserving the clustered structure of repeated observations from the same home.

## What to show in a meeting

1. The workflow diagram in the README.
2. `model_metrics.csv` and the model-performance graph.
3. The group-sensitivity graph, explaining that a smaller maximum–minimum gap is more equitable.
4. `income_bootstrap.csv`, explaining that the interval describes uncertainty at household level.
5. The top feature-importance graph, while noting that importance is not causation.

## Methodological safeguards

- Household thresholds use training observations only.
- Validation data choose the model cut-off; test data do not.
- Missing demographic values are displayed but excluded from fairness-gap calculations.
- Bootstrap resampling is performed by household.
- The public repository contains code and documentation only, not participant-level data or predictions.
- Quick demo results are never treated as dissertation findings.
