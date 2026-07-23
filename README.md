# Household Peak Prediction and Fairness — Kaggle Edition

This is the short, presentation-ready version of the dissertation analysis. The complete experiment is kept in **one Python file**: [`kaggle_household_peak_fairness.py`](kaggle_household_peak_fairness.py).

## What the code does

It predicts whether a home's future electricity demand exceeds that home's **95th percentile training threshold**. It compares a 24-hour seasonal baseline with XGBoost and LightGBM, then checks whether model sensitivity differs across income, dwelling type and household size.

```mermaid
flowchart LR
    A["Prepared household-hour data"] --> B["Existing train / validation / test split"]
    B --> C["Scale load features by household threshold"]
    C --> D["Seasonal baseline, XGBoost, LightGBM"]
    D --> E["Choose F1 threshold on validation only"]
    E --> F["Evaluate test set once"]
    F --> G["Group fairness and household bootstrap"]
    G --> H["CSV results and Seaborn figures"]
```

The design prevents test leakage: the peak threshold was calculated from training data, the decision threshold is selected from validation data, and the test set is used only for final evaluation.

 IDEAL sensor data + historical Open-Meteo data

                    ↓
       supervised_95_features.csv.gz

                    ↓
      Kaggle modelling and fairness code



## Run it on Kaggle

1. In Kaggle, create a **private Dataset** and upload `supervised_95_features.csv.gz` from the prepared-data folder of the main project. Do not make participant-level data public.
2. Create a new Kaggle Notebook and use **Add Input** to attach that private Dataset.
3. Open `kaggle_household_peak_fairness.py`, copy all of it into one notebook code cell, and run the cell. Kaggle already supplies the required libraries in most notebook images.
4. Leave `DEMO_MODE = True` for the first run. This uses a fixed random subset from each split so you can check the tables and graphs quickly.
5. Change `DEMO_MODE = False` and run again for the full dissertation analysis. Only the full-data run is suitable for reporting.
6. Open `/kaggle/working/household_peak_outputs` to download the result tables and figures.

No GPU is required. Set the Kaggle accelerator to **None**. The code uses two CPU threads to control memory use.

## Outputs

- `model_metrics.csv` — test performance for all models.
- `group_sensitivity.csv` — test sensitivity by demographic group.
- `income_bootstrap.csv` — household-clustered confidence intervals for the income sensitivity gap.
- `feature_importance.csv` — tree-model feature importance.
- Four `.png` figures — performance, group sensitivity, confusion matrices and feature importance.

## Important interpretation rule

`DEMO_MODE = True` is only a software check. Its sampled results must not be quoted in the dissertation. Keep the full-data run's Kaggle version, settings, CSV files and figures as the reproducibility record.

## Data

The prepared file is derived from the IDEAL Household Energy Dataset and Open-Meteo historical weather data. It is deliberately excluded from this public repository. See the original data providers for their terms and documentation:

- [IDEAL Household Energy Dataset](https://datashare.ed.ac.uk/handle/10283/3647)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
