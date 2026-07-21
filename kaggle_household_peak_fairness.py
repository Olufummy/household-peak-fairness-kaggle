"""One-file Kaggle workflow for household peak prediction and fairness analysis.

Copy this entire file into one Kaggle notebook cell. The prepared dataset must be
attached privately and named supervised_95_features.csv.gz.
"""

from pathlib import Path
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ----------------------------- 1. SETTINGS -----------------------------
SEED = 42
DEMO_MODE = True              # True = quick software check; False = final analysis
DEMO_ROWS = {"train": 80_000, "validation": 20_000, "test": 20_000}
BOOTSTRAP_REPEATS = 500 if DEMO_MODE else 5_000
OUTPUT_DIR = Path("/kaggle/working/household_peak_outputs")

LOAD_FEATURES = [
    "load_lag_0h_w", "load_lag_1h_w", "load_lag_2h_w", "load_lag_3h_w",
    "load_lag_6h_w", "load_lag_12h_w", "load_lag_24h_w", "load_lag_48h_w",
    "load_roll_24h_mean_w", "load_roll_24h_std_w",
    "load_roll_168h_mean_w", "load_roll_168h_std_w",
]
OTHER_FEATURES = [
    "origin_temperature_2m_c", "origin_relative_humidity_2m_pct",
    "origin_precipitation_mm", "origin_wind_speed_10m_kmh",
    "temperature_lag_24h_c", "humidity_lag_24h_pct",
    "precipitation_lag_24h_mm", "wind_speed_lag_24h_kmh",
    "target_hour_sin", "target_hour_cos", "target_weekday_sin",
    "target_weekday_cos", "target_month_sin", "target_month_cos",
    "target_is_weekend",
]
FEATURES = LOAD_FEATURES + OTHER_FEATURES
GROUPS = ["equivalised_income", "hometype", "household_size"]


# -------------------------- 2. DATA PREPARATION -------------------------
def find_data_file():
    """Find the attached Kaggle input, or a local copy when testing."""
    candidates = list(Path("/kaggle/input").glob("**/supervised_95_features.csv.gz"))
    candidates += list(Path(".").glob("**/supervised_95_features.csv.gz"))
    if not candidates:
        raise FileNotFoundError(
            "Attach a private Kaggle Dataset containing supervised_95_features.csv.gz."
        )
    print(f"Using data: {candidates[0]}")
    return candidates[0]


def load_data(path):
    """Read only required columns, scale load features, and optionally sample."""
    needed = ["homeid", "split", "target_peak", "peak_threshold_w"] + GROUPS + FEATURES
    data = pd.read_csv(path, usecols=needed, low_memory=False)
    data["split"] = data["split"].replace({"val": "validation", "valid": "validation"})

    # Household scale adjustment. The denominator was calculated on training data.
    data[LOAD_FEATURES] = data[LOAD_FEATURES].div(data["peak_threshold_w"], axis=0)
    data[FEATURES] = data[FEATURES].astype("float32")
    data["target_peak"] = data["target_peak"].astype("int8")
    for column in GROUPS:
        data[column] = data[column].fillna("missing").astype(str)

    if DEMO_MODE:
        print("WARNING: DEMO_MODE is on. These sampled results are not reportable.")
        parts = []
        for split, limit in DEMO_ROWS.items():
            part = data[data["split"] == split]
            parts.append(part.sample(min(limit, len(part)), random_state=SEED))
        data = pd.concat(parts, ignore_index=True)

    print(data.groupby("split")["target_peak"].agg(rows="size", peaks="sum"))
    return data


# ------------------------- 3. MODELS AND METRICS ------------------------
def best_f1_threshold(y, score):
    """Find the exact score boundary with maximum validation F1."""
    order = np.argsort(-score, kind="mergesort")
    sorted_y, sorted_score = y[order], score[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    fn = sorted_y.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, float), where=(2 * tp + fp + fn) != 0)
    boundaries = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    return float(sorted_score[boundaries[np.argmax(f1[boundaries])]])


def calculate_metrics(y, score, threshold):
    prediction = (score >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "mcc": matthews_corrcoef(y, prediction),
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def train_models(train, validation):
    """Fit resource-controlled models and return validation probabilities."""
    x_train, y_train = train[FEATURES], train["target_peak"].to_numpy()
    x_valid, y_valid = validation[FEATURES], validation["target_peak"].to_numpy()
    imbalance = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=250 if DEMO_MODE else 600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=imbalance, objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist", n_jobs=2, random_state=SEED,
    )
    lightgbm = lgb.LGBMClassifier(
        n_estimators=250 if DEMO_MODE else 600, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        scale_pos_weight=imbalance, n_jobs=2, random_state=SEED, verbosity=-1,
    )

    models = {"XGBoost": xgb, "LightGBM": lightgbm}
    validation_scores = {
        "Seasonal 24h": (validation["load_lag_24h_w"].to_numpy() >= 1).astype(float)
    }
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        validation_scores[name] = model.predict_proba(x_valid)[:, 1]
    return models, validation_scores


def evaluate(models, validation_scores, validation, test):
    """Freeze validation thresholds, then evaluate the untouched test split."""
    y_valid = validation["target_peak"].to_numpy()
    y_test = test["target_peak"].to_numpy()
    test_scores = {"Seasonal 24h": (test["load_lag_24h_w"].to_numpy() >= 1).astype(float)}
    for name, model in models.items():
        test_scores[name] = model.predict_proba(test[FEATURES])[:, 1]

    rows, predictions = [], {}
    for name, valid_score in validation_scores.items():
        threshold = best_f1_threshold(y_valid, valid_score)
        rows.append({"model": name, **calculate_metrics(y_test, test_scores[name], threshold)})
        predictions[name] = (test_scores[name] >= threshold).astype("int8")
    return pd.DataFrame(rows), predictions


# --------------------------- 4. FAIRNESS AUDIT --------------------------
def group_sensitivity(test, predictions):
    """Sensitivity by group; missing values are shown but excluded from gaps."""
    rows = []
    for model, prediction in predictions.items():
        for attribute in GROUPS:
            for group, indices in test.groupby(attribute, observed=True).groups.items():
                y = test.loc[indices, "target_peak"].to_numpy()
                p = prediction[test.index.get_indexer(indices)]
                positives = int(y.sum())
                rows.append({
                    "model": model, "attribute": attribute, "group": str(group),
                    "rows": len(y), "peaks": positives,
                    "sensitivity": float(p[y == 1].mean()) if positives else np.nan,
                })
    return pd.DataFrame(rows)


def bootstrap_income_gap(test, predictions):
    """Household-clustered percentile CI for the income sensitivity gap."""
    rng, rows = np.random.default_rng(SEED), []
    for model, prediction in predictions.items():
        frame = test[["homeid", "equivalised_income", "target_peak"]].copy()
        frame["true_positive"] = (frame["target_peak"].to_numpy() * prediction)
        homes = frame.groupby(["homeid", "equivalised_income"], observed=True).agg(
            peaks=("target_peak", "sum"), true_positives=("true_positive", "sum")
        ).reset_index()
        homes = homes[homes["equivalised_income"] != "missing"]
        gaps = []
        for _ in range(BOOTSTRAP_REPEATS):
            sampled = homes.iloc[rng.integers(0, len(homes), len(homes))]
            totals = sampled.groupby("equivalised_income", observed=True)[["peaks", "true_positives"]].sum()
            sensitivity = totals["true_positives"].div(totals["peaks"].replace(0, np.nan)).dropna()
            if len(sensitivity) >= 2:
                gaps.append(sensitivity.max() - sensitivity.min())
        point = group_sensitivity(test, {model: prediction})
        point = point[(point["attribute"] == "equivalised_income") & (point["group"] != "missing")]["sensitivity"]
        rows.append({
            "model": model, "households": homes["homeid"].nunique(),
            "repeats": len(gaps), "sensitivity_gap": point.max() - point.min(),
            "ci_2.5%": np.quantile(gaps, 0.025), "ci_97.5%": np.quantile(gaps, 0.975),
        })
    return pd.DataFrame(rows)


# ------------------------------ 5. FIGURES ------------------------------
def make_figures(metrics, group_results, models):
    sns.set_theme(style="whitegrid", context="talk")
    palette = "colorblind"

    long = metrics.melt(
        id_vars="model", value_vars=["roc_auc", "average_precision", "f1", "sensitivity", "balanced_accuracy"],
        var_name="metric", value_name="score",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=long, x="metric", y="score", hue="model", palette=palette)
    plt.ylim(0, 1); plt.title("Test-set model performance"); plt.xlabel(""); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_model_performance.png", dpi=300); plt.show()

    shown = group_results[(group_results["group"] != "missing") & group_results["sensitivity"].notna()]
    chart = sns.catplot(
        data=shown, x="group", y="sensitivity", hue="model", col="attribute",
        kind="bar", palette=palette, sharex=False, height=5, aspect=1.05,
    )
    chart.set_xticklabels(rotation=35, ha="right"); chart.set_axis_labels("", "Sensitivity")
    chart.set_titles("{col_name}"); chart.figure.suptitle("Peak detection by demographic group", y=1.04)
    chart.savefig(OUTPUT_DIR / "figure_group_sensitivity.png", dpi=300); plt.show()

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for axis, row in zip(np.atleast_1d(axes), metrics.itertuples()):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", cbar=False, ax=axis)
        axis.set(title=row.model, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "figure_confusion_matrices.png", dpi=300); plt.show()

    importance = []
    for name, model in models.items():
        importance.extend({"model": name, "feature": f, "importance": v} for f, v in zip(FEATURES, model.feature_importances_))
    importance = pd.DataFrame(importance)
    top = importance.groupby("feature")["importance"].mean().nlargest(12).index
    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance[importance["feature"].isin(top)], y="feature", x="importance", hue="model", palette=palette)
    plt.title("Top model feature importances"); plt.ylabel(""); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_feature_importance.png", dpi=300); plt.show()
    return importance


# ------------------------------- 6. RUN ---------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(find_data_file())
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("The file must contain train, validation and test rows in 'split'.")

    models, validation_scores = train_models(train, validation)
    metrics, predictions = evaluate(models, validation_scores, validation, test)
    groups = group_sensitivity(test, predictions)
    bootstrap = bootstrap_income_gap(test, predictions)
    importance = make_figures(metrics, groups, models)

    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    groups.to_csv(OUTPUT_DIR / "group_sensitivity.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "income_bootstrap.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print("\nTest metrics\n", metrics.round(3).to_string(index=False))
    print("\nIncome bootstrap\n", bootstrap.round(3).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
