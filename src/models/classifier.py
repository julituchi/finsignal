"""
src/models/classifier.py
--------------------------
XGBoost classifier for Module 1: predicting post-earnings return direction.

Design decisions baked in:
  1. Time-series cross-validation (walk-forward), NOT random splits
  2. Class imbalance handling via scale_pos_weight
  3. SHAP explainability on the final model
  4. Saves model artifact to data/processed/module1_model.json
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_CSV  = PROCESSED_DIR / "module1_features.csv"
MODEL_PATH    = PROCESSED_DIR / "module1_model.json"
PLOTS_DIR     = PROCESSED_DIR / "plots"

# Features the model sees: never include forward-looking columns here
FEATURE_COLS = [
    "sentiment_score",
    "sentiment_std",
    "mean_positive",
    "mean_negative",
    "pct_positive",
    "pct_negative",
    "qa_delta_score",          # Q&A minus prepared remarks: key signal
    "prepared_sentiment_score",
    "qa_sentiment_score",
    "transcript_length",
    "prepared_n_sentences",
    "qa_n_sentences",
]

TARGET_COL = "label_1d"   # 1 = stock went up, 0 = down


def load_features() -> pd.DataFrame:
    """Load the merged feature matrix and sort by date (required for time-series CV)."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["earnings_date"])
    df.sort_values("earnings_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def time_series_cv(df: pd.DataFrame,
                    n_splits: int = 4) -> list[tuple]:
    """
    Walk-forward cross-validation splits.

    Why NOT random splits: with random splits, training data can include
    events from 2024 while test data includes events from 2021; the model
    sees the future. Walk-forward ensures training always precedes testing.

    With 80 rows and 4 splits, each test fold ≈ 16 events.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    splits = list(tscv.split(X))
    return splits, X, y


def train_xgboost(X_train: np.ndarray,
                   y_train: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier.

    scale_pos_weight handles class imbalance:
      if 60% of labels are 0 (down), scale_pos_weight = 60/40 = 1.5
      this tells XGBoost to weight the minority class (up) more heavily.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,           # shallow trees → less overfitting on small dataset
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true: np.ndarray,
              y_pred: np.ndarray,
              y_prob: np.ndarray,
              fold: int) -> dict:
    """Compute and print metrics for one CV fold."""
    acc     = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n  Fold {fold}:")
    print(f"    Accuracy : {acc:.3f}")
    print(f"    ROC-AUC  : {roc_auc:.3f}")
    print(f"    Baseline : {max(y_true.mean(), 1-y_true.mean()):.3f}  (always-predict-majority)")

    return {"fold": fold, "accuracy": acc, "roc_auc": roc_auc,
            "n_test": len(y_true)}


def run_cross_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full walk-forward CV and return results DataFrame.
    Also prints a summary at the end.
    """
    splits, X, y = time_series_cv(df)
    results = []

    print(f"Walk-forward CV: {len(splits)} folds on {len(df)} earnings events")
    print("=" * 50)

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model  = train_xgboost(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        result = evaluate(y_test, y_pred, y_prob, fold)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("CV Summary:")
    print(f"  Mean Accuracy : {results_df['accuracy'].mean():.3f} "
          f"± {results_df['accuracy'].std():.3f}")
    print(f"  Mean ROC-AUC  : {results_df['roc_auc'].mean():.3f} "
          f"± {results_df['roc_auc'].std():.3f}")
    print(f"\n  Note: ROC-AUC > 0.55 on earnings direction is a meaningful signal.")
    print(f"  The Efficient Market Hypothesis predicts ~0.50.")

    return results_df


def train_final_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train on the full dataset for the final saved model.
    This is the model used in the Streamlit dashboard.
    """
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values
    model = train_xgboost(X, y)
    model.save_model(str(MODEL_PATH))
    print(f"\n✓ Final model saved to {MODEL_PATH}")
    return model


def plot_shap(model: xgb.XGBClassifier,
               df: pd.DataFrame) -> None:
    """
    Generate SHAP summary plot: shows which features drive predictions most.

    SHAP (SHapley Additive exPlanations) assigns each feature a contribution
    value for each prediction. It answers: "why did the model predict UP here?"

    In a regulated environment (like a bank), this is not optional; a risk
    officer needs to be able to audit what the model is responding to.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame(df[FEATURE_COLS].fillna(0).values, columns=FEATURE_COLS)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Summary plot: feature importance ranked by mean |SHAP value|
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Feature Importance: Module 1 Classifier")
    plt.tight_layout()
    out_path = PLOTS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ SHAP plot saved to {out_path}")


def load_model() -> xgb.XGBClassifier:
    """Load a previously saved model from disk."""
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


if __name__ == "__main__":
    df = load_features()
    print(f"Loaded {len(df)} rows, {len(FEATURE_COLS)} features\n")

    # Step 1: Cross-validation
    cv_results = run_cross_validation(df)

    # Step 2: Train final model on all data
    final_model = train_final_model(df)

    # Step 3: SHAP explainability
    plot_shap(final_model, df)
