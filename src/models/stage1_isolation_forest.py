# =============================================================================
# src/models/stage1_isolation_forest.py — Stage 1: Isolation Forest
# =============================================================================
# Stage 1 screens the dataset WITHOUT using any labels.
# This is called "unsupervised learning" — the model finds anomalies
# purely by looking at the structure of the data.
#
# How Isolation Forest works (simple explanation):
#   1. Build many random decision trees
#   2. For each data point, count how many splits it takes to isolate it
#   3. Points that are easy to isolate (few splits) = anomalies
#   4. Points that take many splits to isolate = normal
#
# Why this matters for CloudGuard:
#   Policy definitions with unusual combinations of features
#   (e.g., PCR=0 AND enforcement_mode=0 AND low scope) are quickly
#   isolated — these are the governance gaps we're looking for.
#
# Outputs:
#   - outputs/models/stage1_iforest.pkl     (saved model)
#   - outputs/results/stage1_results.csv    (anomaly scores + predictions)
#   - outputs/results/stage1_metrics.json   (F1, Recall, Precision, ROC-AUC)
#
# Run: python src/models/stage1_isolation_forest.py
# =============================================================================

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.utils.helpers import setup_logger, ensure_directories, check_file_exists

logger = setup_logger(__name__)


# =============================================================================
# Load Data
# =============================================================================

def load_data() -> tuple:
    """
    Loads the train and test splits created by build_features.py.

    Returns:
        Tuple of (X_train, X_test, y_test)
        - X_train: Feature matrix for training (no labels used!)
        - X_test:  Feature matrix for evaluation
        - y_test:  True labels for evaluation (used ONLY to measure performance)

    Note: Isolation Forest is trained WITHOUT labels (unsupervised).
    We only use y_test to measure how well the model detected true anomalies.
    """
    logger.info("Loading train/test data...")

    for filepath, name in [(config.TRAIN_FILE, "train"), (config.TEST_FILE, "test")]:
        if not check_file_exists(filepath, f"{name} split"):
            raise FileNotFoundError(
                f"Split not found: {filepath}\n"
                "Run build_features.py first: python src/features/build_features.py"
            )

    train = pd.read_csv(config.TRAIN_FILE)
    test  = pd.read_csv(config.TEST_FILE)

    # Features only — no label for training (that's the point of unsupervised)
    X_train = train[config.FEATURE_COLUMNS]
    X_test  = test[config.FEATURE_COLUMNS]
    y_test  = test[config.TARGET_COLUMN]

    logger.info(f"Training features: {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")
    logger.info(f"Test features:     {X_test.shape[0]:,} rows")
    logger.info(f"Test labels:       {y_test.sum():,} non-compliant / {len(y_test):,} total")

    return X_train, X_test, y_test


# =============================================================================
# Train Model
# =============================================================================

def train_isolation_forest(X_train: pd.DataFrame) -> IsolationForest:
    """
    Trains the Isolation Forest model on the training features.

    Key hyperparameter — contamination:
        This tells the model what fraction of the data to expect as anomalies.
        We set it to 0.15 (15%) based on literature estimates for cloud
        misconfiguration rates (IBM X-Force 2024: ~14.9%).

    Args:
        X_train: Feature matrix (no labels)

    Returns:
        Trained IsolationForest model
    """
    logger.info("Training Isolation Forest...")
    logger.info(f"  n_estimators:  {config.IFOREST_CONFIG['n_estimators']}")
    logger.info(f"  contamination: {config.IFOREST_CONFIG['contamination']}")
    logger.info(f"  max_samples:   {config.IFOREST_CONFIG['max_samples']}")

    model = IsolationForest(**config.IFOREST_CONFIG)
    model.fit(X_train)

    logger.info("Training complete.")
    return model


# =============================================================================
# Generate Predictions
# =============================================================================

def predict(model: IsolationForest, X_test: pd.DataFrame) -> tuple:
    """
    Generates anomaly predictions and scores for the test set.

    Isolation Forest outputs two things:
        1. predict(): Returns -1 (anomaly) or +1 (normal)
           We convert this to 1 (non-compliant) and 0 (compliant)

        2. decision_function(): Returns a continuous anomaly score
           Higher score = more anomalous = more likely a governance gap
           We use this for ROC-AUC calculation

    Args:
        model:  Trained IsolationForest
        X_test: Test feature matrix

    Returns:
        Tuple of (predictions, anomaly_scores)
        - predictions:    Array of 0/1 labels
        - anomaly_scores: Array of continuous scores (higher = more anomalous)
    """
    logger.info("Generating predictions on test set...")

    # Raw predictions: -1 = anomaly, +1 = normal
    raw_predictions = model.predict(X_test)

    # Convert to our label convention: 1 = non-compliant (anomaly), 0 = compliant
    # IsolationForest uses -1 for anomalies, we want 1
    predictions = (raw_predictions == -1).astype(int)

    # Anomaly scores: higher = more anomalous
    # decision_function returns negative scores; we negate so higher = more anomalous
    anomaly_scores = -model.decision_function(X_test)

    logger.info(f"Predicted non-compliant: {predictions.sum():,} / {len(predictions):,}")

    return predictions, anomaly_scores


# =============================================================================
# Evaluate Performance
# =============================================================================

def evaluate(y_true: pd.Series, predictions: np.ndarray,
             anomaly_scores: np.ndarray) -> dict:
    """
    Calculates all performance metrics for the paper's results tables.

    Primary metrics (Table V in the paper):
        - F1 Score:   Harmonic mean of precision and recall (main metric)
        - Recall:     Fraction of true governance gaps we detected
        - Precision:  Fraction of our alerts that were real governance gaps
        - MCC:        Matthews Correlation Coefficient (robust to imbalance)

    Secondary metric:
        - ROC-AUC:    Area under the ROC curve (uses continuous scores)

    Args:
        y_true:        True labels from the test set
        predictions:   Binary predictions (0/1)
        anomaly_scores: Continuous anomaly scores

    Returns:
        Dictionary of metric names → values
    """
    logger.info("Evaluating model performance...")

    # Calculate metrics — we set pos_label=1 to focus on the non-compliant class
    f1        = f1_score(y_true, predictions, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, predictions, pos_label=1, zero_division=0)
    precision = precision_score(y_true, predictions, pos_label=1, zero_division=0)
    roc_auc   = roc_auc_score(y_true, anomaly_scores)

    # MCC: Matthews Correlation Coefficient
    # Ranges from -1 to +1. +1 = perfect, 0 = random, -1 = inverse
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_true, predictions)

    metrics = {
        "f1":        round(f1, 4),
        "recall":    round(recall, 4),
        "precision": round(precision, 4),
        "mcc":       round(mcc, 4),
        "roc_auc":   round(roc_auc, 4),
    }

    # Print a clear summary
    logger.info("\n" + "=" * 50)
    logger.info("  Stage 1 — Isolation Forest Results")
    logger.info("=" * 50)
    logger.info(f"  F1 Score:  {f1:.4f}  ← primary metric")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  MCC:       {mcc:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info("=" * 50)

    # Full classification report
    logger.info("\nDetailed Classification Report:")
    logger.info("\n" + classification_report(
        y_true, predictions,
        target_names=["Compliant (0)", "Non-Compliant (1)"],
        zero_division=0
    ))

    return metrics


# =============================================================================
# Save Outputs
# =============================================================================

def save_outputs(model: IsolationForest, X_test: pd.DataFrame,
                 y_test: pd.Series, predictions: np.ndarray,
                 anomaly_scores: np.ndarray, metrics: dict) -> None:
    """
    Saves the trained model, results CSV, metrics JSON, and confusion matrix plot.

    These outputs are used by:
        - Stage 2 (stage2_classifier.py) reads the pseudo-labels from results CSV
        - The paper's results tables are built from metrics JSON
        - The confusion matrix plot can be included as a figure in the paper

    Args:
        model:         Trained IsolationForest model
        X_test:        Test feature matrix
        y_test:        True labels
        predictions:   Binary predictions
        anomaly_scores: Continuous anomaly scores
        metrics:       Performance metrics dictionary
    """
    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(config.MODELS_DIR, "stage1_iforest.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model: {model_path}")

    # ── Save results CSV (used by Stage 2 as pseudo-labels) ──────────────────
    results_df = X_test.copy()
    results_df["true_label"]     = y_test.values
    results_df["predicted_label"] = predictions
    results_df["anomaly_score"]  = anomaly_scores

    results_path = os.path.join(config.RESULTS_DIR, "stage1_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results: {results_path}")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = os.path.join(config.RESULTS_DIR, "stage1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # ── Save confusion matrix plot ────────────────────────────────────────────
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Compliant", "Non-Compliant"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Stage 1: Isolation Forest\nConfusion Matrix (Test Set)")
    plt.tight_layout()

    cm_path = os.path.join(config.FIGURES_DIR, "stage1_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {cm_path}")


# =============================================================================
# Also save pseudo-labels for Stage 2 training
# =============================================================================

def save_pseudo_labels_for_stage2(model: IsolationForest) -> None:
    """
    Runs the Isolation Forest on the TRAINING set to generate pseudo-labels
    that Stage 2 will use for supervised training.

    This is the key link between Stage 1 and Stage 2:
        Stage 1 (unsupervised) → discovers anomalies → produces pseudo-labels
        Stage 2 (supervised)   → trains on pseudo-labels → learns the pattern

    The pseudo-labels are imperfect (Stage 1 F1 ≈ 0.78) but good enough
    to bootstrap a supervised classifier that achieves F1 ≈ 0.91.
    """
    logger.info("Generating pseudo-labels for Stage 2 training data...")

    train = pd.read_csv(config.TRAIN_FILE)
    X_train = train[config.FEATURE_COLUMNS]

    raw_pred = model.predict(X_train)
    pseudo_labels = (raw_pred == -1).astype(int)

    # Add pseudo-labels to training data
    train["pseudo_label"] = pseudo_labels
    train["anomaly_score"] = -model.decision_function(X_train)

    pseudo_path = os.path.join(config.PROCESSED_DIR, "train_with_pseudo_labels.csv")
    train.to_csv(pseudo_path, index=False)

    logger.info(f"Pseudo-labels: {pseudo_labels.sum():,} flagged non-compliant / {len(pseudo_labels):,} total")
    logger.info(f"Saved to: {pseudo_path}")


# =============================================================================
# Public entry point (called by run_pipeline.py)
# =============================================================================

def run_stage1() -> dict:
    """
    Runs the complete Stage 1 pipeline and returns metrics.
    Called by run_pipeline.py with: from src.models.stage1_isolation_forest import run_stage1
    """
    ensure_directories()

    X_train, X_test, y_test = load_data()
    model = train_isolation_forest(X_train)
    predictions, anomaly_scores = predict(model, X_test)
    metrics = evaluate(y_test, predictions, anomaly_scores)
    save_outputs(model, X_test, y_test, predictions, anomaly_scores, metrics)
    save_pseudo_labels_for_stage2(model)

    return metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("CloudGuard — Stage 1: Isolation Forest")
    metrics = run_stage1()

    logger.info("\n" + "=" * 60)
    logger.info("Stage 1 complete!")
    logger.info("Next step: python src/models/stage2_classifier.py")
    logger.info("=" * 60)
