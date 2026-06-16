# =============================================================================
# config.py — CloudGuard Central Configuration
# =============================================================================
# All paths, hyperparameters, and settings live here.
# When you need to change something (e.g., a file path or model setting),
# change it HERE rather than hunting through multiple files.
# =============================================================================

import os

# =============================================================================
# PATHS
# =============================================================================

# The root directory of the project (the folder containing this file)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")          # Downloaded datasets
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")    # Cleaned data
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")       # Train/val/test splits

# Output directories
OUTPUTS_DIR     = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR      = os.path.join(OUTPUTS_DIR, "models")    # Saved model files
RESULTS_DIR     = os.path.join(OUTPUTS_DIR, "results")   # Metric CSV files
FIGURES_DIR     = os.path.join(OUTPUTS_DIR, "figures")   # Plots and charts

# Specific data file paths
AZURE_POLICY_RAW    = os.path.join(RAW_DIR, "azure_policy_definitions.json")
CLOUDSPLOIT_RAW     = os.path.join(RAW_DIR, "cloudsploit_results.json")
NVD_RAW             = os.path.join(RAW_DIR, "nvd_cve_azure.json")

PROCESSED_DATASET   = os.path.join(PROCESSED_DIR, "cloudguard_dataset.csv")
TRAIN_FILE          = os.path.join(SPLITS_DIR, "train.csv")
VAL_FILE            = os.path.join(SPLITS_DIR, "val.csv")
TEST_FILE           = os.path.join(SPLITS_DIR, "test.csv")

# =============================================================================
# DATASET SETTINGS
# =============================================================================

# Random seeds — we run 5 experiments with different seeds and average results
# This is required by the paper's methodology
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.85   # 85% for training
VAL_RATIO   = 0.00   # no separate validation split
TEST_RATIO  = 0.15   # 15% for final evaluation (only used once!)

# Class labels
COMPLIANT_LABEL     = 0   # Resource is properly governed
NON_COMPLIANT_LABEL = 1   # Resource has a policy gap (minority class)

# =============================================================================
# FEATURE NAMES
# =============================================================================

# These are the 7 features described in the paper (Section III-B)
FEATURE_COLUMNS = [
    "pcr",                  # Policy Coverage Ratio (novel — our contribution)
    "dvl",                  # Deployment-to-Vulnerability Lag (novel — our contribution)
    "enforcement_mode",     # 1 = Deny, 0 = Audit or Disabled
    "scope_level",          # 2 = Mgmt Group, 1 = Subscription, 0 = Resource Group
    "policy_age_days",      # Days since policy was last modified
    "resource_type_flag",   # 1 if resource type has any policy assigned
    "vuln_count_30d",       # Rolling 30-day CVE count for this resource type
]

TARGET_COLUMN = "label"    # 0 = compliant, 1 = non-compliant

# =============================================================================
# STAGE 1: ISOLATION FOREST SETTINGS
# =============================================================================

IFOREST_CONFIG = {
    "n_estimators": 200,      # Number of trees in the forest
    "max_samples": 256,       # Samples per tree (OptIForest recommendation)
    "contamination": 0.15,    # Expected fraction of anomalies (from literature)
    "random_state": 42,       # For reproducibility
    "n_jobs": -1,             # Use all CPU cores
}

# Threshold: anomaly scores above this are flagged as policy gaps
IFOREST_ANOMALY_THRESHOLD = 0.7

# =============================================================================
# STAGE 2: CLASSIFIER SETTINGS
# =============================================================================

XGBOOST_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 6.0,  # Class imbalance weight (majority:minority ratio)
    "early_stopping_rounds": 50,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

# SMOTE settings for class imbalance correction
SMOTE_CONFIG = {
    "k_neighbors": 5,         # Number of nearest neighbors for interpolation
    "random_state": 42,
}

# Cross-validation folds for Stage 2
CV_FOLDS = 5

# =============================================================================
# STAGE 3: CNN-LSTM SETTINGS
# =============================================================================

LSTM_CONFIG = {
    "sequence_length": 30,    # 30-day time windows (T in the paper)
    "prediction_horizon": 7,  # Predict 7 days ahead
    "conv_filters": 64,       # Number of CNN filters
    "conv_kernel_size": 3,    # CNN kernel size
    "lstm_units_1": 128,      # First LSTM layer size
    "lstm_units_2": 64,       # Second LSTM layer size
    "dropout_rate": 0.3,      # Dropout for regularization
    "dense_units": 32,        # Dense layer before output
    "learning_rate": 0.001,   # Adam optimizer learning rate
    "batch_size": 64,
    "max_epochs": 100,        # Maximum training epochs
    "patience": 10,           # Early stopping patience
}

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Primary metrics (what the paper reports)
PRIMARY_METRICS = ["f1", "recall", "precision", "mcc"]

# Secondary metrics
SECONDARY_METRICS = ["roc_auc"]

# NOTE: Accuracy is intentionally excluded — it is misleading for imbalanced data.
# A model that predicts "compliant" for everything gets 85.7% accuracy but catches
# zero non-compliant cases. We never report accuracy as a primary metric.

# PCR threshold below which we flag an environment as high-risk
PCR_HIGH_RISK_THRESHOLD = 0.7

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = "INFO"   # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE  = os.path.join(ROOT_DIR, "cloudguard.log")
