# =============================================================================
# src/utils/helpers.py — Shared Helper Functions
# =============================================================================
# These are small utility functions used by multiple scripts.
# Think of this as a toolbox that every other script can borrow from.
# =============================================================================

import os
import logging
import sys

# We import our central config so every script uses the same settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def setup_logger(name: str) -> logging.Logger:
    """
    Creates a logger that prints messages to both the terminal AND a log file.

    Why do we need this?
    Python's built-in print() works fine, but a logger gives us:
    - Timestamps on every message (so you know when things happened)
    - Log levels (INFO, WARNING, ERROR) so you can filter messages
    - A saved log file you can review later

    Usage:
        logger = setup_logger(__name__)
        logger.info("Starting download...")
        logger.warning("File already exists, skipping.")
        logger.error("Download failed!")

    Args:
        name: Usually just pass __name__ (the current file's module name)

    Returns:
        A configured Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Format: "2024-01-15 10:23:45 | INFO | download_datasets | Starting..."
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler 1: Print to terminal (force UTF-8 so Unicode symbols work on Windows)
    console_handler = logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))
    console_handler.setFormatter(formatter)

    # Handler 2: Write to log file
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(formatter)

    # Only add handlers if they aren't already there (prevents duplicate messages)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def ensure_directories() -> None:
    """
    Creates all required project directories if they don't already exist.

    Call this at the start of any script that reads or writes files.
    It's safe to call multiple times — it won't delete anything that exists.

    Directories created:
        data/raw/        ← Where downloaded datasets are saved
        data/processed/  ← Where cleaned data is saved
        data/splits/     ← Where train/val/test splits are saved
        outputs/models/  ← Where trained models are saved
        outputs/results/ ← Where metric CSV files are saved
        outputs/figures/ ← Where plots are saved
    """
    dirs_to_create = [
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.SPLITS_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.FIGURES_DIR,
    ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)  # exist_ok=True means "don't crash if it exists"


def check_file_exists(filepath: str, description: str) -> bool:
    """
    Checks if a file exists and logs a clear message either way.

    Args:
        filepath:    The full path to the file
        description: A human-readable name for what the file is (for logging)

    Returns:
        True if the file exists, False if it doesn't

    Example:
        if check_file_exists(config.AZURE_POLICY_RAW, "Azure Policy dataset"):
            # File exists, load it
        else:
            # File missing, download it first
    """
    logger = setup_logger(__name__)

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"✓ Found {description}: {filepath} ({size_mb:.1f} MB)")
        return True
    else:
        logger.warning(f"✗ Missing {description}: {filepath}")
        return False


def print_dataset_summary(df, name: str = "Dataset") -> None:
    """
    Prints a quick summary of a pandas DataFrame — useful for sanity checks.

    Call this after loading or creating any dataset to confirm it looks right.

    Args:
        df:   A pandas DataFrame
        name: A label for the dataset (e.g., "Training set")
    """
    logger = setup_logger(__name__)

    logger.info(f"\n{'='*50}")
    logger.info(f"  {name} Summary")
    logger.info(f"{'='*50}")
    logger.info(f"  Rows:    {len(df):,}")
    logger.info(f"  Columns: {df.shape[1]}")

    # Show class balance if the label column exists
    if "label" in df.columns:
        counts = df["label"].value_counts()
        total = len(df)
        logger.info(f"  Compliant (0):     {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)")
        logger.info(f"  Non-compliant (1): {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")

    logger.info(f"{'='*50}\n")
