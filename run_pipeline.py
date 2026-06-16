# =============================================================================
# run_pipeline.py — Run the Full CloudGuard Pipeline
# =============================================================================
# This is the "master" script that runs all pipeline stages in order.
#
# Usage:
#   python run_pipeline.py              ← Run everything
#   python run_pipeline.py --stage 1   ← Run only Stage 1
#   python run_pipeline.py --stage 2   ← Run only Stage 2
#
# Note: Stage 3 (CNN-LSTM) is meant to run in Google Colab.
#       This script will remind you of that when it gets to Stage 3.
# =============================================================================

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils.helpers import setup_logger, ensure_directories

logger = setup_logger(__name__)


def run_stage(stage_number: int) -> None:
    """
    Runs a specific pipeline stage.

    Args:
        stage_number: 0 (download), 1 (iForest), 2 (XGBoost), or 3 (CNN-LSTM)
    """

    if stage_number == 0:
        logger.info("\n📥 STAGE 0: Downloading Datasets")
        logger.info("-" * 40)
        from src.data.download_datasets import (
            download_azure_policy_definitions,
            download_cloudsploit_results,
            download_nvd_cve_data,
        )
        download_azure_policy_definitions()
        download_cloudsploit_results()
        download_nvd_cve_data()

    elif stage_number == 1:
        logger.info("\n🌲 STAGE 1: Isolation Forest (Unsupervised Screening)")
        logger.info("-" * 40)
        from src.models.stage1_isolation_forest import run_stage1
        run_stage1()

    elif stage_number == 2:
        logger.info("\n🤖 STAGE 2: XGBoost Classifier (Supervised Classification)")
        logger.info("-" * 40)
        from src.models.stage2_classifier import run_stage2
        run_stage2()

    elif stage_number == 3:
        logger.info("\n🧠 STAGE 3: CNN-LSTM (Temporal Drift Detection)")
        logger.info("-" * 40)
        logger.info("Stage 3 requires a GPU and should be run in Google Colab.")
        logger.info("")
        logger.info("Instructions:")
        logger.info("  1. Go to https://colab.research.google.com")
        logger.info("  2. Upload notebooks/stage3_cnn_lstm.ipynb")
        logger.info("  3. Runtime → Change runtime type → GPU")
        logger.info("  4. Run all cells")
        logger.info("")
        logger.info("After training, download the model file and place it at:")
        logger.info(f"  {os.path.join(config.MODELS_DIR, 'stage3_cnn_lstm.pt')}")


def main():
    parser = argparse.ArgumentParser(description="Run the CloudGuard pipeline")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Run a specific stage (0=download, 1=iForest, 2=XGBoost, 3=CNN-LSTM). Default: run all."
    )
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("  CloudGuard Pipeline — IEEE HiPC 2026")
    logger.info("=" * 60)

    # Make sure all directories exist
    ensure_directories()

    if args.stage is not None:
        # Run only the requested stage
        run_stage(args.stage)
    else:
        # Run all stages in order
        for stage in [0, 1, 2]:
            run_stage(stage)
        run_stage(3)  # This just prints Colab instructions

    elapsed = time.time() - start_time
    logger.info(f"\n✅ Done in {elapsed:.1f}s")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info(f"Models saved to:  {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
