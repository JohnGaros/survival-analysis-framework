"""Test script for automatic recovery system.

This script tests the recovery system using existing production artifacts
from the failed 31-hour run.

Usage:
    python scripts/test_recovery.py
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from survival_framework.recovery import attempt_recovery

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Test recovery system with existing production artifacts."""
    logger.info("="*70)
    logger.info("TESTING AUTOMATIC RECOVERY SYSTEM")
    logger.info("="*70)

    # Use existing production data
    repo_root = Path(__file__).parent.parent
    input_file = str(repo_root / "data/inputs/production/survival_inputs_complete.pkl")

    logger.info(f"\nInput file: {input_file}")
    logger.info("Testing recovery with existing production artifacts...")
    logger.info("")

    # Simulate an MLflow error
    class SimulatedMLflowError(Exception):
        """Simulated MLflow exception for testing."""
        pass

    simulated_error = SimulatedMLflowError(
        "Run '98bd42d60b5644b7815387dd52c24af3' not found"
    )

    # Test recovery
    success = attempt_recovery(
        run_type="production",
        input_file=input_file,
        logger=logger,
        original_error=simulated_error
    )

    logger.info("")
    logger.info("="*70)
    if success:
        logger.info("✓ RECOVERY TEST PASSED")
        logger.info("="*70)
        logger.info("Recovery system successfully:")
        logger.info("  - Detected completed models from incremental metrics")
        logger.info("  - Generated model_summary.csv")
        logger.info("  - Fitted final models")
        logger.info("  - Generated predictions")
        return 0
    else:
        logger.error("✗ RECOVERY TEST FAILED")
        logger.error("="*70)
        logger.error("Recovery system did not generate usable outputs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
