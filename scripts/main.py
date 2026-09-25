"""Step 1 driver: configuration → engineering → reporting → verification.

Processing workflow:
    1. Parse project/configuration/output paths.
    2. Initialize detailed console and file logging.
    3. Load and validate system assumptions.
    4. Run the deterministic camera/optics/motion/bandwidth design module.
    5. Save machine-readable calculations, pass/fail checks and figures.
    6. Exit nonzero if engineering feasibility gates fail.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import argparse
import sys
from pathlib import Path

from vision_poc.configuration import load_config
from vision_poc.engineering import calculate
from vision_poc.logging_utils import configure_logging
from vision_poc.reporting import save_report

ROOT = Path(__file__).resolve().parents[1]

def main(argv: list[str] | None = None) -> int:
    """Run Step 1 system engineering from configuration to saved outputs.

    Args:
        argv: Optional CLI arguments; defaults to command-line input.
    Returns:
        Exit status: 0 for all feasibility gates passing, 1 otherwise.
    Raises:
        No expected exceptions: errors are logged and mapped to exit status 1.
    """
    parser = argparse.ArgumentParser(description="Step 1 machine-vision hardware design")
    parser.add_argument("--config", type=Path, default=ROOT / "config/system.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/step_01")
    args = parser.parse_args(argv)
    logger = configure_logging(ROOT / "logs/step_01.log")
    try:
        logger.info("Step 1 started; configuration=%s", args.config)
        config = load_config(args.config)
        logger.info("Loaded configuration; deriving optical and throughput constraints")
        result = calculate(config)
        save_report(result, args.output)
        for name, ok in result.checks.items():
            logger.info("CHECK %s = %s", name, "PASS" if ok else "FAIL")
        for warning in result.warnings:
            logger.warning("%s", warning)
        logger.info("Step 1 complete; outputs=%s", args.output)
        return 0 if all(result.checks.values()) else 1

    # ------------------------------------------
    # Fixed error:
    # -----------------------------------------
    # except (OSError, ValueError, KeyError, TypeError) as exc:
    #     logger.exception("Step 1 failed: %s", exc)
    #     return 1
    # -----------------------------------------
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Step 1 failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
