"""Evaluate the two non-CGM datasets with separate standard sklearn models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encounter_common import evaluate_models
from mimic_iv_demo import load_study as load_mimic_study, models as mimic_models
from us_diabetes_130 import load_study as load_us_study, models as us_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Run separate patient-held-out encounter evaluations.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    studies = [
        (load_us_study(args.data_root / "us_diabetes_130"), us_models),
        (load_mimic_study(args.data_root / "mimic_iv_demo"), mimic_models),
    ]
    for study, build_models in studies:
        metrics, _, _ = evaluate_models(study, build_models(study), args.output)
        print(f"{study.display_name}: {len(metrics)} models evaluated; outputs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
