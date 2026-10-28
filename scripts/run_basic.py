"""Run the simplified standard-model workflow."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shanghai_t2d.basic import evaluate_basic
from shanghai_t2d.config import StudyConfig
from shanghai_t2d.data import discover_dataset
from shanghai_t2d.features import build_supervised_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard-model ShanghaiT2DM workflow.")
    parser.add_argument("--data-root", type=Path, default=Path("Shangai Data"))
    parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="Use three folds and 250 bootstrap samples.")
    args = parser.parse_args()

    config = StudyConfig.from_json(args.config)
    if args.quick:
        config = replace(config, outer_folds=3, bootstrap_iterations=250)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = discover_dataset(args.data_root)
    # This large intermediate table can always be rebuilt from the supplied data.
    # Keep the results folder focused on evaluated predictions and study evidence.
    supervised = build_supervised_dataset(paths, config)
    results = evaluate_basic(supervised, config, args.output)
    print(f"Basic workflow complete. Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
