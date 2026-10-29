"""Dataset-specific scope and availability checks.

The datasets are intentionally not pooled: their patients, observation units,
time semantics, and supported outcomes are different.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = {
    "shanghai_t2d": {
        "id": "shanghai_t2d",
        "name": "Shanghai T2D",
        "setting": "Continuous glucose monitoring",
        "target": "CGM value 15, 30, or 60 minutes ahead",
        "state_definition": "Latest CGM, one-hour change, and six-hour summary statistics",
        "models": ["Persistence", "Ridge regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "Retrospective CGM forecasting only. It does not establish treatment benefit or long-term disease progression.",
        "required": [RESULTS / "predictions.csv", RESULTS / "metrics.csv", RESULTS / "patient_metrics.csv", RESULTS / "zone_confusion.csv"],
    },
    "us_diabetes_130": {
        "id": "us_diabetes_130",
        "name": "Diabetes 130-US Hospitals",
        "setting": "Hospital encounters",
        "target": "A carefully defined future encounter outcome, such as readmission",
        "state_definition": "Encounter history available before the selected prediction time",
        "models": ["Logistic regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "HbA1c is categorical and must not be treated as a numerical HbA1c time series. Encounter ordering must be audited before temporal claims.",
        "required": [Path(os.environ.get("US_DIABETES_130_PATH", ROOT / "data" / "us_diabetes_130" / "diabetic_data.csv"))],
    },
    "mimic_iv_demo": {
        "id": "mimic_iv_demo",
        "name": "MIMIC-IV Demo",
        "setting": "Demo hospital records",
        "target": "A documented encounter-level outcome supported by the supplied demo tables",
        "state_definition": "Time-stamped admissions and measurements available before the index time",
        "models": ["Logistic regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "The demo subset is for pipeline checks and feasibility only; it is too small for generalizable clinical-performance claims.",
        "required": [Path(os.environ.get("MIMIC_IV_DEMO_PATH", ROOT / "data" / "mimic_iv_demo" / "admissions.csv"))],
    },
}


def dataset_summary(dataset_id: str) -> dict[str, object]:
    if dataset_id not in DATASETS:
        raise KeyError(dataset_id)
    item = DATASETS[dataset_id]
    required = item["required"]
    if dataset_id == "mimic_iv_demo":
        ready = required[0].exists() or required[0].with_suffix(".csv.gz").exists()
    else:
        ready = all(path.exists() for path in required)
    return {
        key: value for key, value in item.items() if key != "required"
    } | {
        "status": "ready" if ready else "not_configured",
        "required_path": str(required[0]),
        "message": (
            "Data and evaluated outputs are available locally."
            if ready
            else "No compatible local data were found. Add the documented source files before enabling this dataset."
        ),
    }


def all_datasets() -> list[dict[str, object]]:
    return [dataset_summary(dataset_id) for dataset_id in DATASETS]
