"""Dataset-specific scope and availability checks.

The datasets are intentionally not pooled: their patients, observation units,
time semantics, and supported outcomes are different.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def discovered_data_path(folder: str, filename: str) -> Path:
    root = ROOT / "data" / folder
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else root / filename

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
        "target": "Readmission within 30 days",
        "state_definition": "Index hospital-encounter summary at discharge",
        "models": ["Logistic regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "HbA1c is categorical and must not be treated as a numerical HbA1c time series. Encounter ordering must be audited before temporal claims.",
        "required": [Path(os.environ.get("US_DIABETES_130_PATH", discovered_data_path("us_diabetes_130", "diabetic_data.csv"))), RESULTS / "us_diabetes_130_metrics.csv", RESULTS / "us_diabetes_130_predictions.csv", RESULTS / "us_diabetes_130_confusion.csv"],
    },
    "mimic_iv_demo": {
        "id": "mimic_iv_demo",
        "name": "MIMIC-IV Demo",
        "setting": "Demo hospital records",
        "target": "In-hospital mortality",
        "state_definition": "Admission demographic and administrative summary",
        "models": ["Logistic regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "The demo subset is for pipeline checks and feasibility only; it is too small for generalizable clinical-performance claims.",
        "required": [Path(os.environ.get("MIMIC_IV_DEMO_PATH", discovered_data_path("mimic_iv_demo", "admissions.csv.gz"))), RESULTS / "mimic_iv_demo_metrics.csv", RESULTS / "mimic_iv_demo_predictions.csv", RESULTS / "mimic_iv_demo_confusion.csv"],
    },
}


def dataset_summary(dataset_id: str) -> dict[str, object]:
    if dataset_id not in DATASETS:
        raise KeyError(dataset_id)
    item = DATASETS[dataset_id]
    required = item["required"]
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
