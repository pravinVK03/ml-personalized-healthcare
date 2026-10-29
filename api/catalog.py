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
        "purpose": "Estimate a person's near-term glucose course from their recent continuous glucose measurements.",
        "target": "CGM value 15, 30, or 60 minutes ahead",
        "input_boundary": "Only CGM measurements recorded at or before the selected prediction time are used.",
        "state_definition": "Latest CGM, one-hour change, and six-hour summary statistics",
        "trajectory_note": "This source contains repeated CGM readings, so its chart shows the recorded glucose path followed by short-horizon forecasts.",
        "feature_groups": [
            {"name": "Recent CGM history", "items": ["Current glucose", "Eight 15-minute glucose lags"]},
            {"name": "Glucose change", "items": ["15-, 30-, and 60-minute changes"]},
            {"name": "Rolling summaries", "items": ["1-, 2-, and 6-hour mean", "Variation, minimum, and maximum"]},
            {"name": "Time context", "items": ["Time of day encoded as sine and cosine"]},
        ],
        "models": ["Persistence", "Ridge regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "Retrospective CGM forecasting only. It does not establish treatment benefit or long-term disease progression.",
        "required": [RESULTS / "predictions.csv", RESULTS / "metrics.csv", RESULTS / "patient_metrics.csv", RESULTS / "zone_confusion.csv"],
    },
    "us_diabetes_130": {
        "id": "us_diabetes_130",
        "name": "Diabetes 130-US Hospitals",
        "setting": "Hospital encounters",
        "purpose": "Estimate whether a hospital encounter is followed by readmission within 30 days.",
        "target": "Readmission within 30 days",
        "input_boundary": "Only fields documented for the index encounter at discharge are used; later encounters are excluded.",
        "state_definition": "Index hospital-encounter summary at discharge",
        "trajectory_note": "This source is evaluated as one encounter at a time. It does not provide a validated within-person time trajectory for the displayed record.",
        "feature_groups": [
            {"name": "Encounter use", "items": ["Time in hospital", "Laboratory procedures", "Procedures", "Medication count", "Diagnosis count"]},
            {"name": "Prior service use", "items": ["Outpatient visits", "Emergency visits", "Inpatient visits"]},
            {"name": "Patient and admission context", "items": ["Age group", "Race", "Gender", "Admission type and source", "Payer and specialty"]},
            {"name": "Diabetes-related fields", "items": ["Maximum serum glucose category", "HbA1c result category", "Insulin status", "Medication change", "Diabetes medication flag"]},
        ],
        "models": ["Logistic regression", "Random forest", "Histogram gradient boosting"],
        "limitations": "HbA1c is categorical and must not be treated as a numerical HbA1c time series. Encounter ordering must be audited before temporal claims.",
        "required": [Path(os.environ.get("US_DIABETES_130_PATH", discovered_data_path("us_diabetes_130", "diabetic_data.csv"))), RESULTS / "us_diabetes_130_metrics.csv", RESULTS / "us_diabetes_130_predictions.csv", RESULTS / "us_diabetes_130_confusion.csv"],
    },
    "mimic_iv_demo": {
        "id": "mimic_iv_demo",
        "name": "MIMIC-IV Demo",
        "setting": "Demo hospital records",
        "purpose": "Estimate in-hospital mortality from the information available at hospital admission.",
        "target": "In-hospital mortality",
        "input_boundary": "Only admission demographic and administrative fields are used; post-admission care and outcome fields are excluded.",
        "state_definition": "Admission demographic and administrative summary",
        "trajectory_note": "This demo source is evaluated as one admission at a time. It does not contain a clinical time series for a disease-trajectory chart.",
        "feature_groups": [
            {"name": "Demographic field", "items": ["Age", "Gender"]},
            {"name": "Admission context", "items": ["Admission type", "Admission location"]},
            {"name": "Administrative context", "items": ["Insurance", "Language", "Marital status", "Race"]},
        ],
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
