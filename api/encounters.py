"""Read evaluated encounter-level examples for the non-CGM datasets."""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from .catalog import RESULTS


MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "histogram_gradient_boosting": "Histogram gradient boosting",
}


def _path(dataset_id: str, suffix: str):
    return RESULTS / f"{dataset_id}_{suffix}.csv"


@lru_cache(maxsize=6)
def _table(dataset_id: str, suffix: str) -> pd.DataFrame:
    return pd.read_csv(_path(dataset_id, suffix))


def overview(dataset_id: str) -> dict[str, object]:
    metrics = _table(dataset_id, "metrics").copy()
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS)
    return {"kind": "encounter", "accuracy": metrics.to_dict(orient="records")}


def record_list(dataset_id: str, model: str) -> list[dict[str, str]]:
    rows = _table(dataset_id, "predictions")
    rows = rows.loc[rows["model"] == model, ["record_id", "patient_id"]].head(300)
    return rows.to_dict(orient="records")


def record_state(dataset_id: str, record_id: str, model: str) -> dict[str, object]:
    rows = _table(dataset_id, "predictions")
    row = rows.loc[(rows["model"] == model) & (rows["record_id"].astype(str) == str(record_id))]
    if row.empty:
        raise KeyError("The selected held-out record is unavailable for this model.")
    item = row.iloc[0]
    matrix = _table(dataset_id, "confusion")
    matrix = matrix.loc[matrix["model"] == model]
    return {
        "record_id": str(item["record_id"]),
        "patient_id": str(item["patient_id"]),
        "state": json.loads(item["state"]),
        "predicted_probability": float(item["predicted_probability"]),
        "predicted_class": int(item["predicted_class"]),
        "observed_outcome": int(item["target"]),
        "confusion": matrix.to_dict(orient="records"),
    }
