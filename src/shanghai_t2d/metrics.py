from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    safe = np.abs(y_true) >= 40.0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(error)),
        "mard_percent": float(np.mean(np.abs(error[safe] / y_true[safe])) * 100)
        if safe.any()
        else np.nan,
        "within_15_percent": float(np.mean(np.abs(error) <= 15.0) * 100),
        "within_30_percent": float(np.mean(np.abs(error) <= 30.0) * 100),
    }


def per_subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["model", "horizon_minutes", "subject_id"]
    for (model, horizon, subject), frame in predictions.groupby(keys, sort=True):
        row: dict[str, object] = {
            "model": model,
            "horizon_minutes": int(horizon),
            "subject_id": str(subject),
            "n_predictions": int(len(frame)),
        }
        row.update(regression_metrics(frame["target_cgm"].to_numpy(), frame["prediction"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate_metrics(
    patient_metrics: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = [
        "mae",
        "rmse",
        "r2",
        "bias",
        "mard_percent",
        "within_15_percent",
        "within_30_percent",
    ]
    for (model, horizon), frame in patient_metrics.groupby(
        ["model", "horizon_minutes"], sort=True
    ):
        for metric_index, metric in enumerate(metric_names):
            values = frame[metric].to_numpy(dtype=float)
            lower, upper = _bootstrap_mean_ci(
                values,
                bootstrap_iterations,
                seed + int(horizon) * 100 + metric_index,
            )
            rows.append(
                {
                    "model": model,
                    "horizon_minutes": int(horizon),
                    "metric": metric,
                    "estimate": float(np.nanmean(values)),
                    "ci_2_5": lower,
                    "ci_97_5": upper,
                    "aggregation": "subject_macro",
                    "n_subjects": int(frame["subject_id"].nunique()),
                    "n_predictions": int(frame["n_predictions"].sum()),
                }
            )
    return pd.DataFrame(rows)

