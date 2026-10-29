"""Read-only access to evaluated Shanghai T2D forecasts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .catalog import RESULTS


MODELS = ["persistence", "ridge", "random_forest", "gradient_boosting"]
MODEL_LABELS = {
    "persistence": "Persistence baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Histogram gradient boosting",
}
HORIZONS = [15, 30, 60]


@lru_cache(maxsize=1)
def metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "metrics.csv")


@lru_cache(maxsize=1)
def patient_metrics() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "patient_metrics.csv", dtype={"subject_id": str})
    frame["subject_id"] = frame["subject_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def zone_confusion() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "zone_confusion.csv")


@lru_cache(maxsize=32)
def patient_predictions(subject_id: str) -> pd.DataFrame:
    columns = [
        "subject_id", "episode_id", "timestamp", "target_timestamp", "horizon_minutes",
        "target_cgm", "current_cgm", "model", "prediction",
    ]
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RESULTS / "predictions.csv", usecols=columns,
        dtype={"subject_id": str, "episode_id": str, "model": str}, chunksize=250_000,
    ):
        selected = chunk.loc[(chunk["subject_id"] == subject_id) & chunk["model"].isin(MODELS)]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(matches, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"])
    return frame.sort_values("timestamp")


def participant_list() -> list[str]:
    return sorted(patient_metrics()["subject_id"].unique().tolist())


def episodes(subject_id: str) -> list[str]:
    return sorted(patient_predictions(subject_id)["episode_id"].dropna().unique().tolist())


def overview() -> dict[str, object]:
    mae = metrics().loc[lambda value: value["metric"].eq("mae")].copy()
    mae["model_label"] = mae["model"].map(MODEL_LABELS)
    return {
        "accuracy": mae[["model", "model_label", "horizon_minutes", "estimate", "ci_2_5", "ci_97_5", "n_subjects"]].to_dict("records"),
        "model_labels": MODEL_LABELS,
    }


def state_and_forecast(subject_id: str, episode_id: str, model: str, origin: str | None) -> dict[str, object]:
    if model not in MODELS:
        raise ValueError("Unknown model")
    frame = patient_predictions(subject_id)
    frame = frame.loc[(frame["episode_id"] == episode_id) & (frame["model"] == model)].copy()
    if frame.empty:
        raise ValueError("No held-out forecasts for this participant and CGM recording")
    state_source = frame.loc[frame["horizon_minutes"] == 15].drop_duplicates("timestamp").sort_values("timestamp")
    latest = state_source.tail(24)
    current = float(latest["current_cgm"].iloc[-1])
    earlier = float(latest["current_cgm"].iloc[max(0, len(latest) - 5)])
    change = current - earlier
    direction = "rising" if change > 10 else "falling" if change < -10 else "stable"
    valid_origins = (
        frame.groupby("timestamp", observed=True)["horizon_minutes"].nunique()
        .loc[lambda value: value == len(HORIZONS)].index.sort_values()
    )
    selected_origin = pd.Timestamp(origin) if origin else valid_origins[-1]
    forecast = frame.loc[frame["timestamp"] == selected_origin].set_index("horizon_minutes").reindex(HORIZONS).reset_index()
    history = state_source.loc[state_source["timestamp"] <= selected_origin].tail(24)
    return {
        "state": {
            "last_cgm_mg_dl": current,
            "one_hour_change_mg_dl": change,
            "six_hour_mean_mg_dl": float(latest["current_cgm"].mean()),
            "six_hour_variation_mg_dl": float(latest["current_cgm"].std(ddof=0)),
            "direction": direction,
        },
        "origins": [timestamp.isoformat() for timestamp in valid_origins],
        "selected_origin": selected_origin.isoformat(),
        "history": [{"time": row.timestamp.isoformat(), "cgm": float(row.current_cgm)} for row in history.itertuples()],
        "forecast": [
            {
                "minutes_ahead": int(row.horizon_minutes),
                "time": row.target_timestamp.isoformat(),
                "forecast_cgm": float(row.prediction),
                "measured_later_cgm": float(row.target_cgm),
                "absolute_error": float(abs(row.prediction - row.target_cgm)),
            }
            for row in forecast.itertuples()
        ],
    }


def participant_error(model: str) -> list[dict[str, object]]:
    if model not in MODELS:
        raise ValueError("Unknown model")
    frame = patient_metrics().loc[lambda value: value["model"].eq(model)]
    return frame[["subject_id", "horizon_minutes", "mae"]].to_dict("records")


def range_agreement(model: str, horizon: int) -> list[dict[str, object]]:
    frame = zone_confusion().loc[
        lambda value: value["model"].eq(model) & value["horizon_minutes"].eq(horizon)
    ]
    return frame.to_dict("records")
