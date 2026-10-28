from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import StudyConfig
from .evaluation import assert_no_patient_leakage, make_subject_folds
from .features import feature_sets
from .metrics import aggregate_metrics, per_subject_metrics
from .models import fit_hist_gradient_boosting


BASIC_MODEL_ORDER = ["persistence", "ridge", "random_forest", "gradient_boosting"]
GLUCOSE_BANDS = ["Below 70", "70 to 180", "Above 180"]


def _fit_standard_ridge(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, solver="lsqr", max_iter=5000)),
        ]
    )
    model.fit(x, y)
    return model


def _fit_random_forest(x: pd.DataFrame, y: pd.Series, *, random_seed: int) -> Pipeline:
    """Fit a bounded standard random forest suitable for the available CGM sample size."""
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=18,
                    min_samples_leaf=30,
                    max_features=0.6,
                    max_samples=0.4,
                    n_jobs=-1,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def _glucose_band(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 70.0, 180.0, np.inf],
        labels=GLUCOSE_BANDS,
        right=False,
    )


def glucose_band_confusion(predictions: pd.DataFrame) -> pd.DataFrame:
    """Observed-versus-predicted glucose bands for visual interpretation only."""
    rows: list[dict[str, object]] = []
    for (model, horizon), frame in predictions.groupby(["model", "horizon_minutes"], observed=True):
        actual = _glucose_band(frame["target_cgm"])
        predicted = _glucose_band(frame["prediction"])
        matrix = pd.crosstab(actual, predicted, dropna=False).reindex(
            index=GLUCOSE_BANDS, columns=GLUCOSE_BANDS, fill_value=0
        )
        for actual_band in GLUCOSE_BANDS:
            for predicted_band in GLUCOSE_BANDS:
                rows.append(
                    {
                        "model": str(model),
                        "horizon_minutes": int(horizon),
                        "actual_band": actual_band,
                        "predicted_band": predicted_band,
                        "count": int(matrix.loc[actual_band, predicted_band]),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_basic(
    supervised: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Evaluate four standard models with patient-grouped outer validation."""
    output_dir = Path(output_dir)
    history_columns = feature_sets(supervised.columns)["history"]
    fold_map = make_subject_folds(
        supervised["subject_id"], config.outer_folds, config.random_seed
    )
    prediction_frames: list[pd.DataFrame] = []
    for horizon_steps in config.forecast_horizons_steps:
        horizon = horizon_steps * config.sampling_minutes
        frame = supervised.loc[supervised["horizon_steps"] == horizon_steps].copy()
        frame["subject_id"] = frame["subject_id"].astype(str)
        frame["episode_id"] = frame["episode_id"].astype(str)
        frame["fold"] = frame["subject_id"].map(fold_map)
        for fold in range(config.outer_folds):
            train = frame.loc[frame["fold"] != fold]
            test = frame.loc[frame["fold"] == fold]
            assert_no_patient_leakage(train, test)
            base = test[
                [
                    "subject_id",
                    "episode_id",
                    "timestamp",
                    "target_timestamp",
                    "horizon_steps",
                    "horizon_minutes",
                    "target_cgm",
                    "current_cgm",
                    "fold",
                ]
            ].copy()
            persistence = base.copy()
            persistence["model"] = "persistence"
            persistence["prediction"] = persistence["current_cgm"]
            prediction_frames.append(persistence)

            ridge = _fit_standard_ridge(train[history_columns], train["target_cgm"])
            ridge_result = base.copy()
            ridge_result["model"] = "ridge"
            ridge_result["prediction"] = ridge.predict(test[history_columns])
            prediction_frames.append(ridge_result)

            forest = _fit_random_forest(
                train[history_columns], train["target_cgm"], random_seed=config.random_seed + fold
            )
            forest_result = base.copy()
            forest_result["model"] = "random_forest"
            forest_result["prediction"] = forest.predict(test[history_columns])
            prediction_frames.append(forest_result)

            hgb = fit_hist_gradient_boosting(
                train[history_columns],
                train["target_cgm"],
                random_seed=config.random_seed + fold,
            )
            hgb_result = base.copy()
            hgb_result["model"] = "gradient_boosting"
            hgb_result["prediction"] = hgb.predict(test[history_columns])
            prediction_frames.append(hgb_result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["model"] = pd.Categorical(
        predictions["model"], categories=BASIC_MODEL_ORDER, ordered=True
    )
    predictions = predictions.sort_values(
        ["horizon_minutes", "model", "subject_id", "timestamp"]
    ).reset_index(drop=True)
    patient = per_subject_metrics(predictions)
    aggregate = aggregate_metrics(
        patient,
        bootstrap_iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )
    zone_confusion = glucose_band_confusion(predictions)
    for name, value in {
        "predictions.csv": predictions,
        "patient_metrics.csv": patient,
        "metrics.csv": aggregate,
        "zone_confusion.csv": zone_confusion,
    }.items():
        value.to_csv(output_dir / name, index=False)
    return {
        "predictions": predictions,
        "patient_metrics": patient,
        "metrics": aggregate,
        "zone_confusion": zone_confusion,
    }
