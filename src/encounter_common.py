"""Shared mechanics for the two separate encounter-level studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SEED = 20261025


@dataclass(frozen=True)
class EncounterStudy:
    dataset_id: str
    display_name: str
    target_label: str
    state_label: str
    patient_column: str
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    raw: pd.DataFrame


def find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    return matches[0]


def preprocessor(study: EncounterStudy, *, compact_categoricals: bool = False) -> ColumnTransformer:
    categorical = [column for column in study.feature_columns if column not in study.numeric_columns]
    encoder = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        if compact_categoricals
        else OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    )
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(study.numeric_columns)),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]), categorical),
    ])


def model_pipeline(study: EncounterStudy, model: object, *, compact_categoricals: bool = False) -> Pipeline:
    return Pipeline([("preprocess", preprocessor(study, compact_categoricals=compact_categoricals)), ("model", model)])


def evaluate_models(study: EncounterStudy, models: dict[str, Pipeline], output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use one reproducible patient-held-out split and write small result tables."""
    frame = study.raw.reset_index(drop=True)
    train_index, test_index = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(frame, frame["target"], groups=frame[study.patient_column]))
    train, test = frame.iloc[train_index].copy(), frame.iloc[test_index].copy()
    x_train, y_train = train.loc[:, study.feature_columns], train["target"]
    x_test, y_test = test.loc[:, study.feature_columns], test["target"]
    metric_rows, prediction_rows, confusion_rows = [], [], []

    for name, pipeline in models.items():
        fit_kwargs: dict[str, object] = {}
        if name == "histogram_gradient_boosting":
            counts = y_train.value_counts()
            fit_kwargs["model__sample_weight"] = y_train.map({label: len(y_train) / (2 * count) for label, count in counts.items()}).to_numpy()
        pipeline.fit(x_train, y_train, **fit_kwargs)
        probability = pipeline.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        metric_rows.append({
            "model": name,
            "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "average_precision": average_precision_score(y_test, probability),
            "n_train": len(train), "n_test": len(test),
            "n_test_patients": test[study.patient_column].nunique(),
            "positive_rate_test": float(y_test.mean()),
        })
        state = test.loc[:, study.feature_columns].astype(object).where(test.loc[:, study.feature_columns].notna(), None)
        prediction_rows.append(pd.DataFrame({
            "model": name, "patient_id": test[study.patient_column].astype(str).to_numpy(),
            "record_id": test.index.astype(str), "target": y_test.to_numpy(),
            "predicted_probability": probability, "predicted_class": prediction,
            "state": [json.dumps(row, default=str) for row in state.to_dict(orient="records")],
        }))
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        confusion_rows.extend({"model": name, "actual": actual, "predicted": predicted, "count": int(matrix[actual, predicted])} for actual in (0, 1) for predicted in (0, 1))

    metrics, predictions, confusion = pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(confusion_rows)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / f"{study.dataset_id}_metrics.csv", index=False)
    predictions.to_csv(output / f"{study.dataset_id}_predictions.csv", index=False)
    confusion.to_csv(output / f"{study.dataset_id}_confusion.csv", index=False)
    return metrics, predictions, confusion
