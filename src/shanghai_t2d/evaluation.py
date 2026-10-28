"""Patient-held-out split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_subject_folds(subjects, n_folds: int, seed: int) -> dict[str, int]:
    """Assign every patient to exactly one deterministic outer-validation fold."""
    unique = np.array(sorted({str(subject) for subject in subjects}))
    if len(unique) < n_folds:
        raise ValueError("The number of folds exceeds the number of patients.")
    random = np.random.default_rng(seed)
    random.shuffle(unique)
    return {subject: index % n_folds for index, subject in enumerate(unique)}


def assert_no_patient_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    overlap = set(train["subject_id"]) & set(test["subject_id"])
    if overlap:
        raise AssertionError(f"Patient leakage detected: {sorted(overlap)}")
