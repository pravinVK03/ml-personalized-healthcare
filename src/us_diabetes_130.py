"""Diabetes 130-US Hospitals: independent 30-day readmission study."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from encounter_common import EncounterStudy, SEED, find_file, model_pipeline


def load_study(root):
    frame = pd.read_csv(find_file(root, "diabetic_data.csv"), na_values="?", low_memory=False)
    numeric = ("time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses")
    categorical = ("race", "gender", "age", "admission_type_id", "admission_source_id", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed")
    selected = frame.loc[frame["readmitted"].notna(), ["patient_nbr", "readmitted", *numeric, *categorical]].copy()
    selected["target"] = (selected["readmitted"] == "<30").astype(int)
    return EncounterStudy("us_diabetes_130", "Diabetes 130-US Hospitals", "Readmission within 30 days", "Index hospital-encounter summary at discharge", "patient_nbr", (*numeric, *categorical), numeric, selected)


def models(study):
    # Model: Logistic Regression
    logistic = model_pipeline(study, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    # Model: Random Forest
    forest = model_pipeline(study, RandomForestClassifier(n_estimators=180, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
    # Model: Histogram Gradient Boosting
    boosting = model_pipeline(study, HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, l2_regularization=1.0, random_state=SEED), compact_categoricals=True)
    return {"logistic_regression": logistic, "random_forest": forest, "histogram_gradient_boosting": boosting}
