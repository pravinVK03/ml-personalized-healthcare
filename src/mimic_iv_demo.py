"""MIMIC-IV Demo: independent admission-time mortality feasibility study."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from encounter_common import EncounterStudy, SEED, find_file, model_pipeline


def load_study(root):
    admissions = pd.read_csv(find_file(root, "admissions.csv.gz"), compression="gzip")
    patients = pd.read_csv(find_file(root, "patients.csv.gz"), compression="gzip")
    diagnoses = pd.read_csv(find_file(root, "diagnoses_icd.csv.gz"), compression="gzip", dtype={"icd_code": str})
    # Cohort definition: documented Type 2 diabetes diagnosis.  The code is
    # used only to identify the retrospective cohort, never as a model feature.
    type_2 = diagnoses.loc[
        ((diagnoses["icd_version"] == 10) & diagnoses["icd_code"].str.startswith("E11", na=False))
        | ((diagnoses["icd_version"] == 9) & diagnoses["icd_code"].str.match(r"^250..[02]$", na=False)),
        "hadm_id",
    ].drop_duplicates()
    frame = admissions.loc[admissions["hadm_id"].isin(type_2)].merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )
    numeric = ("anchor_age",)
    categorical = ("gender", "admission_type", "admission_location", "insurance", "language", "marital_status", "race")
    selected = frame.loc[:, ["subject_id", "hospital_expire_flag", *numeric, *categorical]].copy()
    selected["target"] = selected["hospital_expire_flag"].astype(int)
    return EncounterStudy("mimic_iv_demo", "MIMIC-IV Demo T2D Cohort", "In-hospital mortality", "Admission demographic and administrative summary", "subject_id", (*numeric, *categorical), numeric, selected)


def models(study):
    # Model: Logistic Regression
    logistic = model_pipeline(study, LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    # Model: Random Forest
    forest = model_pipeline(study, RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
    # Model: Histogram Gradient Boosting
    boosting = model_pipeline(study, HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, l2_regularization=1.0, random_state=SEED), compact_categoricals=True)
    return {"logistic_regression": logistic, "random_forest": forest, "histogram_gradient_boosting": boosting}
