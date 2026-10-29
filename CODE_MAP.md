# Code Map for Presentation

This project deliberately keeps each dataset pipeline separate. The React interface reads the evaluated results through one API, so the presentation is integrated without mixing different clinical questions.

## The three independent studies

| Dataset | Code to show | Prediction question | Models |
|---|---|---|---|
| Shanghai T2D | `src/shanghai_t2d/` | What will CGM glucose be in the next 30, 60, or 120 minutes? | Persistence, Ridge regression, Random forest, Histogram gradient boosting |
| Diabetes 130-US Hospitals | `src/us_diabetes_130.py` | Will this diabetes encounter be readmitted within 30 days? (diabetes type unavailable) | Logistic regression, Random forest, Histogram gradient boosting |
| MIMIC-IV Demo T2D Cohort | `src/mimic_iv_demo.py` | Will an ICD-confirmed T2D admission end in in-hospital mortality? | Logistic regression, Random forest, Histogram gradient boosting |

## Shared integration

```text
separate dataset module → held-out evaluation CSV files → FastAPI routes → React dashboard
```

- `src/encounter_common.py` contains only the shared, repeatable encounter preprocessing and evaluation helpers for the US and MIMIC studies.
- `api/catalog.py` describes each dataset, its prediction target, feature boundary, and limitation.
- `api/shanghai.py` and `api/encounters.py` expose read-only evaluated patient examples.
- `frontend/src/App.jsx` integrates all three studies into one dashboard; it does not train models or mix their outcomes.

## A simple explanation to use in a presentation

“We build a retrospective patient-state representation from information that was available at a defined time. We train standard scikit-learn baseline and ensemble models separately for each dataset. We keep all records from a patient on one side of the evaluation split to reduce patient leakage. The UI lets us inspect one held-out example and the full held-out model results. It is a research demonstration, not a clinical decision tool.”

## Important wording

- Call the displayed profile a **research patient digital twin**: a compact representation of available patient information for a specific prediction task.
- Say **held-out evaluation**, rather than real-time prediction.
- The MIMIC-IV Demo T2D cohort is very small (60 admissions, 17 patients), so its metrics are a feasibility demonstration and not evidence of deployable clinical performance.
