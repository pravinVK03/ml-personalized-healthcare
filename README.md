# Multi Dataset Patient State Research Framework

This is a grounded, retrospective research prototype for patient-state modelling and short-term outcome forecasting. It provides a React interface and a Python API. It is not a clinical system, a physiological simulation, or treatment decision support.

## What is implemented

The interface switches among three datasets without pooling them:

| Dataset | Supported unit and target | Local status |
| --- | --- | --- |
| Shanghai T2D | CGM value 15, 30, or 60 minutes ahead | Ready when `results/` has been generated. |
| Diabetes 130-US Hospitals | Future encounter outcome, such as readmission | Disabled until `data/us_diabetes_130/diabetic_data.csv` is supplied and audited. |
| MIMIC-IV Demo | Documented encounter-level outcome | Disabled until compatible demo tables are supplied under `data/mimic_iv_demo/`. |

The other datasets are deliberately unavailable until their data and target definitions are audited. The application never invents an accuracy score, forecast, or patient record for missing data.

## Patient state representation

For Shanghai T2D, the displayed patient state is a transparent summary of measurements available before the selected prediction time:

- Last recorded CGM value.
- Change over the previous hour.
- Mean and variation over the previous six hours.
- Direction over the previous hour.

This is a computational state representation for research. It is not a biological or clinical digital twin.

## Models

The Shanghai pipeline uses standard scikit-learn models: persistence, ridge regression, random forest, and histogram gradient boosting. Evaluation holds out entire patients. The primary accuracy measure is subject-macro mean absolute error (MAE) in mg/dL.

No RNN is included. The current datasets do not share a validated longitudinal target or comparable time scale, so an RNN would add complexity without a fair, supported cross-dataset evaluation.

## Run Shanghai preprocessing

The supplied Shanghai data stay local and are not committed to Git.

```powershell
.\.venv\Scripts\python.exe scripts\run_basic.py
```

This writes four private local result tables required by the React interface: held-out forecasts, aggregate metrics, participant metrics, and glucose-range agreement counts.

## Run the React application

Install frontend packages once:

```powershell
npm --prefix frontend install
npm --prefix frontend run build
```

Start the API in one terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_react_api.py
```

Then open `http://127.0.0.1:8503`.

For frontend development, run `npm --prefix frontend run dev` and open `http://127.0.0.1:5173`. The Vite development server proxies API calls to port 8503.

## Folder structure

```text
api/                       Python API and dataset catalogue
frontend/                  React interface
src/shanghai_t2d/          Shanghai preprocessing and standard models
scripts/run_basic.py       Shanghai evaluation runner
scripts/run_react_api.py   React API runner
configs/study.json         Fixed Shanghai evaluation settings
tests/                     Leakage and metric tests
docs/                      Research report and implementation guide
Shangai Data/              Local supplied data; excluded from Git
results/                   Local evaluated outputs; excluded from Git
```

## Limits

- Dataset-specific units, timelines, features, and outcomes are not automatically comparable or pooled.
- Diabetes 130-US Hospitals has categorical HbA1c data; this project must not claim numerical HbA1c trajectories from it.
- MIMIC-IV Demo is suitable for pipeline feasibility work, not generalizable clinical-performance claims.
- Any new dataset needs a target definition, leakage audit, patient-level split strategy, and documented preprocessing before its results can be enabled.
