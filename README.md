# Multi Dataset Patient State Research Framework

This is a grounded, retrospective research prototype for patient-state modelling and short-term outcome forecasting. It provides a React interface and a Python API. It is not a clinical system, a physiological simulation, or treatment decision support.

## What is implemented

The interface switches among three datasets without pooling them:

| Dataset | Supported unit and target | Local status |
| --- | --- | --- |
| Shanghai T2D | CGM value 15, 30, or 60 minutes ahead | Ready when `results/` has been generated. |
| Diabetes 130-US Hospitals | Readmission within 30 days | Ready: a separate patient-held-out encounter evaluation. |
| MIMIC-IV Demo | In-hospital mortality | Ready: a separate admission-time feasibility evaluation. |

The datasets are never pooled. The two hospital datasets are not CGM trajectories: they show a documented encounter state and a held-out binary-outcome evaluation, while Shanghai T2D shows a short-term CGM forecast.

## Patient state representation

For Shanghai T2D, the displayed patient state is a transparent summary of measurements available before the selected prediction time:

- Last recorded CGM value.
- Change over the previous hour.
- Mean and variation over the previous six hours.
- Direction over the previous hour.

This is a computational state representation for research. It is not a biological or clinical digital twin.

## Models

The Shanghai pipeline uses standard scikit-learn models: persistence, ridge regression, random forest, and histogram gradient boosting. Evaluation holds out entire patients. The primary accuracy measure is subject-macro mean absolute error (MAE) in mg/dL.

No RNN is included. The sources do not share a validated longitudinal target or comparable time scale, so an RNN would add complexity without a fair, supported cross-dataset evaluation.

## Run Shanghai preprocessing

The supplied Shanghai data stay local and are not committed to Git.

```powershell
.\.venv\Scripts\python.exe scripts\run_basic.py
```

This writes four private local result tables required by the React interface: held-out forecasts, aggregate metrics, participant metrics, and glucose-range agreement counts.

## Run the hospital-dataset evaluations

```powershell
.\.venv\Scripts\python.exe scripts\run_encounter_evaluations.py
```

This runs independent patient-held-out studies and writes three result tables for each source: metrics, held-out record predictions, and a confusion matrix. The source modules are intentionally separate for presentation clarity:

- `src/us_diabetes_130.py` — 30-day readmission; model sections are marked with `# Model:` comments.
- `src/mimic_iv_demo.py` — in-hospital mortality; model sections are marked with `# Model:` comments.

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

If port 8503 is already in use, choose another local port:

```powershell
.\.venv\Scripts\python.exe scripts\run_react_api.py --port 8504
```

For frontend development, run `npm --prefix frontend run dev` and open `http://127.0.0.1:5173`. The Vite development server proxies API calls to port 8503.

## Folder structure

```text
api/                       Python API and dataset catalogue
frontend/                  React interface
src/shanghai_t2d/          Shanghai preprocessing and standard models
src/us_diabetes_130.py     US hospital readmission pipeline
src/mimic_iv_demo.py       MIMIC-IV Demo mortality pipeline
src/encounter_common.py    Shared patient-held-out evaluation mechanics
scripts/run_basic.py       Shanghai evaluation runner
scripts/run_react_api.py   React API runner
configs/study.json         Fixed Shanghai evaluation settings
tests/                     Leakage and metric tests
docs/                      Research report and implementation guide
data/shanghai_t2d/         Local Shanghai T2D data; excluded from Git
data/us_diabetes_130/      Location for Diabetes 130-US Hospitals data
data/mimic_iv_demo/        Location for MIMIC-IV Demo data
results/                   Local evaluated outputs; excluded from Git
```

## Limits

- Dataset-specific units, timelines, features, and outcomes are not automatically comparable or pooled.
- Diabetes 130-US Hospitals has categorical HbA1c data; this project must not claim numerical HbA1c trajectories from it.
- MIMIC-IV Demo is suitable for pipeline feasibility work, not generalizable clinical-performance claims.
- Any new dataset needs a target definition, leakage audit, patient-level split strategy, and documented preprocessing before its results can be enabled.
