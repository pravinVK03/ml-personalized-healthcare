# ShanghaiT2DM CGM forecasting

A clear, research-only implementation for short-term continuous glucose monitoring (CGM) forecasting in the supplied ShanghaiT2DM data. It is deliberately small, uses standard scikit-learn models, and keeps patients separate between training and evaluation.

It forecasts CGM **15, 30, and 60 minutes ahead**. The dataset does not provide enough long-term repeat follow-up for next-HbA1c, medication-response, or long-term disease-progression claims.

## Run

From this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\run_basic.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open `http://localhost:8502` if it does not open automatically. The first command creates `results/`; the second command displays it.

To recreate the environment on another computer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## What the implementation does

1. Reads the supplied CGM episode files from `Shangai Data/`.
2. Keeps only continuous 15-minute CGM periods and builds a six-hour past-only history for each prediction time.
3. Splits data by **patient** into five folds. A patient is never in both training and test data.
4. Fits four understandable models: persistence, ridge regression, random forest, and histogram gradient boosting.
5. Saves only the held-out predictions, per-patient metrics, overall accuracy, and glucose-band counts used by the dashboard.

## Models

| Model | Role |
| --- | --- |
| Persistence | Simple baseline: future CGM equals the latest measurement. |
| Ridge regression | Regularised linear model using CGM history and simple time/rolling features. |
| Random forest | Standard nonlinear tree model. |
| Histogram gradient boosting | Standard nonlinear tree model and selected primary model when it has lowest held-out MAE. |

No neural network or ensemble is included. The four models already cover baseline, linear, and nonlinear methods; adding more complexity would make the project harder to explain without a pre-specified, validated benefit.

## Dashboard

The Streamlit dashboard has five concise sections:

- **Current state:** latest measured CGM history for one research participant and episode.
- **Forecast:** a selected saved held-out forecast with its later observed value.
- **Accuracy:** patient-held-out MAE and 95% patient-bootstrap intervals.
- **Patient heatmap:** per-patient MAE across the three forecast horizons.
- **Confusion matrix:** descriptive observed-versus-predicted glucose bands.

“Research Digital Twin” is a UI label for a patient's measured history and model forecast. It is not a physiological simulation, a medical device, or clinical decision support.

## Folder structure

```text
app.py                     Streamlit dashboard
configs/study.json         Fixed study settings
scripts/run_basic.py       One command that builds and evaluates the models
src/shanghai_t2d/          Data, features, models, evaluation, and plots
Shangai Data/              Supplied dataset (unchanged)
results/                   Generated study outputs used by the app
tests/                     Small data and leakage-control tests
docs/                      Final project report and implementation guide
```

`healthtwin/` is a separate existing project and has deliberately not been modified.

The only active files in `results/` are:

- `predictions.csv` — held-out forecasts used for the patient view.
- `patient_metrics.csv` — MAE for each patient.
- `metrics.csv` — overall accuracy and confidence intervals.
- `zone_confusion.csv` — counts for the glucose-band confusion matrix.

## Result interpretation

The primary number is the **60-minute subject-macro MAE** in mg/dL: lower is better. “Subject-macro” means each of the 100 held-out patients receives equal weight, rather than allowing people with longer recordings to dominate. The reported confidence intervals resample patients, not individual CGM rows.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The saved outputs are research evidence only. Do not use them to recommend insulin, medication, diet, or treatment changes.
