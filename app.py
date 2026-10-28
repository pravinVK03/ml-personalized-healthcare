"""Clear research dashboard for the simplified ShanghaiT2DM implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ["persistence", "ridge", "random_forest", "gradient_boosting"]
LABELS = {
    "persistence": "Persistence baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Histogram gradient boosting",
}
HORIZONS = [15, 30, 60]

st.set_page_config(page_title="ShanghaiT2DM Glucose Forecasting", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: str, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype)


@st.cache_data(show_spinner="Loading one patient's held-out forecasts...")
def load_patient_predictions(path: str, subject_id: str) -> pd.DataFrame:
    """Load only one participant from the large out-of-fold prediction file."""
    columns = [
        "subject_id", "episode_id", "timestamp", "target_timestamp", "horizon_minutes",
        "target_cgm", "current_cgm", "model", "prediction",
    ]
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"subject_id": str, "episode_id": str, "model": str},
        chunksize=250_000,
    ):
        selected = chunk.loc[
            (chunk["subject_id"] == str(subject_id)) & chunk["model"].isin(MODELS)
        ]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        return pd.DataFrame(columns=columns)
    result = pd.concat(matches, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["target_timestamp"] = pd.to_datetime(result["target_timestamp"])
    return result


def recent_state(frame: pd.DataFrame) -> dict[str, float | str]:
    values = frame.sort_values("timestamp").drop_duplicates("timestamp").tail(24)
    current = float(values["current_cgm"].iloc[-1])
    earlier = float(values["current_cgm"].iloc[max(0, len(values) - 5)])
    change = current - earlier
    trend = "Rising" if change > 10 else "Falling" if change < -10 else "Stable"
    return {
        "current": current,
        "change": change,
        "mean": float(values["current_cgm"].mean()),
        "variation": float(values["current_cgm"].std(ddof=0)),
        "trend": trend,
    }


def heatmap(frame: pd.DataFrame, title: str, value_label: str, fmt: str = ".1f") -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    sns.heatmap(frame, annot=True, fmt=fmt, cmap="Blues", linewidths=0.5, ax=axis,
                cbar_kws={"label": value_label})
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


required = [RESULTS / name for name in ("metrics.csv", "patient_metrics.csv", "predictions.csv", "zone_confusion.csv")]
st.title("ShanghaiT2DM Glucose Forecasting")
st.caption(
    "A retrospective research dashboard for evaluating short-term CGM forecasts. "
    "It is not a medical device or clinical decision tool."
)
if not all(path.exists() for path in required):
    st.error("Results are not available yet. Run: .\\.venv\\Scripts\\python.exe scripts\\run_basic.py")
    st.stop()

metrics = load_csv(str(RESULTS / "metrics.csv"))
patient_metrics = load_csv(str(RESULTS / "patient_metrics.csv"), {"subject_id": str})
zone_confusion = load_csv(str(RESULTS / "zone_confusion.csv"))
patient_metrics["subject_id"] = patient_metrics["subject_id"].astype(str)

state_tab, forecast_tab, accuracy_tab, patient_heatmap_tab, confusion_tab = st.tabs(
    ["Recorded CGM", "Forecast evaluation", "Accuracy", "Participant error", "Range agreement"]
)

subjects = sorted(patient_metrics["subject_id"].unique())
with st.sidebar:
    st.header("Select study data")
    subject = st.selectbox("Study participant", subjects)
    default_model = "gradient_boosting" if "gradient_boosting" in MODELS else MODELS[0]
    model = st.selectbox("Model", MODELS, index=MODELS.index(default_model), format_func=LABELS.get)

predictions = load_patient_predictions(str(RESULTS / "predictions.csv"), subject)
episodes = sorted(predictions["episode_id"].dropna().unique())
if not episodes:
    st.error("The selected patient has no held-out prediction records.")
    st.stop()
with st.sidebar:
    episode = st.selectbox("CGM recording", episodes)

episode_predictions = predictions.loc[
    (predictions["episode_id"] == episode) & (predictions["model"] == model)
].sort_values("timestamp")

with state_tab:
    st.subheader("Latest recorded CGM values")
    st.caption("This section shows measurements already present in the selected CGM recording. It is not a live patient monitor.")
    state_source = episode_predictions.loc[episode_predictions["horizon_minutes"] == 15]
    state = recent_state(state_source)
    one, two, three, four, five = st.columns(5)
    one.metric("Last CGM measurement", f"{state['current']:.1f} mg/dL")
    two.metric("Change over previous hour", f"{state['change']:+.1f} mg/dL")
    three.metric("Mean over last six hours", f"{state['mean']:.1f} mg/dL")
    four.metric("Variation over last six hours", f"{state['variation']:.1f} mg/dL")
    five.metric("Direction over previous hour", str(state["trend"]))
    history = state_source.drop_duplicates("timestamp").tail(24).set_index("timestamp")
    st.line_chart(history["current_cgm"], x_label="Measurement time", y_label="Recorded CGM (mg/dL)")
    st.info("All values in this section are historical study measurements. They are shown only to explain the model input history.")

with forecast_tab:
    st.subheader("Held-out forecast evaluation")
    st.caption("The model was not trained on this participant. Choose a prediction time to compare the saved forecast with the CGM value measured later.")
    common_origins = (
        episode_predictions.groupby("timestamp", observed=True)["horizon_minutes"].nunique()
        .loc[lambda value: value == len(HORIZONS)]
        .index.sort_values()
    )
    origin = st.select_slider(
        "Prediction time", options=list(common_origins), value=common_origins[-1],
        format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y %H:%M"),
    )
    forecast = episode_predictions.loc[episode_predictions["timestamp"] == origin].copy()
    forecast = forecast.set_index("horizon_minutes").reindex(HORIZONS).reset_index()
    past = episode_predictions.loc[
        (episode_predictions["horizon_minutes"] == 15) & (episode_predictions["timestamp"] <= origin)
    ].drop_duplicates("timestamp").tail(24)
    chart_history = past.rename(columns={"timestamp": "time", "current_cgm": "CGM history"})[["time", "CGM history"]]
    chart_forecast = forecast.rename(columns={"target_timestamp": "time", "prediction": "Model forecast"})[["time", "Model forecast"]]
    chart = chart_history.merge(chart_forecast, on="time", how="outer").sort_values("time").set_index("time")
    st.line_chart(chart, x_label="Time", y_label="CGM (mg/dL)")
    display = forecast[["horizon_minutes", "target_timestamp", "prediction", "target_cgm"]].copy()
    display["absolute_error"] = (display["prediction"] - display["target_cgm"]).abs()
    display.columns = ["Minutes ahead", "Later measurement time", "Forecast CGM", "Measured later CGM", "Absolute error"]
    st.dataframe(display.round({"Predicted CGM": 1, "Observed later CGM": 1, "Absolute error": 1}), hide_index=True, width="stretch")
    st.caption("The later measurement is displayed only to evaluate the saved forecast. It would not be available at the prediction time.")

with accuracy_tab:
    st.subheader("Overall held-out forecast accuracy")
    st.caption("MAE is average absolute error in mg/dL; lower is better. Every participant is held out of model training, and the interval estimates uncertainty across participants.")
    mae = metrics.loc[(metrics["metric"] == "mae") & metrics["model"].isin(MODELS)].copy()
    mae["Model"] = mae["model"].map(LABELS)
    chart = mae.pivot(index="horizon_minutes", columns="Model", values="estimate")
    st.line_chart(chart, x_label="Forecast horizon (minutes)", y_label="Subject-macro MAE (mg/dL)")
    table = mae[["Model", "horizon_minutes", "estimate", "ci_2_5", "ci_97_5", "n_subjects"]].rename(
        columns={"horizon_minutes": "Horizon (min)", "estimate": "MAE", "ci_2_5": "95% CI lower", "ci_97_5": "95% CI upper", "n_subjects": "Patients"}
    )
    st.dataframe(table.round(2), hide_index=True, width="stretch")
    best = mae.loc[mae["horizon_minutes"] == 60].sort_values("estimate").iloc[0]
    st.success(f"Best 60-minute result: {best['Model']} with MAE {best['estimate']:.2f} mg/dL.")

with patient_heatmap_tab:
    st.subheader("Error for each study participant")
    st.caption("Each row is one participant. Each cell is that participant's held-out MAE in mg/dL for the selected model; lower is better.")
    heatmap_model = st.selectbox("Model for participant error", MODELS, format_func=LABELS.get, key="patient_heatmap_model")
    individual = patient_metrics.loc[patient_metrics["model"] == heatmap_model]
    matrix = individual.pivot(index="subject_id", columns="horizon_minutes", values="mae").reindex(columns=HORIZONS)
    matrix.columns = [f"{column} min" for column in matrix.columns]
    heatmap(matrix, f"{LABELS[heatmap_model]}: held-out MAE by participant", "MAE (mg/dL)")

with confusion_tab:
    st.subheader("Forecast versus measured glucose range")
    st.caption("This table counts agreement between a forecast range and the CGM range measured later. It is descriptive only, not a diagnostic classification result.")
    left, right = st.columns(2)
    with left:
        confusion_model = st.selectbox("Model for range agreement", MODELS, format_func=LABELS.get, key="confusion_model")
    with right:
        confusion_horizon = st.selectbox("Minutes ahead", HORIZONS, index=2, key="confusion_horizon")
    cells = zone_confusion.loc[
        (zone_confusion["model"] == confusion_model) & (zone_confusion["horizon_minutes"] == confusion_horizon)
    ]
    order = ["Below 70", "70 to 180", "Above 180"]
    matrix = cells.pivot(index="actual_band", columns="predicted_band", values="count").reindex(index=order, columns=order, fill_value=0)
    matrix.index.name = "Observed band"
    matrix.columns.name = "Predicted band"
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, ax=axis)
    axis.set_title(f"{LABELS[confusion_model]} at {confusion_horizon} minutes")
    axis.set_xlabel("Forecast range")
    axis.set_ylabel("Measured later range")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption("Ranges: below 70 mg/dL, 70–180 mg/dL, and above 180 mg/dL. Values are counts, not percentages.")
