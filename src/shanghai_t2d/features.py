from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig
from .data import CGM_COLUMN, DATE_COLUMN, DatasetPaths, load_episodes


IDENTIFIER_COLUMNS = [
    "subject_id",
    "episode_id",
    "timestamp",
    "target_timestamp",
    "horizon_steps",
    "horizon_minutes",
]


def _continuous_blocks(frame: pd.DataFrame, sampling_minutes: int) -> list[pd.DataFrame]:
    frame = frame[[DATE_COLUMN, CGM_COLUMN]].copy()
    frame = frame.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    frame = (
        frame.groupby(DATE_COLUMN, as_index=False, sort=True)[CGM_COLUMN]
        .median()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )
    expected = pd.Timedelta(minutes=sampling_minutes)
    breaks = frame[CGM_COLUMN].isna() | frame[CGM_COLUMN].shift().isna()
    breaks |= frame[DATE_COLUMN].diff().ne(expected)
    block_ids = breaks.cumsum()
    return [
        block.reset_index(drop=True)
        for _, block in frame.loc[frame[CGM_COLUMN].notna()].groupby(block_ids)
    ]


def _block_features(
    block: pd.DataFrame,
    *,
    subject_id: str,
    episode_id: str,
    config: StudyConfig,
    horizon_steps: int,
) -> pd.DataFrame:
    values = block[CGM_COLUMN].astype(float)
    output = pd.DataFrame(index=block.index)
    output["subject_id"] = subject_id
    output["episode_id"] = episode_id
    output["timestamp"] = block[DATE_COLUMN]
    output["target_timestamp"] = block[DATE_COLUMN].shift(-horizon_steps)
    output["horizon_steps"] = horizon_steps
    output["horizon_minutes"] = horizon_steps * config.sampling_minutes
    output["target_cgm"] = values.shift(-horizon_steps)
    output["current_cgm"] = values

    hour = block[DATE_COLUMN].dt.hour + block[DATE_COLUMN].dt.minute / 60.0
    output["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    output["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)

    for lag in range(config.history_steps):
        output[f"lag_{lag:02d}"] = values.shift(lag)

    difference = values.diff()
    output["delta_1"] = difference
    output["delta_2"] = values - values.shift(2)
    output["delta_4"] = values - values.shift(4)
    for window in (4, 8, 24):
        rolling = values.rolling(window, min_periods=window)
        output[f"roll_mean_{window}"] = rolling.mean()
        output[f"roll_std_{window}"] = rolling.std(ddof=0)
        output[f"roll_min_{window}"] = rolling.min()
        output[f"roll_max_{window}"] = rolling.max()

    first_eligible = config.history_steps - 1
    last_eligible = len(output) - horizon_steps
    if last_eligible <= first_eligible:
        return output.iloc[0:0]
    return output.iloc[first_eligible:last_eligible].reset_index(drop=True)


def make_supervised(
    raw: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    required = {"subject_id", "episode_id", DATE_COLUMN, CGM_COLUMN}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")
    frames: list[pd.DataFrame] = []
    grouped = raw.groupby(["subject_id", "episode_id"], sort=True)
    for (subject_id, episode_id), episode in grouped:
        blocks = _continuous_blocks(episode, config.sampling_minutes)
        for block in blocks:
            if len(block) < config.minimum_episode_rows:
                continue
            for horizon_steps in config.forecast_horizons_steps:
                features = _block_features(
                    block,
                    subject_id=str(subject_id),
                    episode_id=str(episode_id),
                    config=config,
                    horizon_steps=horizon_steps,
                )
                if not features.empty:
                    frames.append(features)
    if not frames:
        raise ValueError("No eligible continuous CGM windows were constructed.")
    result = pd.concat(frames, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.sort_values(
        ["horizon_steps", "subject_id", "episode_id", "timestamp"]
    ).reset_index(drop=True)


def build_supervised_dataset(
    paths: DatasetPaths,
    config: StudyConfig,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    raw = load_episodes(paths, columns=[DATE_COLUMN, CGM_COLUMN])
    valid_range = raw[CGM_COLUMN].between(
        config.physiologic_min_mg_dl,
        config.physiologic_max_mg_dl,
        inclusive="both",
    )
    raw.loc[raw[CGM_COLUMN].notna() & ~valid_range, CGM_COLUMN] = np.nan
    supervised = make_supervised(raw, config)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        supervised.to_csv(output_path, index=False)
    return supervised


def feature_sets(columns: list[str] | pd.Index) -> dict[str, list[str]]:
    available = set(columns)
    history = sorted(column for column in available if column.startswith("lag_"))
    history += [
        "tod_sin",
        "tod_cos",
        "delta_1",
        "delta_2",
        "delta_4",
        "roll_mean_4",
        "roll_std_4",
        "roll_mean_8",
        "roll_std_8",
        "roll_mean_24",
        "roll_std_24",
        "roll_min_24",
        "roll_max_24",
    ]
    result = {"history": history}
    for name, names in result.items():
        absent = set(names) - available
        if absent:
            raise ValueError(f"Feature set {name} is missing columns: {sorted(absent)}")
    return result
