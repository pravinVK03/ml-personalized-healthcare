from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import StudyConfig


DATE_COLUMN = "Date"
CGM_COLUMN = "CGM (mg / dl)"
EPISODE_RE = re.compile(r"^(?P<subject>\d+)_(?P<period>\d+)_(?P<date>\d{8})$")


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    t2dm_dir: Path
    summary_path: Path


def discover_dataset(root: str | Path) -> DatasetPaths:
    root = Path(root).expanduser().resolve()
    summary_files = list(root.rglob("Shanghai_T2DM_Summary.xlsx"))
    t2dm_dirs = [path for path in root.rglob("Shanghai_T2DM") if path.is_dir()]
    if len(summary_files) != 1 or len(t2dm_dirs) != 1:
        raise FileNotFoundError(
            "Expected exactly one Shanghai_T2DM folder and one "
            "Shanghai_T2DM_Summary.xlsx below the supplied data root."
        )
    return DatasetPaths(root=root, t2dm_dir=t2dm_dirs[0], summary_path=summary_files[0])


def parse_episode_id(value: str) -> tuple[str, int, pd.Timestamp]:
    match = EPISODE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Unexpected Shanghai episode identifier: {value!r}")
    return (
        match.group("subject"),
        int(match.group("period")),
        pd.to_datetime(match.group("date"), format="%Y%m%d"),
    )


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def list_episode_files(paths: DatasetPaths) -> list[Path]:
    files = [
        path
        for path in paths.t2dm_dir.iterdir()
        if path.suffix.lower() in {".xls", ".xlsx"} and not path.name.startswith("~$")
    ]
    return sorted(files, key=lambda path: path.stem)


def load_summary(paths: DatasetPaths) -> pd.DataFrame:
    summary = pd.read_excel(paths.summary_path, engine="openpyxl")
    summary = summary.rename(columns={"Patient Number": "episode_id"})
    summary["episode_id"] = summary["episode_id"].astype(str).str.strip()
    parsed = summary["episode_id"].map(parse_episode_id)
    summary["subject_id"] = parsed.map(lambda value: value[0])
    summary["period_number"] = parsed.map(lambda value: value[1])
    summary["nominal_start_date"] = parsed.map(lambda value: value[2])
    return summary


def _read_episode(path: Path) -> pd.DataFrame:
    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
    frame = pd.read_excel(path, engine=engine)
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    date_source = next(
        (original for name, original in normalized.items() if name == "date"), None
    )
    cgm_source = next(
        (original for name, original in normalized.items() if name.startswith("cgm")), None
    )
    if date_source is None or cgm_source is None:
        raise ValueError(f"Required Date/CGM columns are missing from {path.name}")
    frame = frame.rename(columns={date_source: DATE_COLUMN, cgm_source: CGM_COLUMN})
    episode_id = path.stem
    subject_id, period_number, nominal_date = parse_episode_id(episode_id)
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    result[CGM_COLUMN] = pd.to_numeric(result[CGM_COLUMN], errors="coerce")
    result.insert(0, "episode_id", episode_id)
    result.insert(1, "subject_id", subject_id)
    result.insert(2, "period_number", period_number)
    result.insert(3, "nominal_start_date", nominal_date)
    result.insert(4, "source_file", path.name)
    return result


def load_episodes(
    paths: DatasetPaths,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    frames = [_read_episode(path) for path in list_episode_files(paths)]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if columns is None:
        return combined
    identifiers = [
        "episode_id",
        "subject_id",
        "period_number",
        "nominal_start_date",
        "source_file",
    ]
    requested = list(dict.fromkeys(identifiers + list(columns)))
    return combined.loc[:, requested]


def build_audit(
    paths: DatasetPaths,
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary = load_summary(paths)
    rows: list[dict[str, object]] = []
    for path in list_episode_files(paths):
        frame = _read_episode(path)
        timestamps = frame[DATE_COLUMN]
        cgm = frame[CGM_COLUMN]
        valid = timestamps.notna() & cgm.notna()
        valid_frame = frame.loc[valid, [DATE_COLUMN, CGM_COLUMN]].sort_values(DATE_COLUMN)
        deltas = valid_frame[DATE_COLUMN].diff().dt.total_seconds().div(60)
        duplicate_count = int(valid_frame[DATE_COLUMN].duplicated().sum())
        irregular_count = int((deltas.dropna() != config.sampling_minutes).sum())
        out_of_range = cgm.notna() & (
            (cgm < config.physiologic_min_mg_dl)
            | (cgm > config.physiologic_max_mg_dl)
        )
        event_columns = [
            column
            for column in frame.columns
            if column
            not in {
                DATE_COLUMN,
                CGM_COLUMN,
                "episode_id",
                "subject_id",
                "period_number",
                "nominal_start_date",
                "source_file",
            }
        ]
        event_non_missing = int(frame[event_columns].notna().any(axis=1).sum())
        rows.append(
            {
                "episode_id": path.stem,
                "subject_id": frame["subject_id"].iloc[0],
                "file_extension": path.suffix.lower(),
                "rows": int(len(frame)),
                "valid_cgm_rows": int(valid.sum()),
                "missing_cgm_rows": int(cgm.isna().sum()),
                "missing_timestamp_rows": int(timestamps.isna().sum()),
                "duplicate_timestamp_rows": duplicate_count,
                "irregular_valid_intervals": irregular_count,
                "out_of_range_cgm_rows": int(out_of_range.sum()),
                "event_rows": event_non_missing,
                "start": valid_frame[DATE_COLUMN].min(),
                "end": valid_frame[DATE_COLUMN].max(),
                "duration_days": (
                    valid_frame[DATE_COLUMN].max() - valid_frame[DATE_COLUMN].min()
                ).total_seconds()
                / 86400
                if not valid_frame.empty
                else np.nan,
                "cgm_mean": float(valid_frame[CGM_COLUMN].mean()),
                "cgm_std": float(valid_frame[CGM_COLUMN].std()),
            }
        )
    episodes = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    summary_ids = set(summary["episode_id"])
    episode_ids = set(episodes["episode_id"])
    per_subject = episodes.groupby("subject_id").size()
    audit = {
        "dataset_root": str(paths.root),
        "summary_file": str(paths.summary_path),
        "summary_sha256": sha256_file(paths.summary_path),
        "n_summary_rows": int(len(summary)),
        "n_episode_files": int(len(episodes)),
        "n_unique_subjects": int(episodes["subject_id"].nunique()),
        "subjects_with_multiple_episodes": int((per_subject > 1).sum()),
        "max_episodes_per_subject": int(per_subject.max()),
        "summary_ids_without_episode": sorted(summary_ids - episode_ids),
        "episode_ids_without_summary": sorted(episode_ids - summary_ids),
        "total_rows": int(episodes["rows"].sum()),
        "total_valid_cgm_rows": int(episodes["valid_cgm_rows"].sum()),
        "total_missing_cgm_rows": int(episodes["missing_cgm_rows"].sum()),
        "total_duplicate_timestamp_rows": int(episodes["duplicate_timestamp_rows"].sum()),
        "total_irregular_valid_intervals": int(episodes["irregular_valid_intervals"].sum()),
        "total_out_of_range_cgm_rows": int(episodes["out_of_range_cgm_rows"].sum()),
        "median_episode_days": float(episodes["duration_days"].median()),
        "min_episode_days": float(episodes["duration_days"].min()),
        "max_episode_days": float(episodes["duration_days"].max()),
        "hba1c_longitudinal_feasibility": "not_supported",
        "hba1c_reason": (
            "The summary is episode-level and only a small minority of subjects have repeat "
            "episodes; this is insufficient for a robust next-HbA1c study."
        ),
    }
    return summary, episodes, audit


def write_audit(
    paths: DatasetPaths,
    config: StudyConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, episodes, audit = build_audit(paths, config)
    episodes.to_csv(output_dir / "episode_audit.csv", index=False)
    summary.to_csv(output_dir / "summary_normalized.csv", index=False)
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    return audit
