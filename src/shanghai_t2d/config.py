from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    random_seed: int = 20261025
    sampling_minutes: int = 15
    history_steps: int = 24
    forecast_horizons_steps: tuple[int, ...] = (1, 2, 4)
    outer_folds: int = 5
    bootstrap_iterations: int = 2000
    physiologic_min_mg_dl: float = 20.0
    physiologic_max_mg_dl: float = 600.0
    minimum_episode_rows: int = 32

    @classmethod
    def from_json(cls, path: str | Path) -> "StudyConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["forecast_horizons_steps"] = tuple(values["forecast_horizons_steps"])
        return cls(**values)


