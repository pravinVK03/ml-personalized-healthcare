from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from shanghai_t2d.basic import glucose_band_confusion
from shanghai_t2d.config import StudyConfig
from shanghai_t2d.data import CGM_COLUMN, DATE_COLUMN, parse_episode_id
from shanghai_t2d.evaluation import assert_no_patient_leakage, make_subject_folds
from shanghai_t2d.features import feature_sets, make_supervised
from shanghai_t2d.metrics import regression_metrics


class DataTests(unittest.TestCase):
    def test_episode_identifier(self) -> None:
        subject, period, date = parse_episode_id("2001_1_20201117")
        self.assertEqual(subject, "2001")
        self.assertEqual(period, 1)
        self.assertEqual(date, pd.Timestamp("2020-11-17"))

    def test_bad_episode_identifier_fails(self) -> None:
        with self.assertRaises(ValueError):
            parse_episode_id("patient-1")


class FeatureTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        size = 60
        return pd.DataFrame(
            {
                "subject_id": "1",
                "episode_id": "1_0_20200101",
                DATE_COLUMN: pd.date_range("2020-01-01", periods=size, freq="15min"),
                CGM_COLUMN: np.linspace(90.0, 160.0, size),
            }
        )

    def config(self) -> StudyConfig:
        return StudyConfig(
            history_steps=8,
            forecast_horizons_steps=(1,),
            minimum_episode_rows=12,
            outer_folds=2,
            bootstrap_iterations=20,
        )

    def test_future_values_do_not_change_features(self) -> None:
        original = self.frame()
        changed = original.copy()
        cutoff = original.loc[25, DATE_COLUMN]
        changed.loc[changed[DATE_COLUMN] > cutoff, CGM_COLUMN] += 100.0
        first = make_supervised(original, self.config())
        second = make_supervised(changed, self.config())
        first_row = first.loc[first["timestamp"] == cutoff].iloc[0]
        second_row = second.loc[second["timestamp"] == cutoff].iloc[0]
        columns = sorted(
            set(sum(feature_sets(first.columns).values(), []))
        )
        np.testing.assert_allclose(
            first_row[columns].to_numpy(dtype=float),
            second_row[columns].to_numpy(dtype=float),
        )
        self.assertNotEqual(first_row["target_cgm"], second_row["target_cgm"])

    def test_gap_is_not_bridged(self) -> None:
        frame = self.frame().drop(index=20).reset_index(drop=True)
        output = make_supervised(frame, self.config())
        elapsed = output["target_timestamp"] - output["timestamp"]
        self.assertTrue((elapsed == pd.Timedelta(minutes=15)).all())
        forbidden_start = pd.Timestamp("2020-01-01 05:00")
        first_after_gap = output.loc[output["timestamp"] > forbidden_start, "timestamp"].min()
        self.assertGreaterEqual(first_after_gap, forbidden_start + pd.Timedelta(minutes=8 * 15))


class EvaluationTests(unittest.TestCase):
    def test_subject_folds_and_leakage_assertion(self) -> None:
        mapping = make_subject_folds(["a", "b", "c", "d"], 2, 7)
        train_ids = [subject for subject, fold in mapping.items() if fold == 0]
        test_ids = [subject for subject, fold in mapping.items() if fold == 1]
        train = pd.DataFrame({"subject_id": train_ids})
        test = pd.DataFrame({"subject_id": test_ids})
        assert_no_patient_leakage(train, test)
        with self.assertRaises(AssertionError):
            assert_no_patient_leakage(train, pd.DataFrame({"subject_id": train_ids[:1]}))

    def test_regression_metrics(self) -> None:
        values = np.array([80.0, 100.0, 120.0])
        metrics = regression_metrics(values, values)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["within_15_percent"], 100.0)

    def test_glucose_band_confusion_has_all_bands(self) -> None:
        predictions = pd.DataFrame(
            {
                "model": ["ridge", "ridge", "ridge"],
                "horizon_minutes": [15, 15, 15],
                "target_cgm": [65.0, 100.0, 200.0],
                "prediction": [60.0, 190.0, 170.0],
            }
        )
        confusion = glucose_band_confusion(predictions)
        self.assertEqual(len(confusion), 9)
        self.assertEqual(int(confusion["count"].sum()), 3)
        self.assertEqual(
            int(
                confusion.loc[
                    (confusion["actual_band"] == "Below 70")
                    & (confusion["predicted_band"] == "Below 70"),
                    "count",
                ].iloc[0]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
