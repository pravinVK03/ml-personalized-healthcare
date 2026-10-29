from __future__ import annotations

import unittest

from api.catalog import DATASETS, all_datasets


class DatasetCatalogTests(unittest.TestCase):
    def test_three_dataset_adapters_are_explicit(self) -> None:
        self.assertEqual(
            set(DATASETS), {"shanghai_t2d", "us_diabetes_130", "mimic_iv_demo"}
        )
        self.assertEqual(len(all_datasets()), 3)

    def test_datasets_are_not_pooled(self) -> None:
        targets = {item["target"] for item in DATASETS.values()}
        self.assertEqual(len(targets), 3)
        self.assertIn("CGM value 15, 30, or 60 minutes ahead", targets)


if __name__ == "__main__":
    unittest.main()
