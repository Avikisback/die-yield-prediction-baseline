import unittest

import numpy as np
import pandas as pd

from modeling.features import (
    engineer_one_block_signal,
    engineer_spatial_features,
)
from modeling.validation import eligible_rows, repeated_stratified_group_folds


class SpatialFeatureTests(unittest.TestCase):
    def test_old_failure_density_uses_pretest_map(self):
        rows = []
        for row in range(5):
            for col in range(5):
                rows.append(
                    {
                        "wafer_id": "W0",
                        "die_row": row,
                        "die_col": col,
                        "old_label": int((row, col) == (2, 2)),
                        "label": int((row, col) == (0, 0)),
                    }
                )
        frame = pd.DataFrame(rows)
        output = engineer_spatial_features(frame, windows=(3, 5))
        center = output.loc[(output.die_row == 2) & (output.die_col == 2)].iloc[0]
        corner = output.loc[(output.die_row == 0) & (output.die_col == 0)].iloc[0]
        self.assertAlmostEqual(center["spatial_old_fail_density_w3"], 1 / 9)
        self.assertEqual(corner["spatial_old_fail_density_w3"], 0.0)

    def test_posttest_label_does_not_change_spatial_features(self):
        frame = pd.DataFrame(
            {
                "wafer_id": ["W0"] * 4,
                "die_row": [0, 0, 1, 1],
                "die_col": [0, 1, 0, 1],
                "old_label": [0, 1, 0, 0],
                "label": [0, 1, 1, 0],
            }
        )
        changed = frame.copy()
        changed["label"] = 1 - changed["label"]
        left = engineer_spatial_features(frame)
        right = engineer_spatial_features(changed)
        columns = [column for column in left if column.startswith("spatial_")]
        pd.testing.assert_frame_equal(left[columns], right[columns])


class BlockFeatureTests(unittest.TestCase):
    def test_scan_detects_clustered_anomaly(self):
        rng = np.random.default_rng(7)
        normal = rng.normal(0, 1, 2_000)
        anomalous = normal.copy()
        anomalous[850:1100] += 0.7
        normal_features = engineer_one_block_signal(normal, windows=(64, 256))
        anomaly_features = engineer_one_block_signal(anomalous, windows=(64, 256))
        self.assertGreater(
            anomaly_features["block_scan_max_w256"],
            normal_features["block_scan_max_w256"],
        )


class ValidationTests(unittest.TestCase):
    def test_eligibility_and_group_isolation(self):
        frame = pd.DataFrame(
            {
                "wafer_id": np.repeat([f"W{i}" for i in range(10)], 4),
                "old_label": [0, 0, 0, 1] * 10,
                "label": [0, 0, 1, 1] * 10,
            }
        )
        eligible = eligible_rows(frame)
        self.assertTrue(eligible["old_label"].eq(0).all())
        folds = repeated_stratified_group_folds(
            eligible["label"].to_numpy(),
            eligible["wafer_id"].to_numpy(),
            n_splits=5,
            repeats=2,
        )
        self.assertEqual(len(folds), 10)
        for fold in folds:
            train_groups = set(eligible.iloc[fold.train_index]["wafer_id"])
            validation_groups = set(eligible.iloc[fold.validation_index]["wafer_id"])
            self.assertFalse(train_groups & validation_groups)


if __name__ == "__main__":
    unittest.main()

