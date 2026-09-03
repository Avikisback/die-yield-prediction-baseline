import unittest

import numpy as np
import pandas as pd

from modeling.run_experiments import Experiment, run_experiment
from modeling.validation import repeated_stratified_group_folds


class ExperimentSmokeTest(unittest.TestCase):
    def test_linear_oof_experiment(self):
        rng = np.random.default_rng(11)
        rows = []
        for wafer_number in range(10):
            for die_number in range(20):
                label = int(die_number < 2)
                rows.append(
                    {
                        "wafer_id": f"W{wafer_number}",
                        "die_row": die_number // 5,
                        "die_col": die_number % 5,
                        "old_label": 0,
                        "label": label,
                        "feature_1": rng.normal(label * 1.5, 1.0),
                        "spatial_radius": die_number / 20,
                    }
                )
        frame = pd.DataFrame(rows)
        folds = repeated_stratified_group_folds(
            frame["label"].to_numpy(),
            frame["wafer_id"].to_numpy(),
            n_splits=5,
            repeats=1,
        )
        summary, predictions, fold_rows = run_experiment(
            frame,
            Experiment(
                "test_linear", "linear", ("feature_1", "spatial_radius")
            ),
            folds,
            weight_mode="sqrt",
            random_state=42,
        )
        self.assertEqual(len(predictions), len(frame))
        self.assertEqual(len(fold_rows), 5)
        self.assertGreater(summary["average_precision"], 0.1)
        self.assertTrue(predictions["oof_repeats"].eq(1).all())


if __name__ == "__main__":
    unittest.main()

