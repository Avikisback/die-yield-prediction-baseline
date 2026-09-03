import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.cache_features import build_cache
from modeling.run_experiments import load_cache


class CacheSmokeTest(unittest.TestCase):
    def test_chunk_boundaries_preserve_complete_wafers(self):
        rng = np.random.default_rng(5)
        rows = []
        for wafer_number in range(3):
            for die_number in range(12):
                signal = rng.normal(100, 10, 64)
                if die_number == 1:
                    signal[20:30] += 5
                rows.append(
                    {
                        "wafer_id": f"W{wafer_number}",
                        "die_row": die_number // 4,
                        "die_col": die_number % 4,
                        "feature_1": rng.normal(),
                        "block_readings": " ".join(f"{value:.2f}" for value in signal),
                        "old_label": int(die_number == 0),
                        "label": int(die_number in (0, 1)),
                    }
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "train.csv"
            cache = root / "cache"
            pd.DataFrame(rows).to_csv(source, index=False)
            build_cache(source, cache, chunksize=13, include_block=True)
            loaded = load_cache(cache)

            self.assertEqual(len(loaded), len(rows))
            self.assertEqual(loaded["wafer_id"].nunique(), 3)
            self.assertIn("spatial_old_fail_density_w5", loaded)
            self.assertIn("block_scan_max_w256", loaded)
            self.assertFalse(loaded.duplicated(["wafer_id", "die_row", "die_col"]).any())


if __name__ == "__main__":
    unittest.main()
