"""Checks for the multi-resolution pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from mrf import block as blockmod
from mrf import interpret, models, parametric, spatial
from mrf.calibrate import MixtureReference, prior_shift


def make_wafer(side: int = 24, seed: int = 0, features: int = 12):
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:side, 0:side]
    keep = ((rows - side / 2) ** 2 + (cols - side / 2) ** 2) < (side / 2) ** 2
    frame = pd.DataFrame(
        {
            "wafer_id": "W0",
            "die_row": rows[keep],
            "die_col": cols[keep],
            "old_label": (rng.random(int(keep.sum())) < 0.1).astype(int),
        }
    )
    for i in range(features):
        frame[f"feature_{i + 1}"] = rng.normal(100.0, 5.0, len(frame))
    return frame


class SpatialFeatures(unittest.TestCase):
    def test_only_pre_test_information_is_used(self):
        wafer = make_wafer()
        wafer["label"] = 1  # every die fails after the test
        without = spatial.engineer(wafer.drop(columns=["label"]))
        with_label = spatial.engineer(wafer)
        columns = [c for c in without.columns if c.startswith("spatial_")]
        pd.testing.assert_frame_equal(without[columns], with_label[columns])

    def test_density_matches_a_hand_computed_window(self):
        wafer = make_wafer(side=9, seed=3)
        wafer["old_label"] = 0
        wafer.loc[wafer.index[0], "old_label"] = 1
        out = spatial.engineer(wafer)
        self.assertGreater(out["spatial_old_fail_density_w3"].max(), 0.0)
        self.assertLessEqual(out["spatial_old_fail_density_w3"].max(), 1.0)

    def test_hazard_shape_is_finite_and_increasing_in_density(self):
        out = spatial.engineer(make_wafer(seed=5))
        self.assertTrue(np.isfinite(out["spatial_log_hazard_shape"]).all())
        correlation = np.corrcoef(
            out["spatial_log_hazard_shape"], out["spatial_old_fail_density_w5"]
        )[0, 1]
        self.assertGreater(correlation, 0.0)


class Detrending(unittest.TestCase):
    def test_a_planted_gradient_is_removed(self):
        wafer = make_wafer(seed=7, features=6)
        columns = [c for c in wafer.columns if c.startswith("feature_")]
        basis = spatial.coordinate_basis(wafer)
        radius, xn = basis[:, 3], basis[:, 1]
        for column in columns:
            wafer[column] = wafer[column] + 4.0 * radius - 2.5 * xn

        standardised, _ = parametric.detrend_wafer(wafer, columns)
        for position in range(len(columns)):
            self.assertLess(abs(np.corrcoef(standardised[:, position], radius)[0, 1]), 0.2)

    def test_output_is_standardised(self):
        wafer = make_wafer(seed=11)
        columns = [c for c in wafer.columns if c.startswith("feature_")]
        standardised, _ = parametric.detrend_wafer(wafer, columns)
        self.assertLess(abs(float(np.median(standardised))), 0.15)
        self.assertLess(abs(float(standardised.std()) - 1.0), 0.25)


class BlockScans(unittest.TestCase):
    @staticmethod
    def signal(rng, fail: bool, k: int = blockmod.READINGS):
        readings = rng.normal(100.0, 15.0, k)
        readings = 0.6 * readings + 0.4 * uniform_filter1d(readings, 5, mode="nearest")
        if fail:
            seed = int(rng.integers(0, k))
            positions = {seed}
            while len(positions) < 100:
                positions.add(int((seed + int(rng.normal(0, k * 0.05))) % k))
            index = np.fromiter(positions, int)
            readings[index] += rng.normal(4.5, 1.35, len(index))
        return readings

    def test_gain_normalisation_gives_a_comparable_scale(self):
        rng = np.random.default_rng(0)
        rows = [self.signal(rng, False) for _ in range(64)]
        values = pd.Series([" ".join(f"{v:.2f}" for v in row) for row in rows])
        out = blockmod.engineer(values)
        for width in blockmod.GAUSSIAN_WIDTHS:
            peak = out[f"block_scan_gauss{width}"].mean()
            # A scan maximum over pure noise sits a few sigma up, never at 20.
            self.assertGreater(peak, 0.2)
            self.assertLess(peak, 6.0)

    def test_a_planted_cluster_separates_from_noise(self):
        rng = np.random.default_rng(1)
        rows = [self.signal(rng, i % 2 == 0) for i in range(120)]
        values = pd.Series([" ".join(f"{v:.2f}" for v in row) for row in rows])
        out = blockmod.engineer(values)
        failing = np.arange(120) % 2 == 0
        column = out["block_scan_gauss150"].to_numpy()
        self.assertGreater(column[failing].mean(), column[~failing].mean())

    def test_short_and_ragged_rows_do_not_crash(self):
        values = pd.Series(["1.0 2.0 3.0", " ".join(["1.5"] * 4000)])
        out = blockmod.engineer(values)
        self.assertEqual(len(out), 2)
        self.assertTrue(np.isfinite(out.to_numpy()).any())


class WaferRateRecovery(unittest.TestCase):
    def test_a_known_mixing_weight_is_recovered(self):
        rng = np.random.default_rng(4)
        wafers, scores, labels = [], [], []
        rates = {}
        for index in range(30):
            rate = float(rng.uniform(0.01, 0.20))
            n = 900
            fail = rng.random(n) < rate
            score = np.where(fail, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))
            wafers.append(np.full(n, f"W{index}"))
            scores.append(score)
            labels.append(fail)
            rates[f"W{index}"] = fail.mean()
        wafer = np.concatenate(wafers)
        score = np.concatenate(scores)
        label = np.concatenate(labels)

        reference = MixtureReference.fit(score, label, wafer)
        _, recovered = prior_shift(score, np.full(len(score), 0.05), wafer,
                                   reference, leave_out=True, alpha=0.5)
        actual = np.array([rates[k] for k in recovered])
        estimated = np.array(list(recovered.values()))
        self.assertGreater(float(np.corrcoef(actual, estimated)[0, 1]), 0.85)

    def test_shrinkage_scales_the_offset(self):
        rng = np.random.default_rng(6)
        wafer = np.repeat([f"W{i}" for i in range(6)], 300)
        score = rng.normal(0, 1, len(wafer))
        label = rng.random(len(wafer)) < 0.08
        reference = MixtureReference.fit(score, label, wafer)
        base = np.full(len(wafer), 0.05)
        half, rates = prior_shift(score, base, wafer, reference, False, alpha=0.5)
        full, _ = prior_shift(score, base, wafer, reference, False, alpha=1.0,
                              rates=rates)
        none, _ = prior_shift(score, base, wafer, reference, False, alpha=0.0,
                              rates=rates)
        def logit(p):
            return np.log(p / (1 - p))
        np.testing.assert_allclose(logit(none), logit(base), atol=1e-9)
        np.testing.assert_allclose(
            logit(half) - logit(base), 0.5 * (logit(full) - logit(base)), atol=1e-9
        )


class Attribution(unittest.TestCase):
    """The report claims contributions are exact, so they are checked as such."""

    @staticmethod
    def dataset(seed=2, n=800):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {f"feature_{i + 1}": rng.normal(0, 1, n) for i in range(20)}
        )
        for i in range(3):
            frame[f"spatial_s{i}"] = rng.normal(0, 1, n)
        for i in range(3):
            frame[f"block_b{i}"] = rng.normal(0, 1, n)
        y = (frame["feature_1"] + frame["block_b0"] + rng.normal(0, 1, n) > 1.2).astype(int)
        return frame, y.to_numpy()

    def _check_sums_to_logit(self, backend):
        x, y = self.dataset()
        model = models.make_model(backend, C=1.0)
        models.fit(model, x, y, models.positive_weight(y, "sqrt"))
        values, intercept = interpret.contributions(model, x)
        probability = model.predict_proba(x)[:, 1]
        expected = np.log(probability / (1 - probability))
        np.testing.assert_allclose(values.sum(axis=1) + intercept, expected, atol=1e-6)

    def test_flat_model_contributions_are_exact(self):
        self._check_sums_to_logit("linear")

    def test_stacked_model_contributions_are_exact(self):
        self._check_sums_to_logit("stacked")

    def test_group_contributions_cover_every_column(self):
        x, y = self.dataset()
        model = models.make_model("stacked", C=1.0)
        models.fit(model, x, y, models.positive_weight(y, "sqrt"))
        parts = interpret.group_contributions(model, x)
        probability = model.predict_proba(x)[:, 1]
        np.testing.assert_allclose(
            parts["logit"], np.log(probability / (1 - probability)), atol=1e-6
        )

    def test_importance_flags_the_planted_driver(self):
        x, y = self.dataset()
        model = models.make_model("stacked", C=1.0)
        models.fit(model, x, y, models.positive_weight(y, "sqrt"))
        table = interpret.global_importance(model, list(x.columns))
        self.assertIn("block_b0", set(table.head(3)["feature"]))


if __name__ == "__main__":
    unittest.main()
