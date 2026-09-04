"""Checks for the generator-matched fusion package."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import generate_data as reference
from tuned import (
    blocks,
    channels,
    final,
    genstream,
    hazard,
    head as head_module,
    waferrate,
)
from tuned.pipeline import Fusion


def toy_wafer_map(rows: int = 21, cols: int = 23, seed: int = 0) -> np.ndarray:
    """A round wafer with a handful of pre-test failures, as WM-811K stores it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols]
    radius = np.sqrt(((y - rows / 2) / (rows / 2)) ** 2
                     + ((x - cols / 2) / (cols / 2)) ** 2)
    wafer = np.where(radius <= 1.0, 1, 0)
    inside = np.argwhere(wafer == 1)
    chosen = rng.choice(len(inside), size=max(1, len(inside) // 20), replace=False)
    for index in chosen:
        wafer[inside[index][0], inside[index][1]] = 2
    return wafer


def frame_from_map(wafer_map: np.ndarray, wafer_id: str = "W_T_0000",
                   seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row, col = np.where(wafer_map > 0)
    return pd.DataFrame({
        "wafer_id": wafer_id,
        "die_row": row,
        "die_col": col,
        "old_label": (wafer_map[row, col] == 2).astype(np.int8),
        "label": np.maximum((wafer_map[row, col] == 2).astype(np.int8),
                            (rng.random(len(row)) < 0.05).astype(np.int8)),
    })


class HazardTests(unittest.TestCase):
    def test_matches_the_generator_radial_map_exactly(self):
        wafer_map = toy_wafer_map()
        frame = frame_from_map(wafer_map)
        engineered = hazard.engineer(frame)
        expected = reference.compute_radial_map(wafer_map)
        row = frame["die_row"].to_numpy()
        col = frame["die_col"].to_numpy()
        np.testing.assert_allclose(
            engineered["haz_radius"].to_numpy(dtype=np.float64),
            expected[row, col], atol=1e-6)

    def test_matches_the_generator_neighbourhood_density_exactly(self):
        wafer_map = toy_wafer_map(seed=3)
        frame = frame_from_map(wafer_map)
        engineered = hazard.engineer(frame)
        expected = reference.compute_neighborhood_fail_density(wafer_map, 5)
        row = frame["die_row"].to_numpy()
        col = frame["die_col"].to_numpy()
        np.testing.assert_allclose(
            engineered["haz_density_w5"].to_numpy(dtype=np.float64),
            expected[row, col], atol=1e-6)

    def test_normalising_by_observed_dies_inflates_the_radius(self):
        """The convention mrf.spatial uses, measured rather than argued about."""
        frame = frame_from_map(toy_wafer_map())
        engineered = hazard.engineer(frame)
        scale = float(engineered["haz_radius_scale"].iloc[0])
        self.assertLess(scale, 0.85)
        self.assertGreater(scale, 0.6)

    def test_reads_no_post_test_information(self):
        frame = frame_from_map(toy_wafer_map(seed=5))
        first = hazard.engineer(frame)
        shuffled = frame.copy()
        shuffled["label"] = shuffled["label"].sample(
            frac=1.0, random_state=7).to_numpy()
        second = hazard.engineer(shuffled)
        pd.testing.assert_frame_equal(first, second)


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        # The full 2,000 readings, because the detector's envelope widths and
        # the cluster's spread are both fractions of that length; a shortened
        # series tests a different problem.
        self.k = blocks.READINGS

    def _readings(self, n, fail, amplitude=4.5):
        """The generator's own recipe: smoothed noise, then a clustered spike."""
        out = np.empty((n, self.k))
        for i in range(n):
            row = self.rng.normal(100.0, 15.0, size=self.k)
            row = 0.6 * row + 0.4 * np.convolve(
                np.r_[row[-2:], row, row[:2]], np.ones(5) / 5, "valid")
            if fail[i]:
                seed = self.rng.integers(0, self.k)
                positions = (seed + self.rng.normal(
                    0, 0.05 * self.k, size=self.k // 20).astype(int)) % self.k
                row[positions] += amplitude
            out[i] = row
        return out

    def test_parse_pads_short_rows_without_introducing_nan(self):
        values = pd.Series(["1.0 2.0 3.0", "4.0 5.0"])
        parsed = blocks.parse(values, k=4)
        self.assertEqual(parsed.shape, (2, 4))
        self.assertTrue(np.isfinite(parsed).all())

    def test_a_planted_cluster_separates_from_noise(self):
        """One scan statistic on its own should rank a planted cluster well.

        The bar is ROC-AUC 0.65 for the best single scan.  The whole channel
        reaches about 0.74 on the real data with a fitted combination of them,
        so this is the right order of magnitude and not a threshold picked to
        pass.
        """
        label = np.r_[np.zeros(200, bool), np.ones(200, bool)]
        readings = self._readings(len(label), label)
        noise = blocks.NoiseModel.fit(readings)
        columns, _ = blocks.engineer(readings, noise)
        best = max(roc_auc_score(label, columns[name])
                   for name in blocks.scan_names())
        self.assertGreater(best, 0.65)

    def test_no_cluster_means_no_separation(self):
        """The same statistics on two halves of pure noise must find nothing."""
        label = np.r_[np.zeros(200, bool), np.ones(200, bool)]
        readings = self._readings(len(label), np.zeros(len(label), bool))
        noise = blocks.NoiseModel.fit(readings)
        columns, _ = blocks.engineer(readings, noise)
        best = max(abs(roc_auc_score(label, columns[name]) - 0.5)
                   for name in blocks.scan_names())
        self.assertLess(best, 0.12)

    def test_scan_normalisation_is_a_constant_not_the_dies_own_spread(self):
        """Doubling one die's noise must not leave its scan statistic unchanged."""
        label = np.zeros(60, bool)
        readings = self._readings(len(label), label)
        noise = blocks.NoiseModel.fit(readings)
        base, reference = blocks.engineer(readings, noise)
        louder = readings.copy()
        louder[0] = 100.0 + 2.0 * (louder[0] - 100.0)
        after, _ = blocks.engineer(louder, noise, reference=reference)
        name = blocks.scan_names()[0]
        self.assertGreater(abs(after[name][0] - base[name][0]), 1e-3)

    def test_reference_constants_are_reused_across_batches(self):
        label = np.zeros(60, bool)
        readings = self._readings(len(label), label)
        noise = blocks.NoiseModel.fit(readings)
        _, reference = blocks.engineer(readings, noise)
        first, _ = blocks.engineer(readings[:20], noise, reference=reference)
        second, _ = blocks.engineer(readings[:20], noise, reference=dict(reference))
        for name in blocks.scan_names():
            np.testing.assert_allclose(first[name], second[name])


class HeadTests(unittest.TestCase):
    def _data(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        score = rng.normal(size=n)
        offset = np.full(n, -3.0)
        # a deliberately kinked truth: flat below zero, steep above
        truth = offset + np.where(score > 0, 2.5 * score, 0.2 * score)
        y = (rng.random(n) < 1 / (1 + np.exp(-truth))).astype(float)
        return pd.DataFrame({"score": score}), y, offset

    def test_partial_contributions_sum_to_the_evidence(self):
        frame, y, offset = self._data()
        head = head_module.AdditiveHead(("score",)).fit(frame, y, offset)
        parts = sum(head.partial(frame).values())
        np.testing.assert_allclose(head.evidence(frame),
                                   head.intercept_ + parts, atol=1e-9)

    def test_the_offset_enters_with_coefficient_one(self):
        """Shifting every offset by a constant must shift the log-odds by it."""
        frame, y, offset = self._data(seed=1)
        head = head_module.AdditiveHead(("score",)).fit(frame, y, offset)
        base = head.decision(frame, offset)
        moved = head.decision(frame, offset + 1.7)
        np.testing.assert_allclose(moved - base, 1.7, atol=1e-9)

    def test_a_spline_beats_a_single_coefficient_on_a_kinked_truth(self):
        frame, y, offset = self._data(n=20000, seed=2)
        smooth = head_module.AdditiveHead(("score",)).fit(frame, y, offset)
        linear = head_module.AdditiveHead((), ("score",)).fit(frame, y, offset)

        def deviance(model):
            eta = model.decision(frame, offset)
            return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

        self.assertLess(deviance(smooth), deviance(linear) - 1e-4)

    def test_values_beyond_the_knots_are_clamped_not_extrapolated(self):
        frame, y, offset = self._data(seed=3)
        head = head_module.AdditiveHead(("score",)).fit(frame, y, offset)
        far = np.array([1e6])
        edge = np.array([frame["score"].max()])
        np.testing.assert_allclose(head.curve("score", far),
                                   head.curve("score", edge), atol=1e-6)


class WaferRateTests(unittest.TestCase):
    def test_a_known_rate_is_recovered_without_labels(self):
        rng = np.random.default_rng(4)
        recovered, actual = [], []
        for rate in (0.005, 0.02, 0.05, 0.12):
            shape = 1.0 + 3.0 * rng.random(900) + 1.5 * rng.random(900)
            probability = np.clip(rate * shape, 0, 0.4)
            y = rng.random(900) < probability
            # evidence with a realistic amount of separation
            evidence = np.where(y, rng.normal(2.2, 1.0, 900),
                                rng.normal(-0.6, 1.0, 900))
            posterior = waferrate.wafer_posterior(evidence, shape, 0.02)
            recovered.append(posterior.mean)
            actual.append(rate)
        self.assertGreater(float(np.corrcoef(recovered, actual)[0, 1]), 0.9)

    def test_with_no_evidence_the_posterior_is_the_prior(self):
        shape = np.ones(500)
        posterior = waferrate.wafer_posterior(np.zeros(500), shape, 0.02)
        self.assertAlmostEqual(posterior.mean, 0.02, delta=0.004)

    def test_more_dies_shrink_the_posterior_less(self):
        rng = np.random.default_rng(9)
        rate = 0.10
        wide, narrow = None, None
        for size in (120, 4000):
            shape = np.ones(size)
            y = rng.random(size) < rate
            evidence = np.where(y, 3.0, -1.0)
            posterior = waferrate.wafer_posterior(evidence, shape, 0.02)
            spread = float(np.sqrt(posterior.weight
                                   @ (posterior.grid - posterior.mean) ** 2))
            if size == 120:
                wide = spread
            else:
                narrow = spread
        self.assertLess(narrow, wide)


class ChannelTests(unittest.TestCase):
    def test_the_diagonal_direction_recovers_a_planted_shift(self):
        rng = np.random.default_rng(6)
        n, p = 6000, 60
        shift = rng.normal(size=p) * 0.4
        y = rng.random(n) < 0.1
        x = rng.normal(size=(n, p)).astype(np.float32)
        x[y] += shift
        score = channels.DiagonalScore.fit(x, y, ~y)
        self.assertGreater(float(np.corrcoef(score.separation, shift)[0, 1]), 0.95)

    def test_pre_test_failures_add_positives_to_the_direction_fit(self):
        rng = np.random.default_rng(8)
        n, p = 4000, 30
        old = rng.random(n) < 0.05
        new = (rng.random(n) < 0.05) & ~old
        label = old | new
        x = rng.normal(size=(n, p)).astype(np.float32)
        x[label] += 0.5
        with_old = channels.fit_channel(x, label, old, np.ones(n, bool), True)
        without = channels.fit_channel(x, label, old, np.ones(n, bool), False)
        self.assertGreater(with_old.n_positive, without.n_positive)


class GeneratorEquivalenceTests(unittest.TestCase):
    def test_the_streaming_generator_takes_the_same_random_draws(self):
        config = {
            "features": [{"name": f"feature_{i + 1}", "base_mean": 10.0 + i,
                          "base_std": 1.0 + 0.1 * i, "fail_shift": 0.3}
                         for i in range(6)],
            "neighborhood_window": 5, "radial_gradient_strength": 0.3,
            "linear_gradient_strength": 0.15, "neighborhood_influence": 0.2,
            "marginal_fail_fraction": 0.65, "new_fail_rate": 0.05,
        }
        wafer_map = toy_wafer_map(seed=2)
        expected = reference.generate_die_features(
            wafer_map, config, np.random.default_rng(123))
        actual = genstream.die_features_with_truth(
            wafer_map, config, np.random.default_rng(123))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


class FusionTests(unittest.TestCase):
    def _frame(self, wafers=6, seed=0):
        rng = np.random.default_rng(seed)
        pieces, matrices = [], []
        for index in range(wafers):
            wafer_map = toy_wafer_map(seed=seed * 10 + index)
            base = frame_from_map(wafer_map, f"W_T_{index:04d}",
                                  seed=seed * 100 + index)
            engineered = hazard.engineer(base)
            block = pd.DataFrame(
                {f"block_lr_x{j}": rng.normal(size=len(base))
                 + 0.8 * base["label"].to_numpy() for j in range(4)},
                index=base.index)
            pieces.append(pd.concat([base, engineered, block], axis=1))
            matrix = rng.normal(size=(len(base), 12)).astype(np.float32)
            matrix[base["label"].to_numpy() == 1] += 0.6
            matrices.append(matrix)
        frame = pd.concat(pieces, ignore_index=True)
        return frame, np.vstack(matrices)

    def test_fit_and_predict_run_and_split_the_log_odds_exactly(self):
        frame, x = self._frame()
        rows = np.ones(len(frame), dtype=bool)
        model = Fusion().fit(frame, x, rows)
        eligible = frame["old_label"].to_numpy() == 0
        probability = model.predict_proba(frame, x, eligible)
        self.assertEqual(len(probability), int(eligible.sum()))
        self.assertTrue(np.all((probability > 0) & (probability < 1)))

        design = model._design(frame, x, eligible)
        correction, evidence = model._split(design)
        np.testing.assert_allclose(correction + evidence,
                                   model.head_.evidence(design), atol=1e-9)

    def test_scoring_wafers_the_fit_never_saw_still_produces_rates(self):
        frame, x = self._frame(wafers=8, seed=1)
        held_out = frame["wafer_id"].isin(["W_T_0006", "W_T_0007"]).to_numpy()
        model = Fusion().fit(frame, x, ~held_out)
        eligible = held_out & (frame["old_label"].to_numpy() == 0)
        model.predict_proba(frame, x, eligible)
        self.assertEqual(set(model.last_rates_), {"W_T_0006", "W_T_0007"})

    def test_a_saved_model_scores_a_frame_the_same_way(self):
        """joblib round-trip, because tuned.final ships the fitted object."""
        frame, x = self._frame(seed=3)
        rows = np.ones(len(frame), dtype=bool)
        model = Fusion().fit(frame, x, rows)
        eligible = frame["old_label"].to_numpy() == 0
        before = model.predict_proba(frame, x, eligible)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.joblib"
            joblib.dump(model, path)
            after = joblib.load(path).predict_proba(frame, x, eligible)
        np.testing.assert_allclose(before, after)

    def test_without_the_rate_step_the_posterior_is_prior_times_evidence(self):
        frame, x = self._frame(seed=4)
        rows = np.ones(len(frame), dtype=bool)
        model = Fusion(use_rate=False).fit(frame, x, rows)
        eligible = frame["old_label"].to_numpy() == 0
        scored = model.score_frame(frame, x, eligible)
        prior = np.clip(model.overall_rate_ * scored["hazard"].to_numpy(),
                        1e-9, 0.4)
        expected = 1.0 / (1.0 + np.exp(
            -(np.log(prior / (1 - prior)) + scored["evidence"].to_numpy())))
        np.testing.assert_allclose(
            model.predict_proba(frame, x, eligible), expected, atol=1e-12)

    def test_turning_off_the_rate_step_leaves_a_single_population_prior(self):
        frame, x = self._frame(seed=2)
        model = Fusion(use_rate=False).fit(frame, x, np.ones(len(frame), bool))
        eligible = frame["old_label"].to_numpy() == 0
        model.predict_proba(frame, x, eligible)
        self.assertEqual(len(set(model.last_rates_.values())), 1)


class SubmissionTests(unittest.TestCase):
    def _submission(self, labels, eligible):
        return pd.DataFrame({
            "wafer_id": ["W_T_0000"] * len(labels),
            "die_row": np.arange(len(labels)),
            "die_col": np.zeros(len(labels), dtype=int),
            "predicted_label": np.asarray(labels, dtype=np.int8),
        }), np.asarray(eligible, dtype=bool)

    def test_a_well_formed_submission_passes(self):
        frame, eligible = self._submission([0, 1, 1, 0], [True, True, False, True])
        final.check_submission(frame, eligible)

    def test_a_pre_test_failure_predicted_as_pass_is_rejected(self):
        frame, eligible = self._submission([0, 1, 0, 0], [True, True, False, True])
        with self.assertRaises(AssertionError):
            final.check_submission(frame, eligible)

    def test_duplicate_die_keys_are_rejected(self):
        frame, eligible = self._submission([0, 0], [True, True])
        frame.loc[1, "die_row"] = 0
        with self.assertRaises(AssertionError):
            final.check_submission(frame, eligible)


if __name__ == "__main__":
    unittest.main()
