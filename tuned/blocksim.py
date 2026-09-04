"""How much is left in the block channel, measured by re-running its generator.

``generate_block_readings`` is short enough to reproduce exactly, which makes it
possible to ask a question the real data cannot answer: how well would a
detector do if it *knew where each die's anomaly cluster was seeded*?

That number turns out to be the whole story of this channel.  With the seed
known the readings separate at ROC-AUC about 0.87.  With the seed unknown --
which is the real problem, because it is drawn uniformly per die and correlates
with nothing -- the best detector here reaches about 0.76, and the previous
pipeline's scan bank about 0.74.  The gap between 0.76 and 0.87 is not a
modelling failure; it is the cost of searching 2,000 possible positions for a
bump whose signal-to-noise at the right position is only about 1.6.

Run as a script, this prints the comparison and writes it to JSON.  It takes a
couple of minutes and touches none of the real data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tuned import blocks

K = blocks.READINGS
BASE_MEAN, BASE_STD = 100.0, 15.0
FAIL_SHIFT = 0.3 * BASE_STD
ANOMALOUS_FRACTION = 0.05
SMOOTHING = 5


def simulate(n: int, failing: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """``generate_block_readings``, line for line, plus the seed it used."""
    out = np.empty((n, K))
    seeds = np.zeros(n, dtype=int)
    count = max(1, int(K * ANOMALOUS_FRACTION))
    for i in range(n):
        row = rng.normal(BASE_MEAN, BASE_STD, size=K)
        row = 0.6 * row + 0.4 * uniform_filter1d(row, size=SMOOTHING, mode="nearest")
        if failing[i]:
            seed = int(rng.integers(0, K))
            positions = {seed}
            while len(positions) < count:
                positions.add(int(seed + int(rng.normal(0, K * 0.05))) % K)
            index = np.fromiter(positions, dtype=int)
            row[index] += rng.normal(FAIL_SHIFT, FAIL_SHIFT * 0.3, size=len(index))
            seeds[i] = seed
        out[i] = np.round(row, 2)          # the generator writes "%.2f"
    return out, seeds


def occupancy(trials: int = 4000, seed: int = 0) -> np.ndarray:
    """P(a block ``d`` from the seed is anomalous), from the same sampler.

    The positions are drawn with replacement until 100 distinct ones exist, so
    the profile is a Gaussian thinned by saturation near the centre -- close
    enough to a Gaussian of standard deviation about 104 that the detector's own
    envelope bank covers it.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros(K)
    count = max(1, int(K * ANOMALOUS_FRACTION))
    for _ in range(trials):
        positions = {0}
        while len(positions) < count:
            positions.add(int(rng.normal(0, K * 0.05)) % K)
        total[np.fromiter(positions, dtype=int)] += 1
    return total / trials


def oracle_auc(readings: np.ndarray, label: np.ndarray, seeds: np.ndarray,
               profile: np.ndarray) -> tuple[float, float]:
    """Matched filter evaluated at the true seed, and at the best-guess seed."""
    residual = (readings - BASE_MEAN) / BASE_STD
    spectrum = np.fft.rfft(residual, axis=1)
    correlation = np.fft.irfft(
        spectrum * np.conj(np.fft.rfft(profile))[None, :], n=K, axis=1)
    correlation /= correlation[label == 0].std()
    at_seed = correlation[np.arange(len(label)), seeds]
    return (float(roc_auc_score(label, at_seed)),
            float(roc_auc_score(label, correlation.max(axis=1))))


def combined_auc(columns: dict, label: np.ndarray) -> float:
    x = np.column_stack([np.asarray(columns[k], dtype=np.float64)
                         for k in sorted(columns)])
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.05, max_iter=2000))
    probability = cross_val_predict(model, x, label, cv=4,
                                    method="predict_proba")[:, 1]
    return float(roc_auc_score(label, probability))


def run(n_pass: int, n_fail: int, seed: int, output: Path | None) -> dict:
    rng = np.random.default_rng(seed)
    label = np.r_[np.zeros(n_pass, dtype=int), np.ones(n_fail, dtype=int)]
    readings, seeds = simulate(len(label), label.astype(bool), rng)
    profile = occupancy(seed=seed)

    known, unknown = oracle_auc(readings, label, seeds, profile)
    noise = blocks.NoiseModel.fit(readings)
    derived, _ = blocks.engineer(readings, noise)

    result = {
        "dies": int(len(label)),
        "failures": int(label.sum()),
        "matched_filter_at_the_true_seed": known,
        "matched_filter_maximised_over_seeds": unknown,
        "derived_likelihood_ratio_bank": combined_auc(derived, label),
        "occupancy_effective_sd": float(np.sqrt(
            (np.roll(profile, K // 2) * (np.arange(K) - K // 2) ** 2).sum()
            / profile.sum())),
    }
    try:
        from mrf import block as mrf_block
        import pandas as pd

        strings = pd.Series(" ".join(f"{v:.2f}" for v in row) for row in readings)
        previous = mrf_block.engineer(strings)
        result["previous_scan_bank"] = combined_auc(
            {c: previous[c].to_numpy() for c in previous.columns}, label)
    except ImportError:
        pass

    for key, value in result.items():
        print(f"  {key:38s} {value:.4f}" if isinstance(value, float)
              else f"  {key:38s} {value}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pass", type=int, default=8_000)
    parser.add_argument("--n-fail", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path,
                        default=Path("results/tuned_ceiling/block_simulation.json"))
    args = parser.parse_args()
    run(args.n_pass, args.n_fail, args.seed, args.output)


if __name__ == "__main__":
    main()
