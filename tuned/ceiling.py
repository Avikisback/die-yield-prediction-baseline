"""How well could *any* model have done, and where does the rest of the loss go?

A number on its own does not say whether a model is good.  This module builds
the comparison that does, by giving a scorer the things the generator knows and
a real model cannot:

``true_prior``
    the generator's own per-die failure probability, ``rate_w * (1 + 3*density
    + 1.5*radius)`` clipped at 0.4, using the wafer's actual drawn rate.  Any
    model that sees only pre-test information is bounded by this.

``oracle_direction``
    the parametric score built from the generator's own ``fail_shift`` and
    ``base_std`` instead of from fitted mean differences, so nothing is lost to
    estimating 500 numbers from 6,500 failures.

``bayes_oracle``
    both of the above, plus the block channel, combined through the same
    additive head fitted in-sample.  This is the practical ceiling: every
    parameter of the generator is known and only the genuinely unknowable
    latents -- which failures were drawn as marginal, where each die's block
    anomaly was seeded -- remain hidden.

Subtracting the rungs apart says where the remaining loss actually sits, which
is the part worth reporting: how much is the unknown wafer rate, how much is
estimating the direction, and how much is the marginal-failure mechanism that
no amount of modelling can undo.

Nothing in ``tuned.pipeline`` imports this module.  ``input/ground_truth.parquet``
is written by ``tuned.genstream`` and is read here only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import generate_data as reference
from modeling.validation import best_f1_threshold, classification_metrics
from tuned import channels, head as head_module
from tuned.cache import eligible_mask, load
from tuned.pipeline import CAP, Fusion, LOG_SHAPE

KEYS = ["wafer_id", "die_row", "die_col"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def true_direction(config_path: Path):
    """The generator's own discriminant, straight out of ``config.yaml``.

    Returns the diagonal weights ``fail_shift / base_std**2``, the per-feature
    separation ``fail_shift / base_std``, the feature names in the order the
    generator defines them, and each feature's ``base_mean``.
    """
    config = reference.load_config(str(config_path))
    names = [f["name"] for f in config["features"]]
    shift = np.array([f["fail_shift"] for f in config["features"]])
    sd = np.array([f["base_std"] for f in config["features"]])
    mean = np.array([f["base_mean"] for f in config["features"]])
    return shift / sd ** 2, shift / sd, names, mean


def score_with_head(design: pd.DataFrame, y: np.ndarray, offset: np.ndarray,
                    smooth: tuple[str, ...]) -> np.ndarray:
    """Fit the additive head in-sample and return its probabilities.

    In-sample is deliberate: an oracle is meant to be an upper bound, and the
    terms fitted here are one-dimensional smooths over 150,000 rows, where the
    difference between in-sample and out-of-fold is in the fourth decimal.
    """
    head = head_module.AdditiveHead(smooth=smooth, linear=(), n_bases=10,
                                    lam_smooth=20.0).fit(design, y, offset)
    return head.predict_proba(design, offset)


def evaluate(name: str, y: np.ndarray, probability: np.ndarray,
             note: str) -> dict:
    threshold = best_f1_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)
    metrics.update({"level": name, "note": note})
    return metrics


def run(cache_dir: Path, ground_truth: Path, config_path: Path,
        output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load(cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)

    weights, separation, names, mean = true_direction(config_path)
    if names != parametric:
        order = {n: i for i, n in enumerate(names)}
        index = np.array([order[n] for n in parametric])
        weights, separation, mean = weights[index], separation[index], mean[index]

    truth = pd.read_parquet(ground_truth)
    merged = frame[KEYS].merge(truth, on=KEYS, how="left", validate="one_to_one")
    if merged[["wafer_base_rate"]].isna().any().any():
        raise ValueError("ground truth does not cover every die in the cache")

    eligible = eligible_mask(frame)
    y = frame.loc[eligible, "label"].to_numpy(dtype=np.int8)
    wafer = frame.loc[eligible, "wafer_id"].astype(str).to_numpy()
    true_prob = merged.loc[eligible, "true_fail_prob"].to_numpy(dtype=np.float64)
    true_rate = merged.loc[eligible, "wafer_base_rate"].to_numpy(dtype=np.float64)
    marginal = merged.loc[eligible, "is_marginal"].to_numpy(dtype=bool)

    all_rows = np.ones(len(frame), dtype=bool)
    fitted = Fusion().fit(frame, x, all_rows)
    scored = fitted.score_frame(frame, x, eligible)
    block_score = scored["block_score"].to_numpy()
    model_probability = fitted.predict_proba(frame, x, eligible)
    recovered = fitted.last_rates_

    oracle_z = channels.project(x, mean, weights)[eligible]
    oracle_z = (oracle_z - oracle_z[y == 0].mean()) / oracle_z[y == 0].std()

    rows: list[dict] = []
    rows.append(evaluate("model", y, model_probability,
                         "the fitted model, all training wafers in-sample"))
    rows.append(evaluate("true_prior", y, true_prob,
                         "generator's own per-die probability, no die evidence"))

    design = pd.DataFrame({"parametric_score": scored["parametric_score"].to_numpy(),
                           "block_score": block_score,
                           "oracle_score": oracle_z},
                          index=np.arange(len(y)))
    prior_offset = _logit(true_prob)
    rows.append(evaluate(
        "true_prior_plus_fitted_evidence", y,
        score_with_head(design, y, prior_offset,
                        ("parametric_score", "block_score")),
        "true wafer rate, fitted channels: isolates the rate-recovery loss"))
    rows.append(evaluate(
        "bayes_oracle", y,
        score_with_head(design, y, prior_offset, ("oracle_score", "block_score")),
        "true wafer rate and the generator's own parametric direction"))
    rows.append(evaluate(
        "oracle_direction_only", y,
        score_with_head(design, y,
                        _logit(np.clip(fitted.overall_rate_
                                       * scored["hazard"].to_numpy(), 1e-9, CAP)),
                        ("oracle_score", "block_score")),
        "generator's parametric direction, population rate: isolates estimation"))

    hard = ~marginal
    keep = (y == 0) | hard
    rows.append(evaluate(
        "model_hard_failures_only", y[keep], model_probability[keep],
        "the same model scored only against failures that got the full shift"))

    order = sorted(set(wafer))
    actual_fraction = np.array([y[wafer == w].mean() for w in order])
    implied_fraction = np.array(
        [model_probability[wafer == w].mean() for w in order])

    diagnostics = {
        "wafer_rate_recovery_r": float(np.corrcoef(
            [recovered[w] for w in sorted(set(wafer))],
            [true_rate[wafer == w][0] for w in sorted(set(wafer))])[0, 1]),
        "direction_recovery_r": float(np.corrcoef(
            fitted.diagonal_.separation, separation)[0, 1]),
        "oracle_score_auc": float(roc_auc_score(y, oracle_z)),
        "fitted_score_auc": float(roc_auc_score(
            y, scored["parametric_score"].to_numpy())),
        "block_score_auc": float(roc_auc_score(y, block_score)),
        "block_score_ap": float(average_precision_score(y, block_score)),
        "marginal_fraction_of_failures": float(marginal[y == 1].mean()),
        "hazard_reconstruction_r": float(np.corrcoef(
            frame.loc[eligible, LOG_SHAPE].to_numpy(),
            np.log(np.maximum(true_prob / true_rate, 1e-12)))[0, 1]),
        "prior_scale": float(fitted.prior_scale_),
        "overall_rate": float(fitted.overall_rate_),
        # The same question both pipelines actually answer: from a wafer's
        # unlabelled scores, what fraction of its eligible dies will fail?
        "wafer_fraction_recovery_r": float(
            np.corrcoef(implied_fraction, actual_fraction)[0, 1]),
        # A correlation says the ordering is right; this says the level is too.
        "wafer_fraction_mean_ratio": float(
            implied_fraction.mean() / max(actual_fraction.mean(), 1e-12)),
        "wafer_rate_mean_ratio": float(
            np.mean([recovered[w] for w in order])
            / max(np.mean([true_rate[wafer == w][0] for w in order]), 1e-12)),
    }

    mrf_rates = Path("results/mrf_rerun/wafer_rates.json")
    if mrf_rates.exists():
        payload = json.loads(mrf_rates.read_text(encoding="utf-8"))
        table = payload.get("B_stacked_cal") or next(iter(payload.values()))
        if set(order).issubset(table):
            diagnostics["mrf_wafer_fraction_recovery_r"] = float(np.corrcoef(
                [table[w] for w in order], actual_fraction)[0, 1])

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "ceiling.csv", index=False)
    (output_dir / "ceiling.json").write_text(
        json.dumps({"levels": rows, "diagnostics": diagnostics}, indent=2,
                   allow_nan=True), encoding="utf-8")
    per_wafer = pd.DataFrame({
        "wafer_id": sorted(set(wafer)),
        "true_rate": [true_rate[wafer == w][0] for w in sorted(set(wafer))],
        "recovered_rate": [recovered[w] for w in sorted(set(wafer))],
    })
    per_wafer.to_csv(output_dir / "wafer_rate_recovery.csv", index=False)

    show = ["level", "average_precision", "roc_auc", "fail_f1", "fail_precision",
            "fail_recall", "note"]
    print(table[show].to_string(index=False))
    print("\ndiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key:34s} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("input/ground_truth.parquet"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    run(args.cache_dir, args.ground_truth, args.config, args.output_dir)


if __name__ == "__main__":
    main()
