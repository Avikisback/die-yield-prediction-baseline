"""Wafer-grouped cross-validation over the catalogue.

Fold construction, threshold selection and every metric come from
``modeling.validation`` -- the code the published baseline and ``mrf`` were both
scored with -- so the numbers here drop straight into the same table.

Two things are done differently from ``mrf.experiments`` and both are forced by
the model rather than chosen.

**The fit sees the pre-test failures; the score never does.**  Folds are built
over eligible dies exactly as before, but a fold's *training* half is expanded
to every die on those wafers, because ``old_label == 1`` dies are extra labelled
positives for the parametric direction.  Scoring stays on eligible dies only.

**The wafer rate is estimated on the validation wafers themselves.**  It uses no
labels, so a held-out wafer measuring its own rate is exactly what happens at
prediction time -- there is nothing to hold out from.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    repeated_stratified_group_folds,
)
from tuned.cache import eligible_mask, load
from tuned.pipeline import Fusion
from tuned.select import load_chosen

IDENTIFIERS = ["wafer_id", "die_row", "die_col", "old_label", "label"]


@dataclass(frozen=True)
class Experiment:
    name: str
    note: str
    options: dict = field(default_factory=dict)


CATALOGUE = (
    # --- what each piece carries on its own ---------------------------------
    Experiment("prior_only", "pre-test hazard and recovered wafer rate, no die evidence",
               dict(use_parametric=False, use_block=False)),
    Experiment("block_only", "sub-die readings alone",
               dict(use_parametric=False, correct_prior=False, use_rate=False,
                    use_hazard=False)),
    Experiment("parametric_only", "die measurements alone",
               dict(use_block=False, correct_prior=False, use_rate=False,
                    use_hazard=False)),

    # --- Model A: die measurements plus pre-test spatial context -------------
    Experiment("A_linear", "Model A, one coefficient per channel (no spline)",
               dict(use_block=False, use_rate=False, smooth_evidence=False)),
    Experiment("A_smooth", "Model A, smooth per-channel log-likelihood ratios",
               dict(use_block=False, use_rate=False)),
    Experiment("A_full", "Model A, plus the recovered per-wafer rate",
               dict(use_block=False)),

    # --- Model B: Model A plus the block readings ----------------------------
    Experiment("B_linear", "Model B, one coefficient per channel (no spline)",
               dict(use_rate=False, smooth_evidence=False)),
    Experiment("B_smooth", "Model B, smooth per-channel log-likelihood ratios",
               dict(use_rate=False)),
    Experiment("B_full", "Model B, smooth ratios plus the recovered wafer rate",
               dict()),

    # --- ablations kept on the record ----------------------------------------
    Experiment("B_no_old_fails", "ablation: parametric direction without pre-test fails",
               dict(use_old_fails=False)),
    Experiment("B_no_prior_correction", "ablation: hazard shape taken as exact",
               dict(correct_prior=False)),
    Experiment("B_two_rounds", "ablation: refit the head against the recovered rates",
               dict(rounds=2)),
    Experiment("B_three_rounds", "ablation: refit it twice",
               dict(rounds=3)),
    Experiment("B_keep_density_offset",
               "ablation: leave the generator's density term in the parametric score",
               dict(subtract_density=False)),
    Experiment("B_evidence_level_none",
               "ablation: leave the head's constant on the prior side",
               dict(evidence_level="none")),
    Experiment("B_evidence_level_mean_exp",
               "ablation: move the constant so E[exp(evidence)] = 1 over passes",
               dict(evidence_level="mean_exp")),
)


def run_one(frame, x, experiment, folds, eligible_index, wafer, y, base=None):
    # The experiment's own switches win over the selected constants; the two
    # never overlap, but being explicit costs nothing.
    fusion_kwargs = {**(base or {}), **experiment.options}
    total = np.zeros(len(eligible_index), dtype=np.float64)
    seen = np.zeros(len(eligible_index), dtype=np.int16)
    fold_rows = []

    for fold in folds:
        train_wafers = set(wafer[fold.train_index])
        train_rows = np.isin(frame["wafer_id"].astype(str).to_numpy(),
                             list(train_wafers))
        if np.any(np.diff(fold.validation_index) <= 0):
            raise AssertionError(
                "validation indices must be ascending: predictions come back in "
                "row order, not in fold order")
        score_rows = np.zeros(len(frame), dtype=bool)
        score_rows[eligible_index[fold.validation_index]] = True

        model = Fusion(**fusion_kwargs).fit(frame, x, train_rows)
        probability = model.predict_proba(frame, x, score_rows)
        total[fold.validation_index] += probability
        seen[fold.validation_index] += 1
        fold_rows.append({
            "experiment": experiment.name,
            "repeat": fold.repeat,
            "fold": fold.fold,
            "validation_wafers": int(len(set(wafer[fold.validation_index]))),
            "validation_rows": int(len(fold.validation_index)),
            "validation_failures": int(y[fold.validation_index].sum()),
            "average_precision": float(
                average_precision_score(y[fold.validation_index], probability)),
        })

    if np.any(seen == 0):
        raise AssertionError("A die received no out-of-fold prediction")
    probability = total / seen
    threshold = best_f1_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)
    metrics.update({
        # Predicted failures over observed ones.  A model can rank well and
        # still be badly wrong about how many failures there are, and which of
        # those two a fab needs is not for the ranking metric to decide.
        "calibration": float(probability.sum() / max(y.sum(), 1)),
        "experiment": experiment.name,
        "note": experiment.note,
        "eligible_rows": int(len(y)),
        "eligible_failures": int(y.sum()),
        "wafers": int(len(set(wafer))),
        **{f"opt_{k}": v for k, v in experiment.options.items()},
    })
    predictions = pd.DataFrame({
        "wafer_id": wafer,
        "label": y,
        "experiment": experiment.name,
        "oof_probability": probability,
    })
    return metrics, predictions, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiments", nargs="*")
    parser.add_argument("--selection", type=Path,
                        default=Path("results/tuned_selection/selection.json"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(CATALOGUE)
    if args.experiments:
        wanted = set(args.experiments)
        chosen = [e for e in chosen if e.name in wanted]
        missing = wanted.difference(e.name for e in chosen)
        if missing:
            raise ValueError(f"Unknown experiments: {sorted(missing)}")

    print(f"Loading {args.cache_dir}", flush=True)
    frame = load(args.cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)
    eligible = eligible_mask(frame)
    eligible_index = np.flatnonzero(eligible)
    y = frame.loc[eligible, "label"].to_numpy(dtype=np.int8)
    wafer = frame.loc[eligible, "wafer_id"].astype(str).to_numpy()
    print(f"  {len(frame)} dies, {eligible.sum()} eligible, {int(y.sum())} failures "
          f"({100 * y.mean():.3f}%), {len(set(wafer))} wafers", flush=True)

    folds = repeated_stratified_group_folds(
        y, wafer, n_splits=args.n_splits, repeats=args.repeats,
        random_state=args.random_state)

    base = load_chosen(args.selection)
    if base:
        print(f"  using selected constants {base}", flush=True)

    summaries, predictions, fold_rows = [], [], []
    for experiment in chosen:
        started = time.perf_counter()
        print(f"\n{experiment.name}: {experiment.note}", flush=True)
        summary, prediction, rows = run_one(
            frame, x, experiment, folds, eligible_index, wafer, y, base)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        summaries.append(summary)
        predictions.append(prediction)
        fold_rows.extend(rows)
        print("  AP=%.4f  ROC-AUC=%.4f  fail F1=%.4f  P=%.3f R=%.3f  (%.0fs)"
              % (summary["average_precision"], summary["roc_auc"], summary["fail_f1"],
                 summary["fail_precision"], summary["fail_recall"],
                 summary["seconds"]), flush=True)

    table = pd.DataFrame(summaries).sort_values("average_precision", ascending=False)
    table.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "oof_predictions.parquet", index=False)
    show = ["experiment", "average_precision", "roc_auc", "fail_f1",
            "fail_precision", "fail_recall", "accuracy", "seconds"]
    print("\n" + table[show].to_string(index=False))


if __name__ == "__main__":
    main()
