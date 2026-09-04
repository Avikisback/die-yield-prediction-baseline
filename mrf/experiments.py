"""Model A vs Model B on identical wafer-grouped folds.

Fold construction, threshold selection and metric definitions are imported from
``modeling.validation`` -- the same code the earlier baseline was scored with --
so every number here is directly comparable to the published baseline table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    eligible_rows,
    repeated_stratified_group_folds,
)
from mrf import models
from mrf.cache import load
from mrf.calibrate import MixtureReference, prior_shift

IDENTIFIERS = ["wafer_id", "die_row", "die_col", "old_label", "label"]


@dataclass(frozen=True)
class Experiment:
    name: str
    backend: str
    prefixes: tuple
    calibrate: bool = False
    note: str = ""


CATALOGUE = (
    # --- each resolution on its own, to size what it carries ---------------
    Experiment("block_only", "linear", ("block_",),
               note="sub-die readings alone"),
    Experiment("parametric_only", "linear", ("feature_",),
               note="die measurements alone"),

    # --- Model A: die measurements plus spatial context --------------------
    Experiment("A_flat", "linear", ("feature_", "spatial_"),
               note="Model A, one logistic over all 513 inputs (baseline recipe)"),
    Experiment("A_stacked", "stacked", ("feature_", "spatial_"),
               note="Model A, diagonal parametric score + logistic head"),
    Experiment("A_stacked_cal", "stacked", ("feature_", "spatial_"), calibrate=True,
               note="Model A, plus the recovered per-wafer rate"),

    # --- Model B: Model A plus the block readings --------------------------
    Experiment("B_flat", "linear", ("feature_", "spatial_", "block_"),
               note="Model B, one logistic over all inputs (baseline recipe)"),
    Experiment("B_stacked", "stacked", ("feature_", "spatial_", "block_"),
               note="Model B, diagonal parametric score + logistic head"),
    Experiment("B_stacked_cal", "stacked", ("feature_", "spatial_", "block_"),
               calibrate=True,
               note="Model B, plus the recovered per-wafer rate"),

    # --- ablations kept on the record --------------------------------------
    Experiment("B_detrended", "stacked", ("dz_", "dstat_", "spatial_", "block_"),
               note="ablation: per-wafer gradient removal (rejected)"),
    Experiment("B_boost", "lightgbm", ("feature_", "spatial_", "block_"),
               note="ablation: gradient boosting on the same inputs"),
)


def select(frame, prefixes):
    columns = [c for c in frame.columns if c.startswith(prefixes)]
    if not columns:
        raise ValueError("No columns matched " + repr(prefixes))
    return sorted(columns)


def run_one(frame, experiment, folds, weight_mode, random_state, C, alpha):
    columns = select(frame, experiment.prefixes)
    x = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["wafer_id"].astype(str).to_numpy()

    total = np.zeros(len(frame), dtype=np.float64)
    seen = np.zeros(len(frame), dtype=np.int16)
    fold_rows = []
    for fold in folds:
        weight = models.positive_weight(y[fold.train_index], weight_mode)
        model = models.make_model(
            experiment.backend,
            random_state=random_state + fold.repeat * 100 + fold.fold,
            C=C,
        )
        models.fit(model, x.iloc[fold.train_index], y[fold.train_index], weight)
        probability = model.predict_proba(x.iloc[fold.validation_index])[:, 1]
        total[fold.validation_index] += probability
        seen[fold.validation_index] += 1
        fold_rows.append(
            {
                "experiment": experiment.name,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "validation_wafers": int(len(np.unique(groups[fold.validation_index]))),
                "validation_rows": int(len(fold.validation_index)),
                "validation_failures": int(y[fold.validation_index].sum()),
                "average_precision": float(
                    average_precision_score(y[fold.validation_index], probability)
                ),
            }
        )
    if np.any(seen == 0):
        raise AssertionError("A die received no out-of-fold prediction")
    probability = total / seen

    rates = {}
    if experiment.calibrate:
        # Reference densities come from out-of-fold scores and are evaluated
        # leave-one-wafer-out, so a wafer's own labels never inform its own
        # rate -- only its own scores do.
        # The mixture is fitted in log-odds space.  Probabilities for a 4%
        # positive rate crowd against zero, where a fixed kernel bandwidth
        # smears the two components into each other.
        score = np.log(np.clip(probability, 1e-9, 1 - 1e-9)
                       / (1 - np.clip(probability, 1e-9, 1 - 1e-9)))
        reference = MixtureReference.fit(score, y, groups)
        probability, rates = prior_shift(
            score, probability, groups, reference, leave_out=True, alpha=alpha
        )

    threshold = best_f1_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)
    metrics.update(
        {
            "experiment": experiment.name,
            "backend": experiment.backend,
            "calibrated": experiment.calibrate,
            "features": len(columns),
            "eligible_rows": len(frame),
            "eligible_failures": int(y.sum()),
            "wafers": int(frame["wafer_id"].nunique()),
            "weight_mode": weight_mode,
            "C": C,
            "alpha": alpha if experiment.calibrate else 0.0,
            "note": experiment.note,
        }
    )
    predictions = frame[["wafer_id", "die_row", "die_col", "label"]].copy()
    predictions["experiment"] = experiment.name
    predictions["oof_probability"] = probability
    return metrics, predictions, fold_rows, rates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--positive-weight", default="sqrt",
                        choices=("none", "sqrt", "balanced"))
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="shrinkage on the per-wafer log-odds offset")
    parser.add_argument("--experiments", nargs="*")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chosen = list(CATALOGUE)
    if args.experiments:
        wanted = set(args.experiments)
        chosen = [e for e in chosen if e.name in wanted]
        missing = wanted.difference(e.name for e in chosen)
        if missing:
            raise ValueError("Unknown experiments: " + repr(sorted(missing)))

    prefixes = tuple({p for e in chosen for p in e.prefixes})
    probe = pd.read_parquet(sorted(Path(args.cache_dir).glob("part-*.parquet"))[0])
    columns = IDENTIFIERS + sorted(c for c in probe.columns if c.startswith(prefixes))
    print("Loading %d columns from %s" % (len(columns), args.cache_dir), flush=True)
    frame = eligible_rows(load(args.cache_dir, columns=columns))
    print("  %d eligible dies, %d failures (%.3f%%), %d wafers"
          % (len(frame), int(frame["label"].sum()), 100 * frame["label"].mean(),
             frame["wafer_id"].nunique()), flush=True)

    folds = repeated_stratified_group_folds(
        frame["label"].to_numpy(dtype=np.int8),
        frame["wafer_id"].astype(str).to_numpy(),
        n_splits=args.n_splits,
        repeats=args.repeats,
        random_state=args.random_state,
    )

    summaries, predictions, fold_rows, all_rates = [], [], [], {}
    for experiment in chosen:
        started = time.perf_counter()
        print("\n%s: %s" % (experiment.name, experiment.note), flush=True)
        summary, prediction, rows, rates = run_one(
            frame, experiment, folds, args.positive_weight, args.random_state,
            args.C, args.alpha
        )
        summary["seconds"] = round(time.perf_counter() - started, 1)
        summaries.append(summary)
        predictions.append(prediction)
        fold_rows.extend(rows)
        if rates:
            all_rates[experiment.name] = rates
        print("  AP=%.4f  ROC-AUC=%.4f  fail F1=%.4f  P=%.3f R=%.3f  (%ss)"
              % (summary["average_precision"], summary["roc_auc"], summary["fail_f1"],
                 summary["fail_precision"], summary["fail_recall"], summary["seconds"]),
              flush=True)

    table = pd.DataFrame(summaries).sort_values("average_precision", ascending=False)
    table.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "oof_predictions.parquet", index=False
    )
    if all_rates:
        (args.output_dir / "wafer_rates.json").write_text(
            json.dumps(all_rates, indent=2), encoding="utf-8"
        )
    show = ["experiment", "features", "average_precision", "roc_auc", "fail_f1",
            "fail_precision", "fail_recall", "accuracy", "seconds"]
    print("\n" + table[show].to_string(index=False))


if __name__ == "__main__":
    main()
