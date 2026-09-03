"""Run leakage-safe out-of-fold Model A/Model B experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from modeling.models import ModelConfig, fit_model, make_model, resolve_backend
from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    eligible_rows,
    repeated_stratified_group_folds,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    backend: str
    feature_columns: tuple[str, ...]


def load_cache(cache_dir: Path) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No part-*.parquet files found in {cache_dir}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def feature_sets(frame: pd.DataFrame, boosted_backend: str) -> list[Experiment]:
    parametric = tuple(sorted(c for c in frame if c.startswith("feature_")))
    spatial = tuple(sorted(c for c in frame if c.startswith("spatial_")))
    block = tuple(sorted(c for c in frame if c.startswith("block_")))
    if not parametric:
        raise ValueError("No feature_* columns found in the cache")
    if not spatial:
        raise ValueError("No spatial_* columns found in the cache")

    experiments = [
        Experiment("parametric_linear", "linear", parametric),
        Experiment("model_a_linear", "linear", parametric + spatial),
        Experiment("model_a_boost", boosted_backend, parametric + spatial),
    ]
    if block:
        experiments.extend(
            (
                Experiment("block_only_boost", boosted_backend, block),
                Experiment("model_b_linear", "linear", parametric + spatial + block),
                Experiment("model_b_boost", boosted_backend, parametric + spatial + block),
            )
        )
    return experiments


def _positive_weight(y: np.ndarray, mode: str) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        raise ValueError("No eligible failures are present")
    ratio = negatives / positives
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(ratio))
    if mode == "balanced":
        return float(ratio)
    raise ValueError(f"Unknown positive-weight mode: {mode}")


def run_experiment(
    frame: pd.DataFrame,
    experiment: Experiment,
    folds,
    weight_mode: str,
    random_state: int,
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    x = frame.loc[:, experiment.feature_columns].replace([np.inf, -np.inf], np.nan)
    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["wafer_id"].astype(str).to_numpy()
    probability_sum = np.zeros(len(frame), dtype=np.float64)
    prediction_count = np.zeros(len(frame), dtype=np.int16)
    fold_rows: list[dict[str, object]] = []

    for fold in folds:
        train_index = fold.train_index
        validation_index = fold.validation_index
        weight = _positive_weight(y[train_index], weight_mode)
        model = make_model(
            ModelConfig(
                backend=experiment.backend,
                positive_weight=weight,
                random_state=random_state + fold.repeat * 100 + fold.fold,
            )
        )
        fit_model(model, x.iloc[train_index], y[train_index], weight)
        fold_probability = model.predict_proba(x.iloc[validation_index])[:, 1]
        probability_sum[validation_index] += fold_probability
        prediction_count[validation_index] += 1
        fold_rows.append(
            {
                "experiment": experiment.name,
                "backend": resolve_backend(experiment.backend),
                "repeat": fold.repeat,
                "fold": fold.fold,
                "train_wafers": int(len(np.unique(groups[train_index]))),
                "validation_wafers": int(len(np.unique(groups[validation_index]))),
                "validation_rows": int(len(validation_index)),
                "validation_failures": int(y[validation_index].sum()),
                "positive_weight": weight,
                "average_precision": float(
                    average_precision_score(y[validation_index], fold_probability)
                ),
            }
        )

    if np.any(prediction_count == 0):
        raise AssertionError("At least one eligible row did not receive an OOF prediction")
    probability = probability_sum / prediction_count
    threshold = best_f1_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)
    metrics.update(
        {
            "experiment": experiment.name,
            "backend": resolve_backend(experiment.backend),
            "features": len(experiment.feature_columns),
            "eligible_rows": len(frame),
            "eligible_failures": int(y.sum()),
            "wafers": int(frame["wafer_id"].nunique()),
            "weight_mode": weight_mode,
        }
    )
    identifiers = frame[["wafer_id", "die_row", "die_col", "label"]].copy()
    identifiers["experiment"] = experiment.name
    identifiers["oof_probability"] = probability
    identifiers["oof_repeats"] = prediction_count
    return metrics, identifiers, fold_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--backend",
        choices=("auto", "lightgbm", "hist_gradient_boosting"),
        default="auto",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--positive-weight", choices=("none", "sqrt", "balanced"), default="sqrt"
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional experiment names; by default all applicable experiments run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = eligible_rows(load_cache(args.cache_dir))
    if "label" not in frame:
        raise ValueError("A labeled train cache is required")

    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["wafer_id"].astype(str).to_numpy()
    folds = repeated_stratified_group_folds(
        y,
        groups,
        n_splits=args.n_splits,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    experiments = feature_sets(frame, args.backend)
    if args.experiments:
        requested = set(args.experiments)
        experiments = [item for item in experiments if item.name in requested]
        missing = requested.difference(item.name for item in experiments)
        if missing:
            raise ValueError(f"Unknown or unavailable experiments: {sorted(missing)}")

    summaries: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    all_folds: list[dict[str, object]] = []
    for experiment in experiments:
        print(
            f"Running {experiment.name}: {len(experiment.feature_columns)} features, "
            f"backend={resolve_backend(experiment.backend)}"
        )
        summary, predictions, fold_rows = run_experiment(
            frame,
            experiment,
            folds,
            args.positive_weight,
            args.random_state,
        )
        summaries.append(summary)
        all_predictions.append(predictions)
        all_folds.extend(fold_rows)
        print(
            f"  AP={summary['average_precision']:.5f}, "
            f"fail F1={summary['fail_f1']:.5f}, threshold={summary['threshold']:.5f}"
        )

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["average_precision", "fail_f1"], ascending=False
    )
    summary_frame.to_csv(args.output_dir / "experiment_summary.csv", index=False)
    pd.DataFrame(all_folds).to_csv(
        args.output_dir / "fold_average_precision.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_parquet(
        args.output_dir / "oof_predictions.parquet", index=False
    )
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
    )
    print("\n", summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
