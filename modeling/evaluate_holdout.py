"""Fit on all training wafers and evaluate once on the labeled test wafers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.models import ModelConfig, fit_model, make_model
from modeling.run_experiments import feature_sets, load_cache
from modeling.validation import classification_metrics, eligible_rows


def _positive_weight(y: np.ndarray, mode: str) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    ratio = negatives / positives
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[
        mode
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("test_cache", type=Path)
    parser.add_argument("experiment_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=("model_a_linear", "model_b_linear"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = eligible_rows(load_cache(args.train_cache))
    test = eligible_rows(load_cache(args.test_cache))
    summary_records = json.loads(args.experiment_summary.read_text(encoding="utf-8"))
    summaries = {record["experiment"]: record for record in summary_records}
    experiment_map = {
        item.name: item for item in feature_sets(train, boosted_backend="lightgbm")
    }
    y_train = train["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for name in args.experiments:
        if name not in summaries or name not in experiment_map:
            raise ValueError(f"Experiment {name} is missing from inputs")
        saved = summaries[name]
        experiment = experiment_map[name]
        backend = str(saved["backend"])
        threshold = float(saved["threshold"])
        weight_mode = str(saved["weight_mode"])
        positive_weight = _positive_weight(y_train, weight_mode)
        missing = sorted(set(experiment.feature_columns).difference(test.columns))
        if missing:
            raise ValueError(f"Test cache is missing features: {missing[:10]}")

        x_train = train.loc[:, experiment.feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        x_test = test.loc[:, experiment.feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        print(
            f"Fitting {name} on {len(train)} eligible training dies; "
            f"evaluating {len(test)} held-out dies"
        )
        model = make_model(
            ModelConfig(
                backend=backend,
                positive_weight=positive_weight,
                random_state=args.random_state,
            )
        )
        fit_model(model, x_train, y_train, positive_weight)
        probability = model.predict_proba(x_test)[:, 1]
        metrics = classification_metrics(y_test, probability, threshold)
        metrics.update(
            {
                "experiment": name,
                "backend": backend,
                "features": len(experiment.feature_columns),
                "train_eligible_rows": len(train),
                "test_eligible_rows": len(test),
                "test_failures": int(y_test.sum()),
                "train_oof_average_precision": float(saved["average_precision"]),
                "train_oof_fail_f1": float(saved["fail_f1"]),
            }
        )
        result_rows.append(metrics)
        prediction = test[["wafer_id", "die_row", "die_col", "label"]].copy()
        prediction["experiment"] = name
        prediction["probability"] = probability
        prediction["predicted_label"] = (probability >= threshold).astype(np.int8)
        prediction_frames.append(prediction)
        joblib.dump(model, args.output_dir / f"{name}.joblib")
        print(
            f"  test AP={metrics['average_precision']:.5f}, "
            f"fail F1={metrics['fail_f1']:.5f}, "
            f"precision={metrics['fail_precision']:.5f}, "
            f"recall={metrics['fail_recall']:.5f}"
        )

    result = pd.DataFrame(result_rows).sort_values(
        ["average_precision", "fail_f1"], ascending=False
    )
    result.to_csv(args.output_dir / "holdout_summary.csv", index=False)
    (args.output_dir / "holdout_summary.json").write_text(
        json.dumps(result_rows, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        args.output_dir / "holdout_predictions.parquet", index=False
    )
    print("\n", result.to_string(index=False))


if __name__ == "__main__":
    main()
