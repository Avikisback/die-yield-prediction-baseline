"""Fit one selected experiment on all training wafers and create a submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from modeling.models import ModelConfig, fit_model, make_model
from modeling.run_experiments import feature_sets, load_cache
from modeling.validation import eligible_rows


def _load_experiment_summary(path: Path, name: str) -> dict[str, object]:
    records = json.loads(path.read_text(encoding="utf-8"))
    matches = [record for record in records if record["experiment"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary for {name}, found {len(matches)}")
    return matches[0]


def _positive_weight(y: np.ndarray, mode: str) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        raise ValueError("No eligible training failures are present")
    ratio = negatives / positives
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[
        mode
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("prediction_cache", type=Path)
    parser.add_argument("experiment_summary", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--experiment",
        default="model_b_boost",
        choices=(
            "parametric_linear",
            "model_a_linear",
            "model_a_boost",
            "block_only_boost",
            "model_b_linear",
            "model_b_boost",
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "lightgbm", "hist_gradient_boosting"),
        default="auto",
        help="auto reuses the backend recorded in the experiment summary",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _load_experiment_summary(args.experiment_summary, args.experiment)
    threshold = float(summary["threshold"])
    weight_mode = str(summary["weight_mode"])
    backend = str(summary["backend"]) if args.backend == "auto" else args.backend

    train_all = load_cache(args.train_cache)
    prediction_all = load_cache(args.prediction_cache)
    train = eligible_rows(train_all)
    prediction_eligible = eligible_rows(prediction_all)
    experiment_map = {
        item.name: item for item in feature_sets(train, boosted_backend=backend)
    }
    if args.experiment not in experiment_map:
        raise ValueError(f"Experiment {args.experiment} is unavailable in the cache")
    experiment = experiment_map[args.experiment]
    missing = sorted(set(experiment.feature_columns).difference(prediction_eligible.columns))
    if missing:
        raise ValueError(f"Prediction cache is missing features: {missing[:10]}")

    x_train = train.loc[:, experiment.feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    x_prediction = prediction_eligible.loc[:, experiment.feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = train["label"].to_numpy(dtype=np.int8)
    weight = _positive_weight(y_train, weight_mode)
    model = make_model(
        ModelConfig(backend=backend, positive_weight=weight, random_state=args.random_state)
    )
    fit_model(model, x_train, y_train, weight)
    probabilities = model.predict_proba(x_prediction)[:, 1]

    submission = prediction_all[["wafer_id", "die_row", "die_col"]].copy()
    submission["predicted_label"] = 1
    submission.loc[prediction_eligible.index, "predicted_label"] = (
        probabilities >= threshold
    ).astype(np.int8)
    submission["predicted_label"] = submission["predicted_label"].astype(np.int8)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_csv, index=False)

    model_output = args.model_output or args.output_csv.with_suffix(".joblib")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)
    metadata = {
        "experiment": args.experiment,
        "backend": backend,
        "threshold": threshold,
        "positive_weight": weight,
        "weight_mode": weight_mode,
        "feature_columns": list(experiment.feature_columns),
        "eligible_training_rows": len(train),
        "eligible_prediction_rows": len(prediction_eligible),
        "submission_rows": len(submission),
    }
    model_output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(submission)} predictions to {args.output_csv}")
    print(f"Saved fitted model to {model_output}")


if __name__ == "__main__":
    main()

