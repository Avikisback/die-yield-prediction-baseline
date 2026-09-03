"""Fit the selected models, score the held-out wafers, write the submission.

Test-time scores are produced by the same cross-fitted ensemble that produced
the out-of-fold scores on the training wafers.  That matters for the per-wafer
rate step: its reference densities are estimated from out-of-fold scores, so the
scores it is later asked to explain have to live on the same scale.  A single
model refitted on all the data would be measurably sharper and would bias the
recovered rates.

A separate all-data fit is kept for interpretation only, because that is the
model whose coefficients the report describes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    eligible_rows,
    repeated_stratified_group_folds,
)
from mrf import models
from mrf.cache import load
from mrf.calibrate import MixtureReference, prior_shift
from mrf.experiments import IDENTIFIERS, select

# Raw measurements, not the per-wafer detrended ones.  Detrending was tried and
# rejected: see the ablation in the report.  Regressing each feature on the
# wafer's coordinate basis removes the process gradient, but on real WM-811K maps
# the new failures are themselves clustered against radius and against pre-test
# defect neighbourhoods, so the fit absorbs signal along with the nuisance.
# Scored with the generator's own coefficients, average precision falls from
# 0.5271 on raw values to 0.5065 on detrended ones.
SPECS = {
    "model_a": ("feature_", "spatial_"),
    "model_b": ("feature_", "spatial_", "block_"),
}
BACKEND = "stacked"
ALPHA = 0.5


def _to_logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _align(target: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    """Put held-out scores on the out-of-fold scale, robustly.

    Only the overall location and spread are matched, using every held-out wafer
    at once, so differences *between* wafers -- the thing the rate estimate reads
    -- are untouched.
    """
    def centre_spread(v):
        m = np.median(v)
        return m, max(1.4826 * np.median(np.abs(v - m)), 1e-9)

    tm, ts = centre_spread(target)
    rm, rs = centre_spread(reference_scores)
    return (target - tm) / ts * rs + rm


def run(cache_train, cache_test, cache_predict, output_dir, n_splits, repeats,
        random_state, weight_mode, C):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = tuple({p for spec in SPECS.values() for p in spec})
    probe = pd.read_parquet(sorted(Path(cache_train).glob("part-*.parquet"))[0])
    columns = IDENTIFIERS + sorted(c for c in probe.columns if c.startswith(prefixes))

    train = eligible_rows(load(cache_train, columns=columns))
    test = eligible_rows(load(cache_test, columns=columns))
    predict_all = load(cache_predict, columns=[c for c in columns if c != "label"])
    predict = eligible_rows(predict_all)
    print("train %d / test %d / predict %d eligible dies"
          % (len(train), len(test), len(predict)), flush=True)

    y_train = train["label"].to_numpy(dtype=np.int8)
    y_test = test["label"].to_numpy(dtype=np.int8)
    groups = train["wafer_id"].astype(str).to_numpy()
    folds = repeated_stratified_group_folds(
        y_train, groups, n_splits=n_splits, repeats=repeats, random_state=random_state
    )

    rows, stored = [], {}
    for name, spec in SPECS.items():
        feature_columns = select(train, spec)
        x_train = train.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
        x_test = test.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
        x_predict = predict.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)

        oof = np.zeros(len(train))
        seen = np.zeros(len(train), dtype=np.int16)
        test_total = np.zeros(len(test))
        predict_total = np.zeros(len(predict))
        for fold in folds:
            weight = models.positive_weight(y_train[fold.train_index], weight_mode)
            model = models.make_model(
                BACKEND, random_state=random_state + fold.fold, C=C
            )
            models.fit(model, x_train.iloc[fold.train_index],
                       y_train[fold.train_index], weight)
            oof[fold.validation_index] += model.predict_proba(
                x_train.iloc[fold.validation_index])[:, 1]
            seen[fold.validation_index] += 1
            test_total += model.predict_proba(x_test)[:, 1]
            predict_total += model.predict_proba(x_predict)[:, 1]
        oof /= seen
        p_test = test_total / len(folds)
        p_predict = predict_total / len(folds)

        threshold = best_f1_threshold(y_train, oof)
        rows.append(dict(classification_metrics(y_test, p_test, threshold),
                         model=name, stage="uncalibrated",
                         features=len(feature_columns)))

        oof_score = _to_logit(oof)
        reference = MixtureReference.fit(oof_score, y_train, groups)
        p_test_cal, test_rates = prior_shift(
            _align(_to_logit(p_test), oof_score), p_test,
            test["wafer_id"].astype(str).to_numpy(), reference, leave_out=False,
            alpha=ALPHA)
        p_predict_cal, predict_rates = prior_shift(
            _align(_to_logit(p_predict), oof_score), p_predict,
            predict["wafer_id"].astype(str).to_numpy(), reference, leave_out=False,
            alpha=ALPHA)

        oof_cal, _ = prior_shift(oof_score, oof, groups, reference, leave_out=True,
                                 alpha=ALPHA)
        threshold_cal = best_f1_threshold(y_train, oof_cal)
        rows.append(dict(classification_metrics(y_test, p_test_cal, threshold_cal),
                         model=name, stage="wafer_rate_calibrated",
                         features=len(feature_columns)))

        # An all-data fit, kept only so the report can quote real coefficients.
        explain = models.make_model(BACKEND, random_state=random_state, C=C)
        models.fit(explain, x_train, y_train, models.positive_weight(y_train, weight_mode))
        joblib.dump({"model": explain, "columns": feature_columns,
                     "threshold": float(threshold),
                     "threshold_calibrated": float(threshold_cal)},
                    output_dir / (name + ".joblib"))

        stored[name] = {
            "threshold": float(threshold),
            "threshold_calibrated": float(threshold_cal),
            "test_wafer_rates": test_rates,
            "predict_wafer_rates": predict_rates,
        }
        out = test[["wafer_id", "die_row", "die_col", "label"]].copy()
        out["probability"] = p_test
        out["probability_calibrated"] = p_test_cal
        out.to_parquet(output_dir / (name + "_test_predictions.parquet"), index=False)
        train_out = train[["wafer_id", "die_row", "die_col", "label"]].copy()
        train_out["oof_probability"] = oof
        train_out["oof_probability_calibrated"] = oof_cal
        train_out.to_parquet(output_dir / (name + "_oof_predictions.parquet"), index=False)

        if name == "model_b":
            submission = predict_all[["wafer_id", "die_row", "die_col"]].copy()
            # Dies that already failed the pre-test are failures by definition
            # and never reach the model.
            submission["predicted_label"] = 1
            submission.loc[predict.index, "predicted_label"] = (
                p_predict_cal >= threshold_cal).astype(np.int8)
            submission["predicted_label"] = submission["predicted_label"].astype(np.int8)
            submission.to_csv(output_dir / "submission.csv", index=False)
            print("submission: %d rows, %d predicted failures"
                  % (len(submission), int(submission["predicted_label"].sum())), flush=True)

        latest = rows[-1]
        print("%-9s AP=%.4f -> calibrated AP=%.4f | F1 %.4f -> %.4f"
              % (name, rows[-2]["average_precision"], latest["average_precision"],
                 rows[-2]["fail_f1"], latest["fail_f1"]), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "holdout_summary.csv", index=False)
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "thresholds.json").write_text(
        json.dumps(stored, indent=2), encoding="utf-8")
    show = ["model", "stage", "features", "average_precision", "roc_auc", "fail_f1",
            "fail_precision", "fail_recall", "accuracy"]
    print("\n" + table[show].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_train", type=Path)
    parser.add_argument("cache_test", type=Path)
    parser.add_argument("cache_predict", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--positive-weight", default="sqrt")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    run(args.cache_train, args.cache_test, args.cache_predict, args.output_dir,
        args.n_splits, args.repeats, args.random_state, args.positive_weight, args.C)


if __name__ == "__main__":
    main()
