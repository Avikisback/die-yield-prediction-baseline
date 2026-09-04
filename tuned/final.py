"""Fit on the training wafers, score the held-out ones, write the submission.

``mrf.final`` has to score the held-out wafers with the same cross-fitted
ensemble that produced the out-of-fold scores, because its rate step compares a
wafer's scores against reference densities estimated from those out-of-fold
scores -- put a sharper model's scores in front of it and the recovered rates
move.  It then has to rescale the held-out scores to match, which is a
correction with no principled size.

Nothing here needs that.  The rate is recovered from calibrated likelihood
ratios, which are on an absolute scale by construction: a die whose measurements
are ten times more likely under failure than under passing reports ``log 10``
whether it was scored by one model or fifteen.  So this fits one model on all
160 training wafers and applies it directly.

Out-of-fold predictions are still produced, for one job only: choosing the
decision threshold.  That has to come from data the threshold is not then scored
against.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.validation import (
    best_f1_threshold,
    classification_metrics,
    repeated_stratified_group_folds,
)
from tuned.cache import eligible_mask, load
from tuned.pipeline import Fusion
from tuned.select import load_chosen

SPECS = {
    "model_a": dict(use_block=False),
    "model_b": dict(),
}


def check_submission(submission: pd.DataFrame, eligible: np.ndarray) -> None:
    """The format the problem statement asks for, checked before it is written.

    Every die gets a row -- pre-test failures included, forced to 1 -- the keys
    are unique, and the label is 0 or 1.
    """
    expected = ["wafer_id", "die_row", "die_col", "predicted_label"]
    if list(submission.columns) != expected:
        raise AssertionError(f"submission columns are {list(submission.columns)}")
    if submission.duplicated(["wafer_id", "die_row", "die_col"]).any():
        raise AssertionError("submission has duplicate die keys")
    if not submission["predicted_label"].isin((0, 1)).all():
        raise AssertionError("predicted_label must be 0 or 1")
    forced = submission.loc[~eligible, "predicted_label"]
    if len(forced) and not (forced == 1).all():
        raise AssertionError("pre-test failures must be predicted as failures")


def _load(cache_dir: Path):
    frame = load(cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    return frame.drop(columns=parametric), x


def run(cache_train: Path, cache_test: Path, cache_predict: Path,
        output_dir: Path, n_splits: int, repeats: int, random_state: int,
        selection: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_chosen(selection)
    if base:
        print(f"using selected constants {base}", flush=True)
    train, x_train = _load(cache_train)
    test, x_test = _load(cache_test)
    predict_all, x_predict = _load(cache_predict)

    train_eligible = eligible_mask(train)
    test_eligible = eligible_mask(test)
    predict_eligible = eligible_mask(predict_all)
    train_index = np.flatnonzero(train_eligible)
    y_train = train.loc[train_eligible, "label"].to_numpy(dtype=np.int8)
    y_test = test.loc[test_eligible, "label"].to_numpy(dtype=np.int8)
    groups = train.loc[train_eligible, "wafer_id"].astype(str).to_numpy()
    print("train %d eligible / test %d eligible / predict %d eligible"
          % (train_eligible.sum(), test_eligible.sum(), predict_eligible.sum()),
          flush=True)

    folds = repeated_stratified_group_folds(
        y_train, groups, n_splits=n_splits, repeats=repeats,
        random_state=random_state)

    rows, stored = [], {}
    for name, spec in SPECS.items():
        options = {**base, **spec}
        started = time.perf_counter()
        oof = np.zeros(len(y_train))
        seen = np.zeros(len(y_train), dtype=np.int16)
        for fold in folds:
            train_rows = np.isin(train["wafer_id"].astype(str).to_numpy(),
                                 list(set(groups[fold.train_index])))
            if np.any(np.diff(fold.validation_index) <= 0):
                raise AssertionError("validation indices must be ascending")
            score_rows = np.zeros(len(train), dtype=bool)
            score_rows[train_index[fold.validation_index]] = True
            model = Fusion(**options).fit(train, x_train, train_rows)
            oof[fold.validation_index] += model.predict_proba(
                train, x_train, score_rows)
            seen[fold.validation_index] += 1
        oof /= seen
        threshold = best_f1_threshold(y_train, oof)

        final = Fusion(**options).fit(
            train, x_train, np.ones(len(train), dtype=bool))
        p_test = final.predict_proba(test, x_test, test_eligible)
        test_rates = dict(final.last_rates_)
        p_predict = final.predict_proba(predict_all, x_predict, predict_eligible)
        predict_rates = dict(final.last_rates_)

        metrics = classification_metrics(y_test, p_test, threshold)
        metrics.update({"model": name, "stage": "held_out",
                        "threshold_source": "training out-of-fold",
                        "seconds": round(time.perf_counter() - started, 1)})
        rows.append(metrics)
        stored[name] = {"threshold": float(threshold), "options": options,
                        "test_wafer_rates": test_rates,
                        "predict_wafer_rates": predict_rates,
                        "prior_scale": float(final.prior_scale_),
                        "overall_rate": float(final.overall_rate_)}

        joblib.dump({"model": final, "threshold": float(threshold)},
                    output_dir / f"{name}.joblib")
        out = test.loc[test_eligible, ["wafer_id", "die_row", "die_col", "label"]].copy()
        out["probability"] = p_test
        out.to_parquet(output_dir / f"{name}_test_predictions.parquet", index=False)
        oof_frame = train.loc[train_eligible,
                              ["wafer_id", "die_row", "die_col", "label"]].copy()
        oof_frame["oof_probability"] = oof
        oof_frame.to_parquet(output_dir / f"{name}_oof_predictions.parquet", index=False)

        if name == "model_b":
            submission = predict_all[["wafer_id", "die_row", "die_col"]].copy()
            # Dies that already failed the pre-test are failures by definition
            # and never reach the model.
            predicted = np.ones(len(submission), dtype=np.int8)
            predicted[predict_eligible] = (p_predict >= threshold).astype(np.int8)
            submission["predicted_label"] = predicted
            check_submission(submission, predict_eligible)
            submission.to_csv(output_dir / "submission.csv", index=False)
            print("submission: %d rows, %d predicted failures (%d forced pre-test)"
                  % (len(submission), int(predicted.sum()),
                     int((~predict_eligible).sum())), flush=True)

        print("%-9s AP=%.4f  ROC-AUC=%.4f  fail F1=%.4f  P=%.3f R=%.3f  (%.0fs)"
              % (name, metrics["average_precision"], metrics["roc_auc"],
                 metrics["fail_f1"], metrics["fail_precision"],
                 metrics["fail_recall"], metrics["seconds"]), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "holdout_summary.csv", index=False)
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "thresholds.json").write_text(
        json.dumps(stored, indent=2), encoding="utf-8")
    show = ["model", "stage", "average_precision", "roc_auc", "fail_f1",
            "fail_precision", "fail_recall", "accuracy"]
    print("\n" + table[show].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_train", type=Path)
    parser.add_argument("cache_test", type=Path)
    parser.add_argument("cache_predict", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--selection", type=Path,
                        default=Path("results/tuned_selection/selection.json"))
    args = parser.parse_args()
    run(args.cache_train, args.cache_test, args.cache_predict, args.output_dir,
        args.n_splits, args.repeats, args.random_state, args.selection)


if __name__ == "__main__":
    main()
