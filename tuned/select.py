"""Choose the three free constants on training out-of-fold scores only.

The model has very few knobs, and all three are about how much freedom to allow
rather than about what to model:

``lam_smooth``  how much curvature the per-channel log-likelihood ratios may have
``n_bases``     how many spline coefficients each of them gets
``block_C``     the ridge on the logistic that collapses 57 block statistics to one

Every one is picked here, on five wafer-grouped folds of the *training* wafers,
and nothing else is tuned anywhere.  The held-out wafers are scored once, by
``tuned.final``, after this has run.

One repeat rather than three, and twelve points rather than a fine grid: each
point costs five model fits over 150,000 dies, and the spread across the whole
grid turns out to be smaller than the gap between any two rows of the results
table.  Spending an hour to sharpen a decision that does not matter would be the
wrong trade; spending it to be able to say the constants were never chosen
against the held-out wafers is the right one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.validation import best_f1_threshold, classification_metrics
from modeling.validation import repeated_stratified_group_folds
from tuned.cache import eligible_mask, load
from tuned.pipeline import Fusion

# The penalty competes with the *unnormalised* log-likelihood over 150,000 dies,
# so values below a few hundred are indistinguishable from no penalty at all.
GRID = {
    "lam_smooth": (20.0, 2_000.0, 50_000.0),
    "n_bases": (6, 10),
    "block_C": (0.02, 0.1),
}


def load_chosen(path: Path) -> dict:
    """The constants ``main`` settled on, or an empty dict if it never ran."""
    if not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))["chosen"]
    return {"lam_smooth": float(payload["lam_smooth"]),
            "n_bases": int(payload["n_bases"]),
            "block_C": float(payload["block_C"])}


def cross_validate(frame, x, eligible_index, wafer, y, folds, **options):
    total = np.zeros(len(y))
    seen = np.zeros(len(y), dtype=np.int16)
    wafer_all = frame["wafer_id"].astype(str).to_numpy()
    for fold in folds:
        train_rows = np.isin(wafer_all, list(set(wafer[fold.train_index])))
        if np.any(np.diff(fold.validation_index) <= 0):
            raise AssertionError("validation indices must be ascending")
        score_rows = np.zeros(len(frame), dtype=bool)
        score_rows[eligible_index[fold.validation_index]] = True
        model = Fusion(**options).fit(frame, x, train_rows)
        total[fold.validation_index] += model.predict_proba(frame, x, score_rows)
        seen[fold.validation_index] += 1
    return total / seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load(args.cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)
    eligible = eligible_mask(frame)
    eligible_index = np.flatnonzero(eligible)
    y = frame.loc[eligible, "label"].to_numpy(dtype=np.int8)
    wafer = frame.loc[eligible, "wafer_id"].astype(str).to_numpy()
    folds = repeated_stratified_group_folds(
        y, wafer, n_splits=args.n_splits, repeats=1,
        random_state=args.random_state)

    rows = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        options = dict(zip(keys, values))
        started = time.perf_counter()
        probability = cross_validate(frame, x, eligible_index, wafer, y, folds,
                                     **options)
        threshold = best_f1_threshold(y, probability)
        metrics = classification_metrics(y, probability, threshold)
        rows.append({**options,
                     "average_precision": metrics["average_precision"],
                     "roc_auc": metrics["roc_auc"],
                     "fail_f1": metrics["fail_f1"],
                     "seconds": round(time.perf_counter() - started, 1)})
        print("  %s  AP=%.4f  F1=%.4f  (%.0fs)"
              % (options, rows[-1]["average_precision"], rows[-1]["fail_f1"],
                 rows[-1]["seconds"]), flush=True)

    table = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    table.to_csv(args.output_dir / "selection.csv", index=False)
    best = table.iloc[0]
    (args.output_dir / "selection.json").write_text(
        json.dumps({"chosen": {k: best[k] for k in keys},
                    "average_precision": float(best["average_precision"]),
                    "grid": {k: list(v) for k, v in GRID.items()},
                    "n_splits": args.n_splits, "repeats": 1}, indent=2),
        encoding="utf-8")
    print("\n" + table.to_string(index=False))
    print("\nchosen:", {k: best[k] for k in keys})


if __name__ == "__main__":
    main()
