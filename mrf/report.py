"""Assemble RESULTS.md from the saved result files, so the write-up cannot drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Published numbers from the earlier baseline in this repository, kept here so
# the comparison is explicit about what it is comparing against.
BASELINE_CV = {
    "model_a_linear": {"average_precision": 0.5137, "fail_f1": 0.5268},
    "model_b_linear": {"average_precision": 0.5758, "fail_f1": 0.5537},
    "block_only_boost": {"average_precision": 0.1243, "fail_f1": 0.1872},
}
BASELINE_HOLDOUT = {
    "model_a_linear": {"average_precision": 0.5320, "fail_f1": 0.5297},
    "model_b_linear": {"average_precision": 0.5934, "fail_f1": 0.5600},
}

LABELS = {
    "block_only": "block readings alone",
    "parametric_only": "die measurements alone",
    "A_flat": "Model A, flat logistic",
    "A_stacked": "Model A, stacked",
    "A_stacked_cal": "**Model A, stacked + wafer rate**",
    "B_flat": "Model B, flat logistic",
    "B_stacked": "Model B, stacked",
    "B_stacked_cal": "**Model B, stacked + wafer rate**",
    "B_detrended": "ablation: per-wafer detrending",
    "B_boost": "ablation: gradient boosting",
}


def table(frame: pd.DataFrame, columns, headers, fmt) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] + ["--:"] * (len(headers) - 1)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column, spec in zip(columns, fmt):
            value = row[column]
            cells.append(spec.format(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(results_dir: Path, final_dir: Path, figures_dir: Path, output: Path):
    cv = pd.read_csv(results_dir / "experiment_summary.csv")
    cv["label"] = cv["experiment"].map(LABELS).fillna(cv["experiment"])
    cv = cv.sort_values("average_precision", ascending=False)

    parts = ["# Results\n",
             "All cross-validation numbers use five wafer-grouped folds repeated three "
             "times, the fold builder and metric code from `modeling/validation.py`, and "
             "the same seed-42 dataset the earlier baseline was scored on. Only dies with "
             "`old_label == 0` are fitted or scored.\n",
             "## Cross-validation, all experiments\n",
             table(cv, ["label", "features", "average_precision", "roc_auc", "fail_f1",
                        "fail_precision", "fail_recall"],
                   ["experiment", "features", "AP", "ROC-AUC", "fail F1",
                    "precision", "recall"],
                   ["{}", "{:.0f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}"]),
             ""]

    rows = []
    for name, baseline in (("A_stacked_cal", "model_a_linear"),
                           ("B_stacked_cal", "model_b_linear")):
        here = cv.loc[cv["experiment"] == name]
        if here.empty:
            continue
        here = here.iloc[0]
        ref = BASELINE_CV[baseline]
        rows.append({
            "model": "Model A" if name.startswith("A") else "Model B",
            "baseline AP": ref["average_precision"],
            "this AP": here["average_precision"],
            "AP gain": here["average_precision"] - ref["average_precision"],
            "baseline F1": ref["fail_f1"],
            "this F1": here["fail_f1"],
            "F1 gain": here["fail_f1"] - ref["fail_f1"],
        })
    if rows:
        parts += ["## Against the published baseline (cross-validation)\n",
                  table(pd.DataFrame(rows),
                        ["model", "baseline AP", "this AP", "AP gain",
                         "baseline F1", "this F1", "F1 gain"],
                        ["model", "baseline AP", "this AP", "gain",
                         "baseline F1", "this F1", "gain"],
                        ["{}"] + ["{:+.4f}" if "gain" in c else "{:.4f}"
                                  for c in ["baseline AP", "this AP", "AP gain",
                                            "baseline F1", "this F1", "F1 gain"]]),
                  ""]

    holdout_path = final_dir / "holdout_summary.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path)
        parts += ["## Held-out test wafers (40 wafers never used for fitting)\n",
                  table(holdout, ["model", "stage", "average_precision", "roc_auc",
                                  "fail_f1", "fail_precision", "fail_recall", "accuracy"],
                        ["model", "stage", "AP", "ROC-AUC", "fail F1", "precision",
                         "recall", "accuracy"],
                        ["{}", "{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}",
                         "{:.4f}"]),
                  "",
                  "Baseline holdout for reference: Model A AP %.4f / F1 %.4f, "
                  "Model B AP %.4f / F1 %.4f.\n"
                  % (BASELINE_HOLDOUT["model_a_linear"]["average_precision"],
                     BASELINE_HOLDOUT["model_a_linear"]["fail_f1"],
                     BASELINE_HOLDOUT["model_b_linear"]["average_precision"],
                     BASELINE_HOLDOUT["model_b_linear"]["fail_f1"])]

    notes_path = figures_dir / "figure_notes.json"
    if notes_path.exists():
        notes = json.loads(notes_path.read_text(encoding="utf-8"))
        parts += ["## Diagnostics\n",
                  "- Correlation between the fitted parametric coefficients and the "
                  "generator's own `fail_shift / base_std`: **%.3f**"
                  % notes.get("coefficient_recovery_r", float("nan")),
                  "- Correlation between each held-out wafer's recovered failure rate "
                  "and its actual rate, using no labels: **%.3f**"
                  % notes.get("wafer_rate_recovery_r", float("nan")),
                  ""]

    figures = sorted(figures_dir.glob("*.png"))
    if figures:
        parts += ["## Figures\n"]
        parts += ["- `%s`" % p.name for p in figures]
        parts += [""]

    output.write_text("\n".join(parts), encoding="utf-8")
    print("wrote %s" % output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/mrf"))
    parser.add_argument("--final", type=Path, default=Path("results/mrf_final"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--output", type=Path, default=Path("RESULTS.md"))
    args = parser.parse_args()
    build(args.results, args.final, args.figures, args.output)


if __name__ == "__main__":
    main()
