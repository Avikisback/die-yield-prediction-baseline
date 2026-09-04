"""Regenerate RESULTS.md from the saved result files, so it cannot drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame, columns, headers, formats) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---:" if f else "---" for f in formats) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column, fmt in zip(columns, formats):
            value = row[column]
            cells.append(format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build(results: Path) -> str:
    tuned = pd.read_csv(results / "tuned" / "experiment_summary.csv")
    holdout = pd.read_csv(results / "tuned_final" / "holdout_summary.csv")
    ceiling = pd.read_csv(results / "tuned_ceiling" / "ceiling.csv")
    diagnostics = json.loads(
        (results / "tuned_ceiling" / "ceiling.json").read_text(encoding="utf-8")
    )["diagnostics"]
    mrf_path = results / "mrf_rerun" / "experiment_summary.csv"
    mrf = pd.read_csv(mrf_path) if mrf_path.exists() else None
    selection = json.loads(
        (results / "tuned_selection" / "selection.json").read_text(encoding="utf-8")
    ) if (results / "tuned_selection" / "selection.json").exists() else None

    numeric = ["average_precision", "roc_auc", "fail_f1", "fail_precision",
               "fail_recall"]
    tuned_numeric = numeric + (["calibration"] if "calibration" in tuned else [])
    parts = ["# Results", ""]
    parts.append(
        "Every cross-validation number below uses five wafer-grouped folds repeated "
        "three times, the fold builder and metric code in `modeling/validation.py`, "
        "and the same seed-42 dataset the earlier baseline and `mrf` were scored on. "
        "Only dies with `old_label == 0` are scored."
    )
    parts += ["", "## Cross-validation, all experiments", ""]
    parts.append(_table(
        tuned.sort_values("average_precision", ascending=False),
        ["experiment", *tuned_numeric, "note"],
        ["experiment", "AP", "ROC-AUC", "fail F1", "precision", "recall"]
        + (["predicted/actual"] if "calibration" in tuned else []) + ["note"],
        ["", ".4f", ".4f", ".4f", ".4f", ".4f"]
        + ([".3f"] if "calibration" in tuned else []) + [""]))

    if mrf is not None:
        parts += ["", "## The same folds, the same data, the previous pipeline", ""]
        parts.append(
            "`mrf` re-run here rather than quoted, so the comparison cannot be "
            "an artefact of a different environment."
        )
        parts += [""]
        parts.append(_table(
            mrf.sort_values("average_precision", ascending=False),
            ["experiment", *numeric],
            ["experiment", "AP", "ROC-AUC", "fail F1", "precision", "recall"],
            ["", ".4f", ".4f", ".4f", ".4f", ".4f"]))

    parts += ["", "## Held-out test wafers (40 wafers, scored once)", ""]
    parts.append(_table(
        holdout, ["model", *numeric, "accuracy"],
        ["model", "AP", "ROC-AUC", "fail F1", "precision", "recall", "accuracy"],
        ["", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f"]))
    parts += ["", "Published baseline holdout for reference: "
              "Model A AP 0.5320 / fail F1 0.5297, "
              "Model B AP 0.5934 / fail F1 0.5600."]

    parts += ["", "## Ceiling", "",
              "What the model reaches against what the generator's own knowledge "
              "could reach, on the training wafers.  See `tuned/ceiling.py` for "
              "what each rung is allowed to see.", ""]
    parts.append(_table(
        ceiling, ["level", "average_precision", "roc_auc", "fail_f1", "note"],
        ["level", "AP", "ROC-AUC", "fail F1", "what it is allowed to know"],
        ["", ".4f", ".4f", ".4f", ""]))

    block = {}
    for name in ("block_simulation", "block_cnn"):
        path = results / "tuned_ceiling" / f"{name}.json"
        if path.exists():
            block[name] = json.loads(path.read_text(encoding="utf-8"))
    if block:
        parts += ["", "## The sub-die channel on its own", "",
                  "Both numbers below come from re-running the generator's own "
                  "block process (`tuned/blocksim.py`), which is the only way to "
                  "ask what a detector *could* have reached.", ""]
        simulation = block.get("block_simulation", {})
        if simulation:
            parts += [
                f"- matched filter evaluated at the **true** cluster seed: "
                f"**{simulation['matched_filter_at_the_true_seed']:.4f}** ROC-AUC",
                f"- the same filter maximised over all 2,000 seeds: "
                f"{simulation['matched_filter_maximised_over_seeds']:.4f}",
                f"- this pipeline's likelihood-ratio bank: "
                f"**{simulation['derived_likelihood_ratio_bank']:.4f}**",
            ]
            if "previous_scan_bank" in simulation:
                parts.append(f"- `mrf`'s scan bank on the same dies: "
                             f"{simulation['previous_scan_bank']:.4f}")
            parts.append("")
            parts.append(
                "The seed is drawn uniformly per die and correlates with nothing, "
                "so the gap between the last row and the first is not a modelling "
                "failure -- it is the price of searching 2,000 positions.")
        cnn = block.get("block_cnn", {})
        if cnn:
            parts += ["", "A 1-D convolutional network was trained on the raw "
                      "readings to check whether the derived statistic misses "
                      "anything (`tuned/blockcnn.py`, GPU):", "",
                      "| detector | ROC-AUC | AP | seconds |",
                      "|---|--:|--:|--:|",
                      f"| derived likelihood ratio | {cnn['derived_auc']:.4f} | "
                      f"{cnn['derived_ap']:.4f} | {cnn['derived_seconds']:.0f} |",
                      f"| 1-D CNN | {cnn['cnn_auc']:.4f} | {cnn['cnn_ap']:.4f} | "
                      f"{cnn['cnn_seconds']:.0f} |",
                      f"| both together | {cnn['combined_auc']:.4f} | | |", "",
                      "The network does not beat the closed form, and the two "
                      "together are worse than the closed form alone, so the "
                      "network is not carrying anything the derivation missed."]

    parts += ["", "## Diagnostics", ""]
    for key, value in diagnostics.items():
        parts.append(f"- `{key}`: **{value:.4f}**")

    if selection:
        parts += ["", "## Selected constants", "",
                  "Chosen on five wafer-grouped folds of the training wafers only "
                  "(`tuned/select.py`), before the held-out wafers were scored.", ""]
        for key, value in selection["chosen"].items():
            parts.append(f"- `{key}` = {value} (grid {selection['grid'][key]})")

    parts += ["", "## Figures", ""]
    for name in ("channel_shapes.png", "wafer_rate_recovery.png",
                 "precision_recall.png", "ceiling.png", "block_channel.png",
                 "hazard_shape.png"):
        parts.append(f"- `results/tuned_figures/{name}`")
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, default=Path("results"), nargs="?")
    parser.add_argument("--output", type=Path, default=Path("RESULTS_TUNED.md"))
    args = parser.parse_args()
    args.output.write_text(build(args.results), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
