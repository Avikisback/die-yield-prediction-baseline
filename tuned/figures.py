"""Figures, each one showing a claim the report makes rather than decorating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from tuned.cache import eligible_mask, load  # noqa: E402

INK = "#1b1b1b"
ACCENT = "#c2410c"
MUTED = "#94a3b8"
SECOND = "#0f766e"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def channel_shapes(model, design, y, path: Path) -> None:
    """The fitted log-likelihood ratios, against the straight line they replace.

    This is the whole argument for the smooth head in one picture: the fail
    class is a mixture, so the parametric channel's ratio is nearly flat where
    the marginal failures sit and steep in the tail where the full-shift ones
    do, and one coefficient cannot be both.
    """
    names = [n for n in ("parametric_score", "block_score") if n in design]
    figure, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.5))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        values = design[name].to_numpy()
        lo, hi = np.quantile(values, [0.0005, 0.9995])
        grid = np.linspace(lo, hi, 400)
        curve = model.head_.curve(name, grid)
        curve = curve - curve[np.searchsorted(grid, np.median(values))]

        slope, intercept = np.polyfit(values, np.interp(values, grid, curve), 1)
        twin = axis.twinx()
        twin.hist(values[y == 0], bins=70, range=(lo, hi), color=MUTED,
                  alpha=0.30, label="dies that passed")
        twin.hist(values[y == 1], bins=70, range=(lo, hi), color=SECOND,
                  alpha=0.45, label="dies that failed")
        twin.set_yticks([])
        axis.set_zorder(twin.get_zorder() + 1)
        axis.patch.set_visible(False)
        line = axis.plot(grid, curve, color=ACCENT, lw=2,
                         label="fitted (spline)")
        dashed = axis.plot(grid, slope * grid + intercept, color=MUTED, lw=1.4,
                           ls="--", label="best single coefficient")
        axis.set_xlabel(name.replace("_", " "))
        axis.set_ylabel("log-odds contribution")
        handles = line + dashed + [twin.containers[0][0], twin.containers[1][0]]
        axis.legend(handles, [h.get_label() for h in handles],
                    loc="upper left", frameon=False, fontsize=8)
    figure.suptitle(
        "Per-channel log-likelihood ratios, against the straight line they replace",
        y=1.02)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def wafer_rate_recovery(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.scatter(frame["true_rate"], frame["recovered_rate"], s=16,
                 color=ACCENT, alpha=0.75, edgecolor="none")
    limit = float(max(frame["true_rate"].max(), frame["recovered_rate"].max())) * 1.05
    axis.plot([0, limit], [0, limit], color=MUTED, lw=1, ls="--")
    r = float(np.corrcoef(frame["true_rate"], frame["recovered_rate"])[0, 1])
    axis.set_xlabel("wafer_base_rate drawn by the generator")
    axis.set_ylabel("posterior mean, recovered without labels")
    axis.set_title(f"Per-wafer rate recovery (r = {r:.3f})")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def precision_recall(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                     path: Path) -> None:
    """PR curves plus what a fixed inspection budget actually buys."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    colours = [ACCENT, SECOND, MUTED, "#7c3aed", "#b45309"]
    for (name, (y, probability)), colour in zip(curves.items(), colours):
        precision, recall, _ = precision_recall_curve(y, probability)
        left.plot(recall, precision, lw=1.8, color=colour,
                  label=f"{name} (AP {average_precision_score(y, probability):.4f})")
        order = np.argsort(-probability)
        caught = np.cumsum(y[order]) / max(y.sum(), 1)
        budget = np.arange(1, len(order) + 1) / len(order)
        right.plot(100 * budget, 100 * caught, lw=1.8, color=colour, label=name)
    left.set_xlabel("recall on newly failing dies")
    left.set_ylabel("precision")
    left.set_title("Precision-recall, out-of-fold")
    left.legend(frameon=False, fontsize=8)
    right.set_xlim(0, 20)
    right.set_xlabel("share of eligible dies re-inspected (%)")
    right.set_ylabel("share of new failures caught (%)")
    right.set_title("Inspection budget")
    right.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


# Scored against a different set of positives, so it does not belong on the
# same axis as the rest; it stays in the results table instead.
DIFFERENT_POPULATION = ("model_hard_failures_only",)


def ceiling(table: pd.DataFrame, path: Path) -> None:
    table = table.loc[~table["level"].isin(DIFFERENT_POPULATION)]
    order = table.sort_values("average_precision")
    figure, axis = plt.subplots(figsize=(6.6, 0.46 * len(order) + 1.2))
    colours = [ACCENT if name == "model" else MUTED for name in order["level"]]
    axis.barh(order["level"], order["average_precision"], color=colours, height=0.62)
    for y, value in enumerate(order["average_precision"]):
        axis.text(value + 0.004, y, f"{value:.4f}", va="center", fontsize=8)
    axis.set_xlabel("average precision")
    axis.set_xlim(0, float(order["average_precision"].max()) * 1.18)
    axis.set_title("What the model reaches, and what nothing could exceed")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def block_channel(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(4.6, 4.2))
    for (name, score), colour in zip(scores.items(), [ACCENT, MUTED, SECOND]):
        fpr, tpr, _ = roc_curve(y, score)
        axis.plot(fpr, tpr, lw=1.8, color=colour,
                  label=f"{name} (AUC {roc_auc_score(y, score):.4f})")
    axis.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("Sub-die block channel alone")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def hazard_calibration(shape: np.ndarray, mrf_shape: np.ndarray, y: np.ndarray,
                       path: Path) -> None:
    """Observed failure rate against each reconstruction of the hazard shape."""
    figure, axis = plt.subplots(figsize=(4.8, 3.8))
    for name, values, colour in (("reconstructed on the generator's grid", shape, ACCENT),
                                 ("normalised by the observed dies", mrf_shape, MUTED)):
        edges = np.quantile(values, np.linspace(0, 1, 13))
        edges = np.unique(edges)
        index = np.clip(np.searchsorted(edges, values, "right") - 1, 0, len(edges) - 2)
        centre = np.array([values[index == b].mean() for b in range(len(edges) - 1)])
        rate = np.array([y[index == b].mean() for b in range(len(edges) - 1)])
        axis.plot(centre, rate, "o-", color=colour, lw=1.6, ms=4, label=name)
    axis.set_xlabel("hazard shape  1 + 3*density + 1.5*radius")
    axis.set_ylabel("observed new-failure rate")
    axis.set_title("Does the shape predict the rate it claims to?")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mrf-oof", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load(args.cache_dir)
    parametric = sorted(c for c in frame.columns if c.startswith("feature_"))
    x = frame.loc[:, parametric].to_numpy(dtype=np.float32)
    frame = frame.drop(columns=parametric)
    eligible = eligible_mask(frame)
    y = frame.loc[eligible, "label"].to_numpy(dtype=np.int8)

    bundle = joblib.load(args.results_dir / "tuned_final" / "model_b.joblib")
    model = bundle["model"]
    design = model._design(frame, x, eligible)
    channel_shapes(model, design, y, args.output_dir / "channel_shapes.png")

    recovery = pd.read_csv(args.results_dir / "tuned_ceiling" / "wafer_rate_recovery.csv")
    wafer_rate_recovery(recovery, args.output_dir / "wafer_rate_recovery.png")
    ceiling(pd.read_csv(args.results_dir / "tuned_ceiling" / "ceiling.csv"),
            args.output_dir / "ceiling.png")

    oof = pd.read_parquet(args.results_dir / "tuned" / "oof_predictions.parquet")
    curves = {}
    for name, label in (("B_full", "this model"), ("B_linear", "linear head"),
                        ("A_full", "no block readings")):
        piece = oof.loc[oof["experiment"] == name]
        if len(piece):
            curves[label] = (piece["label"].to_numpy(),
                             piece["oof_probability"].to_numpy())
    if args.mrf_oof is not None and args.mrf_oof.exists():
        mrf = pd.read_parquet(args.mrf_oof)
        piece = mrf.loc[mrf["experiment"] == "B_stacked_cal"]
        if len(piece):
            curves["mrf B stacked + rate"] = (piece["label"].to_numpy(),
                                              piece["oof_probability"].to_numpy())
    precision_recall(curves, args.output_dir / "precision_recall.png")

    keyed = frame.loc[eligible, ["wafer_id", "die_row", "die_col"]].copy()
    keyed["derived"] = design["block_score"].to_numpy()
    scores = {"derived likelihood ratio": keyed["derived"].to_numpy()}
    block_labels = y
    if args.mrf_oof is not None and args.mrf_oof.exists():
        mrf = pd.read_parquet(args.mrf_oof)
        piece = mrf.loc[mrf["experiment"] == "block_only",
                        ["wafer_id", "die_row", "die_col", "label",
                         "oof_probability"]]
        if len(piece):
            merged = keyed.merge(piece, on=["wafer_id", "die_row", "die_col"],
                                 how="inner")
            block_labels = merged["label"].to_numpy()
            scores = {"derived likelihood ratio": merged["derived"].to_numpy(),
                      "mrf scan bank": merged["oof_probability"].to_numpy()}
    block_channel(block_labels, scores, args.output_dir / "block_channel.png")

    shape = frame.loc[eligible, "haz_shape"].to_numpy(dtype=np.float64)
    scale = frame.loc[eligible, "haz_radius_scale"].to_numpy(dtype=np.float64)
    radius = frame.loc[eligible, "haz_radius"].to_numpy(dtype=np.float64)
    hazard_calibration(shape, shape + 1.5 * radius * (1.0 / scale - 1.0), y,
                       args.output_dir / "hazard_shape.png")

    (args.output_dir / "figure_notes.json").write_text(json.dumps({
        "channel_shapes.png": "fitted per-channel log-likelihood ratios",
        "wafer_rate_recovery.png": "recovered vs drawn per-wafer base rate",
        "precision_recall.png": "out-of-fold PR curves and inspection budget",
        "ceiling.png": "the oracle ladder",
        "block_channel.png": "ROC of the sub-die channel on its own",
        "hazard_shape.png": "calibration of the two radius conventions",
    }, indent=2), encoding="utf-8")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
