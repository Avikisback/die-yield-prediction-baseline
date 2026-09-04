"""Every figure the report needs, drawn from saved predictions and models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import average_precision_score, precision_recall_curve

from modeling.validation import eligible_rows
from mrf import block as blockmod
from mrf import interpret
from mrf.cache import load

PASS_COLOUR = "#2e7d32"
FAIL_COLOUR = "#c62828"
NEW_COLOUR = "#ef6c00"
ACCENT = "#1565c0"
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def _grid(frame, wafer_id, column):
    return interpret.wafer_layer(frame, wafer_id, column)


def wafer_maps(cache_test, predictions, output, wafer_ids):
    """Pre-test state, model risk, truth, and where the risk came from."""
    columns = ["wafer_id", "die_row", "die_col", "old_label", "label"]
    raw = load(cache_test, columns=columns)
    merged = raw.merge(predictions, on=["wafer_id", "die_row", "die_col"],
                       how="left", suffixes=("", "_p"))

    fig, axes = plt.subplots(len(wafer_ids), 5,
                             figsize=(15, 3.0 * len(wafer_ids)))
    axes = np.atleast_2d(axes)
    state_map = ListedColormap([PASS_COLOUR, FAIL_COLOUR, NEW_COLOUR])
    for r, wafer_id in enumerate(wafer_ids):
        here = merged.loc[merged["wafer_id"].astype(str) == str(wafer_id)].copy()
        here["state"] = np.where(here["old_label"] == 1, 1,
                                 np.where(here["label"] == 1, 2, 0))
        panels = [
            ("state", "pre-test / post-test state", state_map, None),
            ("probability_calibrated", "Model B risk", "magma", None),
            ("contrib_parametric", "die-level contribution", "coolwarm", "sym"),
            ("contrib_spatial", "spatial contribution", "coolwarm", "sym"),
            ("contrib_block", "block contribution", "coolwarm", "sym"),
        ]
        for c, (column, title, cmap, mode) in enumerate(panels):
            ax = axes[r, c]
            grid = _grid(here, wafer_id, column)
            kwargs = {}
            if mode == "sym":
                limit = np.nanmax(np.abs(grid)) or 1.0
                kwargs = {"vmin": -limit, "vmax": limit}
            image = ax.imshow(grid, cmap=cmap, interpolation="nearest", **kwargs)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                fails = int((here["label"] == 1).sum() - (here["old_label"] == 1).sum())
                ax.set_ylabel("%s\n%d new fails" % (wafer_id, fails), fontsize=8)
            else:
                plt.colorbar(image, ax=ax, fraction=0.046, shrink=0.8)
    fig.suptitle("Where each prediction comes from: green pass, red pre-test fail, "
                 "orange newly failed", fontsize=10)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def precision_recall(frames, output, base_rate):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, (y, p) in frames.items():
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, label="%s (AP=%.4f)"
                     % (label, average_precision_score(y, p)))
        budget = np.linspace(0.001, 0.25, 200)
        order = np.argsort(-p)
        hits = np.cumsum(y[order])
        idx = np.clip((budget * len(y)).astype(int), 1, len(y) - 1)
        axes[1].plot(100 * budget, hits[idx] / max(y.sum(), 1), label=label)
    axes[0].axhline(base_rate, ls="--", c="grey", lw=1,
                    label="always predict fail (%.1f%%)" % (100 * base_rate))
    axes[0].set_xlabel("recall"), axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-recall on the minority class")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel("% of dies flagged for review")
    axes[1].set_ylabel("share of true new failures caught")
    axes[1].set_title("Yield of a fixed inspection budget")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.25)
    fig.savefig(output)
    plt.close(fig)


def delta_analysis(y, p_a, p_b, output):
    """What Model B adds, and to which dies."""
    rank_a = pd.Series(p_a).rank(pct=True).to_numpy()
    rank_b = pd.Series(p_b).rank(pct=True).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    fails = y == 1
    axes[0].scatter(100 * rank_a[fails], 100 * rank_b[fails], s=7, alpha=0.35,
                    c=ACCENT, edgecolors="none")
    axes[0].plot([0, 100], [0, 100], ls="--", c="grey", lw=1)
    rescued = fails & (rank_a < 0.90) & (rank_b >= 0.90)
    axes[0].scatter(100 * rank_a[rescued], 100 * rank_b[rescued], s=14,
                    c=NEW_COLOUR, edgecolors="none",
                    label="%d rescued by block data" % rescued.sum())
    axes[0].set_xlabel("Model A percentile"), axes[0].set_ylabel("Model B percentile")
    axes[0].set_title("True new failures, ranked by each model")
    axes[0].legend(fontsize=8)

    # Model A's own top 5% sits entirely in its top band by construction, so
    # comparing the two models band by band would be tautological.  What is
    # actually informative is where Model A's misses live and how many of them
    # the block readings pull back.
    edges = np.linspace(0, 1, 11)
    bucket = np.clip(np.digitize(rank_a, edges) - 1, 0, 9)
    total, rescued_n = [], []
    for b in range(10):
        here = (bucket == b) & fails
        total.append(int(here.sum()))
        rescued_n.append(int((here & (rank_b >= 0.95)).sum()))
    x = np.arange(10)
    axes[1].bar(x, total, 0.62, label="true new failures", color="#b0bec5")
    axes[1].bar(x, rescued_n, 0.62, label="reached Model B's top 5%", color=ACCENT)
    for i, (t, r) in enumerate(zip(total, rescued_n)):
        if t:
            axes[1].text(i, t + max(total) * 0.015, "%d/%d" % (r, t),
                         ha="center", fontsize=6.5, color="#546e7a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["%d-%d" % (10 * i, 10 * i + 10) for i in range(10)],
                            rotation=45, fontsize=7)
    axes[1].set_xlabel("Model A percentile band")
    axes[1].set_ylabel("true new failures")
    axes[1].set_title("Where Model A's misses are, and which ones block data recovers")
    axes[1].legend(fontsize=8)
    fig.savefig(output)
    plt.close(fig)


def importance(model_path, output, config_path):
    saved = joblib.load(model_path)
    model, columns = saved["model"], saved["columns"]
    table = interpret.global_importance(model, columns)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    grouped = table.groupby("resolution")["abs_coefficient"].sum().sort_values()
    axes[0].barh(grouped.index, grouped.to_numpy(), color=ACCENT)
    axes[0].set_title("Total |coefficient| by resolution")
    axes[0].set_xlabel("sum of |coefficient|")

    top = table.head(18).iloc[::-1]
    colours = {"parametric": "#90a4ae", "spatial": NEW_COLOUR, "block": ACCENT}
    axes[1].barh(top["feature"], top["coefficient"],
                 color=[colours.get(r, "grey") for r in top["resolution"]])
    axes[1].set_title("Strongest individual drivers")
    axes[1].tick_params(axis="y", labelsize=6.5)
    axes[1].axvline(0, c="black", lw=0.8)

    truth = interpret.true_shift_directions(config_path)
    learned = table.loc[table["feature"].str.fullmatch(r"feature_\d+")].copy()
    joined = learned.merge(truth, on="feature")
    axes[2].scatter(joined["true_shift_in_sd"], joined["coefficient"], s=6,
                    alpha=0.4, c=ACCENT, edgecolors="none")
    r = float(np.corrcoef(joined["true_shift_in_sd"], joined["coefficient"])[0, 1])
    axes[2].set_xlabel("generator's true fail_shift / base_std")
    axes[2].set_ylabel("fitted coefficient")
    axes[2].set_title("Recovered physics (r = %.3f)" % r)
    axes[2].axhline(0, c="grey", lw=0.6), axes[2].axvline(0, c="grey", lw=0.6)
    fig.savefig(output)
    plt.close(fig)
    return table, r


def block_pattern(csv_path, keys, labels, output):
    found = interpret.find_block_readings(csv_path, keys)
    bank = blockmod.ScanBank()
    fig, axes = plt.subplots(len(keys), 2, figsize=(12, 2.6 * len(keys)))
    axes = np.atleast_2d(axes)
    for r, key in enumerate(keys):
        readings = found.get(key)
        if readings is None:
            continue
        centre = np.median(readings)
        spread = 1.4826 * np.median(np.abs(readings - centre)) or 1.0
        z = ((readings - centre) / spread)[None, :]
        scanned = bank.scan(z)
        trace = np.fft.irfft(
            np.fft.rfft(z, axis=1) * bank._filters["gauss150"][None, :],
            n=bank.k, axis=1)[0] / bank._gains["gauss150"]
        peak = int(np.argmax(trace))

        axes[r, 0].plot(readings, lw=0.35, c="#546e7a")
        axes[r, 0].axvspan(max(peak - 150, 0), min(peak + 150, bank.k),
                           color=NEW_COLOUR, alpha=0.18)
        axes[r, 0].set_ylabel(labels[r], fontsize=8)
        if r == 0:
            axes[r, 0].set_title("raw block readings, detected window shaded", fontsize=9)
        axes[r, 1].plot(trace, lw=1.0, c=ACCENT)
        axes[r, 1].axvline(peak, c=NEW_COLOUR, lw=1.2)
        axes[r, 1].axhline(0, c="grey", lw=0.6)
        axes[r, 1].set_ylabel("scan sigma", fontsize=8)
        if r == 0:
            axes[r, 1].set_title("matched Gaussian scan (sigma=150 blocks)", fontsize=9)
        axes[r, 1].text(0.99, 0.9, "peak %.2f sigma at block %d"
                        % (float(scanned["block_scan_gauss150"][0]), peak),
                        transform=axes[r, 1].transAxes, ha="right", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("block index")
    fig.savefig(output)
    plt.close(fig)


def wafer_rate_recovery(predictions, rates, output):
    actual, estimated, sizes = [], [], []
    for wafer_id, rate in rates.items():
        here = predictions.loc[predictions["wafer_id"].astype(str) == str(wafer_id)]
        if here.empty:
            continue
        actual.append(float(here["label"].mean()))
        estimated.append(float(rate))
        sizes.append(len(here))
    actual, estimated = np.array(actual), np.array(estimated)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(100 * actual, 100 * estimated, s=np.array(sizes) / 12,
               alpha=0.7, c=ACCENT, edgecolors="none")
    top = max(actual.max(), estimated.max()) * 105
    ax.plot([0, top], [0, top], ls="--", c="grey", lw=1)
    r = float(np.corrcoef(actual, estimated)[0, 1])
    ax.set_xlabel("actual new-failure rate (%)")
    ax.set_ylabel("rate recovered from unlabelled scores (%)")
    ax.set_title("Per-wafer rate, recovered without labels (r = %.3f)" % r)
    fig.savefig(output)
    plt.close(fig)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("cache_test", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--test-csv", default="input/test.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notes = {}

    a = pd.read_parquet(args.final_dir / "model_a_test_predictions.parquet")
    b = pd.read_parquet(args.final_dir / "model_b_test_predictions.parquet")
    y = b["label"].to_numpy(dtype=np.int8)

    precision_recall(
        {
            "Model A (die + spatial)": (y, a["probability"].to_numpy()),
            "Model B (+ block)": (y, b["probability"].to_numpy()),
            "Model B + wafer rate": (y, b["probability_calibrated"].to_numpy()),
        },
        args.output_dir / "precision_recall.png",
        base_rate=float(y.mean()),
    )
    delta_analysis(y, a["probability"].to_numpy(), b["probability"].to_numpy(),
                   args.output_dir / "model_a_to_b_delta.png")

    table, r_coef = importance(args.final_dir / "model_b.joblib",
                               args.output_dir / "global_importance.png", args.config)
    table.to_csv(args.output_dir / "global_importance.csv", index=False)
    notes["coefficient_recovery_r"] = r_coef

    thresholds = json.loads((args.final_dir / "thresholds.json").read_text())
    notes["wafer_rate_recovery_r"] = wafer_rate_recovery(
        b, thresholds["model_b"]["test_wafer_rates"],
        args.output_dir / "wafer_rate_recovery.png")

    # Per-die attribution laid back onto the wafer, for the busiest wafers.
    saved = joblib.load(args.final_dir / "model_b.joblib")
    columns = saved["columns"]
    test = eligible_rows(load(args.cache_test,
                              columns=["wafer_id", "die_row", "die_col", "old_label",
                                       "label"] + columns))
    parts = interpret.group_contributions(
        saved["model"], test.loc[:, columns].replace([np.inf, -np.inf], np.nan))
    detail = test[["wafer_id", "die_row", "die_col"]].join(parts)
    detail = detail.merge(b, on=["wafer_id", "die_row", "die_col"], how="left")
    detail.to_parquet(args.output_dir / "test_attributions.parquet", index=False)

    busiest = (detail.assign(new=detail["label"])
               .groupby("wafer_id")["new"].sum().sort_values(ascending=False))
    wafer_maps(args.cache_test, detail, args.output_dir / "wafer_maps.png",
               list(busiest.index[:3]))

    # One convincing true positive, one near miss, one clean pass.
    hits = detail.loc[(detail["label"] == 1)].nlargest(1, "probability_calibrated")
    misses = detail.loc[(detail["label"] == 1)].nsmallest(1, "probability_calibrated")
    clean = detail.loc[(detail["label"] == 0)].nsmallest(1, "probability_calibrated")
    picks = pd.concat([hits, misses, clean])
    keys = list(zip(picks["wafer_id"], picks["die_row"], picks["die_col"]))
    block_pattern(args.test_csv, keys,
                  ["caught failure", "missed failure", "healthy die"],
                  args.output_dir / "block_pattern.png")

    # `detail` was rebuilt with a clean index in the same row order as `test`,
    # so its label is already the positional row that top_drivers wants.
    top = interpret.top_drivers(saved["model"],
                               test.loc[:, columns].replace([np.inf, -np.inf], np.nan),
                               [int(hits.index[0])], k=10)
    top.to_csv(args.output_dir / "example_drivers.csv", index=False)
    (args.output_dir / "figure_notes.json").write_text(json.dumps(notes, indent=2),
                                                       encoding="utf-8")
    print(json.dumps(notes, indent=2))
    print("figures written to %s" % args.output_dir)


if __name__ == "__main__":
    main()
