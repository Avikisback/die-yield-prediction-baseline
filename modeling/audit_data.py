"""Memory-safe audit of generated train/test/validation CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit_labeled(path: Path, chunksize: int = 20_000) -> dict[str, object]:
    total = old_failures = final_failures = eligible = new_failures = 0
    wafer_counts: dict[str, list[int]] = {}
    for chunk in pd.read_csv(
        path,
        usecols=["wafer_id", "old_label", "label"],
        chunksize=chunksize,
    ):
        total += len(chunk)
        old_failures += int(chunk["old_label"].sum())
        final_failures += int(chunk["label"].sum())
        eligible_mask = chunk["old_label"].eq(0)
        eligible += int(eligible_mask.sum())
        new_failures += int(chunk.loc[eligible_mask, "label"].sum())
        grouped = chunk.assign(
            eligible=eligible_mask.astype(np.int8),
            new_fail=(eligible_mask & chunk["label"].eq(1)).astype(np.int8),
        ).groupby("wafer_id")[["eligible", "new_fail"]].sum()
        for wafer_id, row in grouped.iterrows():
            counts = wafer_counts.setdefault(str(wafer_id), [0, 0])
            counts[0] += int(row["eligible"])
            counts[1] += int(row["new_fail"])

    wafer_rates = np.array(
        [fail / count if count else 0.0 for count, fail in wafer_counts.values()]
    )
    return {
        "file": str(path),
        "rows": total,
        "wafers": len(wafer_counts),
        "old_failures": old_failures,
        "final_failures": final_failures,
        "eligible_dies": eligible,
        "new_failures": new_failures,
        "eligible_new_fail_rate": new_failures / eligible if eligible else 0.0,
        "wafers_with_zero_new_failures": int(np.sum(wafer_rates == 0)),
        "wafer_new_fail_rate_median": float(np.median(wafer_rates)),
        "wafer_new_fail_rate_q90": float(np.quantile(wafer_rates, 0.9)),
        "wafer_new_fail_rate_max": float(wafer_rates.max(initial=0.0)),
    }


def audit_unlabeled(path: Path, chunksize: int = 20_000) -> dict[str, object]:
    total = old_failures = 0
    wafers: set[str] = set()
    for chunk in pd.read_csv(
        path,
        usecols=["wafer_id", "old_label"],
        chunksize=chunksize,
    ):
        total += len(chunk)
        old_failures += int(chunk["old_label"].sum())
        wafers.update(chunk["wafer_id"].astype(str).unique())
    return {
        "file": str(path),
        "rows": total,
        "wafers": len(wafers),
        "old_failures": old_failures,
        "eligible_dies": total - old_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = {
        "train": audit_labeled(arguments.input_dir / "train.csv"),
        "test": audit_labeled(arguments.input_dir / "test.csv"),
        "validation": audit_unlabeled(arguments.input_dir / "validation.csv"),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
