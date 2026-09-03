"""Turn the very large generated CSVs into one compact Parquet file per wafer.

The CSVs carry 2,000 block readings per die as a single string, so a full split
is several gigabytes.  Everything downstream works from the cache instead:
features are extracted once, the strings are dropped, and the result is stored
as float32.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mrf import block as blockmod
from mrf import parametric, spatial

IDENTIFIERS = ("wafer_id", "die_row", "die_col", "old_label")


def iter_wafers(csv_path: Path, chunksize: int):
    """Yield complete wafers from a CSV whose rows are wafer-contiguous."""
    pending = pd.DataFrame()
    seen: set[str] = set()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        combined = pd.concat((pending, chunk), ignore_index=True)
        last = str(combined["wafer_id"].iloc[-1])
        complete = combined.loc[combined["wafer_id"].astype(str) != last]
        pending = combined.loc[combined["wafer_id"].astype(str) == last].copy()
        for wafer_id, wafer in complete.groupby("wafer_id", sort=False):
            key = str(wafer_id)
            if key in seen:
                raise ValueError(f"Wafer {key} is not contiguous in {csv_path}")
            seen.add(key)
            yield key, wafer.reset_index(drop=True)
    for wafer_id, wafer in pending.groupby("wafer_id", sort=False):
        key = str(wafer_id)
        if key in seen:
            raise ValueError(f"Wafer {key} is not contiguous in {csv_path}")
        yield key, wafer.reset_index(drop=True)


def engineer_wafer(
    wafer: pd.DataFrame,
    bank: blockmod.ScanBank | None,
    keep_raw: bool,
) -> pd.DataFrame:
    feature_columns = [c for c in wafer.columns if c.startswith("feature_")]
    if not feature_columns:
        raise ValueError("No feature_* columns present")

    contextual = spatial.engineer(wafer)
    pieces: list[pd.DataFrame] = [
        contextual.loc[:, [c for c in contextual.columns if c.startswith("spatial_")]]
    ]

    standardised, gradient = parametric.detrend_wafer(wafer, feature_columns)
    pieces.append(
        pd.DataFrame(
            standardised,
            columns=parametric.detrended_names(feature_columns),
            index=wafer.index,
        )
    )
    pieces.append(
        pd.DataFrame(
            {k: v.astype(np.float32) for k, v in parametric.summary_columns(standardised).items()},
            index=wafer.index,
        )
    )

    if keep_raw:
        pieces.append(wafer.loc[:, feature_columns].astype(np.float32))
    if "block_readings" in wafer.columns:
        pieces.append(blockmod.engineer(wafer["block_readings"], bank))

    head = wafer.loc[:, [c for c in IDENTIFIERS if c in wafer.columns]].copy()
    if "label" in wafer.columns:
        head["label"] = wafer["label"].astype(np.int8)
    head["wafer_gradient_strength"] = np.float32(np.median(gradient))
    return pd.concat([head] + pieces, axis=1)


def build(
    csv_path: Path,
    output_dir: Path,
    chunksize: int = 6_000,
    keep_raw: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = blockmod.ScanBank()
    manifest: list[dict[str, object]] = []
    started = time.perf_counter()
    for number, (wafer_id, wafer) in enumerate(iter_wafers(csv_path, chunksize), start=1):
        engineered = engineer_wafer(wafer, bank, keep_raw)
        name = f"part-{number:04d}-{wafer_id}.parquet"
        engineered.to_parquet(output_dir / name, index=False)
        manifest.append({"wafer_id": wafer_id, "rows": len(engineered), "file": name})
        if number % 10 == 0:
            rate = number / (time.perf_counter() - started)
            print(f"  {number} wafers  ({rate:.2f}/s)", flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {"source": str(csv_path.resolve()), "keep_raw": keep_raw, "wafers": manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(int(item["rows"]) for item in manifest)
    print(
        f"Wrote {len(manifest)} wafers / {total} dies to {output_dir} "
        f"in {time.perf_counter() - started:.0f}s"
    )


def load(cache_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(Path(cache_dir).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No part-*.parquet files in {cache_dir}")
    return pd.concat(
        (pd.read_parquet(path, columns=columns) for path in paths), ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunksize", type=int, default=6_000)
    parser.add_argument("--drop-raw", action="store_true")
    args = parser.parse_args()
    build(args.csv_path, args.output_dir, args.chunksize, keep_raw=not args.drop_raw)


if __name__ == "__main__":
    main()
