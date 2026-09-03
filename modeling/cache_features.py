"""Build per-wafer Parquet feature caches from the large generated CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modeling.features import engineer_wafer_features


def iter_wafer_groups(csv_path: Path, chunksize: int):
    """Yield complete wafers from a CSV whose rows are wafer-contiguous."""
    pending = pd.DataFrame()
    closed: set[str] = set()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        combined = pd.concat((pending, chunk), ignore_index=True)
        last_wafer = str(combined["wafer_id"].iloc[-1])
        complete = combined.loc[combined["wafer_id"].astype(str) != last_wafer]
        pending = combined.loc[combined["wafer_id"].astype(str) == last_wafer].copy()
        for wafer_id, wafer in complete.groupby("wafer_id", sort=False):
            key = str(wafer_id)
            if key in closed:
                raise ValueError(f"Wafer {key} is not contiguous in {csv_path}")
            closed.add(key)
            yield key, wafer.reset_index(drop=True)
    for wafer_id, wafer in pending.groupby("wafer_id", sort=False):
        key = str(wafer_id)
        if key in closed:
            raise ValueError(f"Wafer {key} is not contiguous in {csv_path}")
        yield key, wafer.reset_index(drop=True)


def build_cache(
    csv_path: Path,
    output_dir: Path,
    chunksize: int,
    include_block: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    for number, (wafer_id, wafer) in enumerate(
        iter_wafer_groups(csv_path, chunksize), start=1
    ):
        engineered = engineer_wafer_features(wafer, include_block=include_block)
        float_columns = [
            column
            for column in engineered.columns
            if column.startswith(("feature_", "spatial_", "block_"))
        ]
        engineered[float_columns] = engineered[float_columns].astype("float32")
        filename = f"part-{number:04d}-{wafer_id}.parquet"
        engineered.to_parquet(output_dir / filename, index=False)
        manifests.append(
            {"wafer_id": wafer_id, "rows": len(engineered), "file": filename}
        )
        if number % 10 == 0:
            print(f"Cached {number} wafers from {csv_path.name}")

    manifest = {
        "source": str(csv_path.resolve()),
        "include_block": include_block,
        "wafers": manifests,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(manifests)} wafer files to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunksize", type=int, default=10_000)
    parser.add_argument(
        "--without-block", action="store_true", help="Skip block feature extraction"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_cache(
        arguments.csv_path,
        arguments.output_dir,
        chunksize=arguments.chunksize,
        include_block=not arguments.without_block,
    )

