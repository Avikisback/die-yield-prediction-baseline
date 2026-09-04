"""Turn the generated CSVs into one compact Parquet file per wafer.

Each split's CSV carries 2,000 block readings per die as a string, so the whole
thing is several gigabytes and is unusable as a starting point for every
experiment.  This pass reads it once, extracts the hazard shape and the block
likelihood-ratio statistics, keeps the raw parametric measurements, drops the
strings, and stores float32.

The block statistics need a null to be measured against.  That null -- the
readings' level, scale and noise spectrum, and each scan's null centre and
spread -- is estimated from the *training* split and then reused verbatim for
test and validation, so a statistic means the same thing in every split.  It is
estimated from the readings alone and never touches a label, which is what lets
the same constants be applied to the unlabelled split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tuned import blocks, hazard

IDENTIFIERS = ("wafer_id", "die_row", "die_col", "old_label")
NULL_FILE = "block_null.json"


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


def _load_null(path: Path | None):
    if path is None:
        return None, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    reference = {k: tuple(v) for k, v in payload["reference"].items()}
    return blocks.NoiseModel.from_dict(payload["noise"]), reference


def build(csv_path: Path, output_dir: Path, chunksize: int = 4_000,
          null_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    noise, reference = _load_null(null_path)
    manifest: list[dict[str, object]] = []
    started = time.perf_counter()

    for number, (wafer_id, wafer) in enumerate(iter_wafers(csv_path, chunksize), 1):
        feature_columns = [c for c in wafer.columns if c.startswith("feature_")]
        if not feature_columns:
            raise ValueError("No feature_* columns present")

        head = wafer.loc[:, [c for c in IDENTIFIERS if c in wafer.columns]].copy()
        if "label" in wafer.columns:
            head["label"] = wafer["label"].astype(np.int8)
        pieces = [head, hazard.engineer(wafer),
                  wafer.loc[:, feature_columns].astype(np.float32)]

        if "block_readings" in wafer.columns:
            readings = blocks.parse(wafer["block_readings"])
            if noise is None:
                noise = blocks.NoiseModel.fit(readings)
            columns, reference = blocks.engineer(readings, noise,
                                                 reference=reference)
            pieces.append(pd.DataFrame(columns, index=wafer.index))

        engineered = pd.concat(pieces, axis=1)
        engineered.to_parquet(
            output_dir / f"part-{number:04d}-{wafer_id}.parquet", index=False
        )
        manifest.append({"wafer_id": wafer_id, "rows": int(len(engineered))})
        if number % 10 == 0:
            rate = number / (time.perf_counter() - started)
            print(f"  {number} wafers ({rate:.2f}/s)", flush=True)

    if noise is not None and null_path is None:
        (output_dir / NULL_FILE).write_text(
            json.dumps({"noise": noise.to_dict(),
                        "reference": {k: list(v) for k, v in reference.items()}}),
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        json.dumps({"source": str(csv_path.resolve()), "wafers": manifest}, indent=2),
        encoding="utf-8",
    )
    total = sum(int(item["rows"]) for item in manifest)
    print(f"Wrote {len(manifest)} wafers / {total} dies to {output_dir} "
          f"in {time.perf_counter() - started:.0f}s")


def eligible_mask(frame: pd.DataFrame) -> np.ndarray:
    """The scored population: dies that passed the pre-test."""
    return frame["old_label"].to_numpy(dtype=np.int8) == 0


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
    parser.add_argument("--chunksize", type=int, default=4_000)
    parser.add_argument("--null", type=Path, default=None,
                        help="block_null.json written by the training cache")
    args = parser.parse_args()
    build(args.csv_path, args.output_dir, args.chunksize, args.null)


if __name__ == "__main__":
    main()
