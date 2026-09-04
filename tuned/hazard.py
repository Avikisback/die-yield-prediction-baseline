"""Pre-test hazard shape, reconstructed on the grid the generator used.

``generate_die_features`` turns a wafer map into a per-die failure probability

    p = rate_w * (1 + 3 * density5 + 1.5 * radius),   clipped to [0, 0.4]

where ``rate_w`` is a per-wafer draw and the bracket is a pure function of
pre-test information.  Reproducing the bracket exactly needs the *grid* the
generator worked on, not just the dies that survived onto the wafer:

* ``density5`` is ``uniform_filter(old_fail, 5) / uniform_filter(valid, 5)``
  with zero padding, which the observed dies already determine; but
* ``radius`` is normalised by the largest distance anywhere in the wafer-map
  array -- a corner, where there is no die at all.  Normalising by the largest
  distance among *observed* dies instead inflates every radius by roughly
  sqrt(2), which turns a coefficient of 1.5 into an effective 1.06.

``mrf.spatial`` takes the second route.  This module takes the first, and
``radius_scale`` records the ratio so the difference is measurable rather than
assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter

DENSITY_WINDOWS = (3, 5, 7, 11)
GENERATOR_WINDOW = 5
DENSITY_COEFFICIENT = 3.0
RADIUS_COEFFICIENT = 1.5
PROBABILITY_CAP = 0.4
REQUIRED = ("wafer_id", "die_row", "die_col", "old_label")


def grid_shape(row: np.ndarray, col: np.ndarray) -> tuple[int, int]:
    """Best reconstruction of the wafer-map array shape from the die list.

    WM-811K maps are stored tightly, so the extreme dies sit on the array
    border and ``max + 1`` recovers the shape.  ``tests`` checks this against
    the maps the generator actually used.
    """
    return int(row.max()) + 1, int(col.max()) + 1


def radial_map(n_rows: int, n_cols: int) -> tuple[np.ndarray, float]:
    """The generator's ``compute_radial_map`` and its normalising distance."""
    centre_row, centre_col = n_rows / 2.0, n_cols / 2.0
    y, x = np.mgrid[0:n_rows, 0:n_cols]
    radial = np.sqrt((y - centre_row) ** 2 + (x - centre_col) ** 2)
    peak = float(radial.max())
    if peak > 0:
        radial = radial / peak
    return radial, peak


def engineer(wafer: pd.DataFrame, windows=DENSITY_WINDOWS) -> pd.DataFrame:
    """Add ``haz_*`` columns for one wafer; reads no post-test information."""
    missing = sorted(set(REQUIRED).difference(wafer.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if wafer["wafer_id"].nunique(dropna=False) != 1:
        raise ValueError("engineer expects exactly one wafer")
    if wafer.duplicated(["die_row", "die_col"]).any():
        raise ValueError("Duplicate die coordinates within a wafer")

    row = wafer["die_row"].to_numpy(dtype=np.int64)
    col = wafer["die_col"].to_numpy(dtype=np.int64)
    n_rows, n_cols = grid_shape(row, col)

    valid = np.zeros((n_rows, n_cols), dtype=np.float64)
    old_fail = np.zeros_like(valid)
    valid[row, col] = 1.0
    old_fail[row, col] = wafer["old_label"].to_numpy(dtype=np.int8) == 1

    radial, peak = radial_map(n_rows, n_cols)
    radius = radial[row, col]
    observed_peak = float(np.hypot(row - n_rows / 2.0, col - n_cols / 2.0).max())

    result = pd.DataFrame(index=wafer.index)
    result["haz_radius"] = radius.astype(np.float32)
    # The ratio mrf.spatial implicitly divides by.  Carried as a column so the
    # difference between the two conventions is auditable per wafer.
    result["haz_radius_scale"] = np.float32(observed_peak / max(peak, 1e-9))

    densities: dict[int, np.ndarray] = {}
    for window in windows:
        fail_average = uniform_filter(old_fail, size=window, mode="constant", cval=0.0)
        valid_average = uniform_filter(valid, size=window, mode="constant", cval=0.0)
        density = np.divide(
            fail_average, valid_average,
            out=np.zeros_like(fail_average), where=valid_average > 0,
        )
        densities[window] = density[row, col]
        result[f"haz_density_w{window}"] = densities[window].astype(np.float32)

    shape = (
        1.0
        + DENSITY_COEFFICIENT * densities[GENERATOR_WINDOW]
        + RADIUS_COEFFICIENT * radius
    )
    result["haz_shape"] = shape.astype(np.float32)
    result["haz_log_shape"] = np.log(shape).astype(np.float32)

    if old_fail.sum() > 0:
        nearest = distance_transform_edt(~old_fail.astype(bool))[row, col]
        nearest = nearest / max(float(np.hypot(n_rows, n_cols)), 1.0)
    else:
        nearest = np.ones(len(wafer), dtype=np.float64)
    result["haz_nearest_old_fail"] = nearest.astype(np.float32)

    padded = np.pad(valid.astype(bool), 1, constant_values=False)
    edge = distance_transform_edt(padded)[1:-1, 1:-1][row, col]
    edge_peak = float(edge.max())
    result["haz_edge_distance"] = (edge / max(edge_peak, 1e-9)).astype(np.float32)

    result["haz_wafer_old_fail_rate"] = np.float32(old_fail.sum() / max(valid.sum(), 1.0))
    result["haz_wafer_die_count"] = np.float32(valid.sum())
    return result


def die_prior(shape: np.ndarray, rate: float | np.ndarray) -> np.ndarray:
    """The generator's per-die failure probability for a given wafer rate."""
    return np.clip(np.asarray(rate) * np.asarray(shape), 1e-9, PROBABILITY_CAP)
