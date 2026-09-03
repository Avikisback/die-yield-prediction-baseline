"""Pre-test spatial context for one wafer.

Every column here is a function of die coordinates and ``old_label`` only.
The post-test ``label`` is never read, so the same code runs unchanged on the
unlabelled validation split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter

# The generator scores a 5x5 neighbourhood; the neighbouring widths let a model
# see whether a cluster is tight or diffuse.
DENSITY_WINDOWS = (3, 5, 7, 11)

REQUIRED = ("wafer_id", "die_row", "die_col", "old_label")


def _grids(wafer: pd.DataFrame):
    row = wafer["die_row"].to_numpy(dtype=np.int64)
    col = wafer["die_col"].to_numpy(dtype=np.int64)
    n_rows = int(row.max()) + 1
    n_cols = int(col.max()) + 1
    valid = np.zeros((n_rows, n_cols), dtype=np.float32)
    old_fail = np.zeros_like(valid)
    valid[row, col] = 1.0
    old_fail[row, col] = (wafer["old_label"].to_numpy(dtype=np.int8) == 1)
    return row, col, n_rows, n_cols, valid, old_fail


def coordinate_basis(wafer: pd.DataFrame) -> np.ndarray:
    """Return the design matrix spanning the wafer's process gradients.

    The generator adds ``radial_map * a`` and ``linear_map * b`` to every
    feature, where ``linear_map`` is ``cos(t)*xn + sin(t)*yn``.  A basis of
    ``[1, xn, yn, radius]`` therefore contains both gradients exactly; the three
    quadratic terms absorb the error in locating the wafer centre from the
    observed dies alone.
    """
    row, col, n_rows, n_cols, _, _ = _grids(wafer)
    yn = (row - n_rows / 2.0) / (n_rows / 2.0)
    xn = (col - n_cols / 2.0) / (n_cols / 2.0)
    radius = np.sqrt(
        (row - n_rows / 2.0) ** 2 + (col - n_cols / 2.0) ** 2
    )
    peak = radius.max()
    if peak > 0:
        radius = radius / peak
    ones = np.ones_like(xn)
    return np.column_stack(
        (ones, xn, yn, radius, xn * xn, yn * yn, xn * yn)
    ).astype(np.float64)


def engineer(
    wafer: pd.DataFrame, windows: tuple[int, ...] = DENSITY_WINDOWS
) -> pd.DataFrame:
    """Add ``spatial_*`` columns for one wafer, preserving row order."""
    missing = sorted(set(REQUIRED).difference(wafer.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if wafer["wafer_id"].nunique(dropna=False) != 1:
        raise ValueError("engineer expects exactly one wafer")
    if wafer.duplicated(["die_row", "die_col"]).any():
        raise ValueError("Duplicate die coordinates within a wafer")

    result = wafer.copy()
    row, col, n_rows, n_cols, valid, old_fail = _grids(result)

    basis = coordinate_basis(result)
    result["spatial_col_norm"] = basis[:, 1].astype(np.float32)
    result["spatial_row_norm"] = basis[:, 2].astype(np.float32)
    radius = basis[:, 3]
    result["spatial_radius"] = radius.astype(np.float32)

    # Padding stops the transform from treating the area just outside the
    # reconstructed footprint as a distant border.
    padded = np.pad(valid.astype(bool), 1, constant_values=False)
    edge = distance_transform_edt(padded)[1:-1, 1:-1]
    edge_peak = float(edge[row, col].max())
    if edge_peak > 0:
        edge = edge / edge_peak
    result["spatial_edge_distance"] = edge[row, col].astype(np.float32)

    densities: dict[int, np.ndarray] = {}
    for window in windows:
        fail_average = uniform_filter(old_fail, size=window, mode="constant", cval=0.0)
        valid_average = uniform_filter(valid, size=window, mode="constant", cval=0.0)
        density = np.divide(
            fail_average,
            valid_average,
            out=np.zeros_like(fail_average),
            where=valid_average > 0,
        )
        densities[window] = density[row, col].astype(np.float64)
        result[f"spatial_old_fail_density_w{window}"] = densities[window].astype(
            np.float32
        )

    if old_fail.sum() > 0:
        nearest = distance_transform_edt(~old_fail.astype(bool))[row, col]
        nearest = nearest / max(float(np.hypot(n_rows, n_cols)), 1.0)
    else:
        nearest = np.ones(len(result), dtype=np.float64)
    result["spatial_nearest_old_fail"] = nearest.astype(np.float32)

    # The generator draws a per-wafer rate and then modulates it:
    #   p = rate * (1 + 3 * density_w5 + 1.5 * radius)
    # so the log of the bracket is the entire die-to-die shape of the hazard and
    # log(rate) is a pure per-wafer offset.  Handing the model that bracket
    # directly saves it from having to rediscover the interaction.
    hazard = 1.0 + 3.0 * densities[5] + 1.5 * radius
    result["spatial_log_hazard_shape"] = np.log(hazard).astype(np.float32)

    result["spatial_wafer_old_fail_rate"] = np.float32(
        old_fail.sum() / max(valid.sum(), 1.0)
    )
    result["spatial_wafer_die_count"] = np.float32(valid.sum())
    result["spatial_wafer_rows"] = np.float32(n_rows)
    result["spatial_wafer_cols"] = np.float32(n_cols)
    return result
