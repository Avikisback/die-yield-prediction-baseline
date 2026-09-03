"""Feature engineering for die-level, spatial, and block-level signals.

All spatial features in this module use only pre-test information.  In
particular, the post-test ``label`` column is never consulted when constructing
neighborhood features.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter


DEFAULT_SPATIAL_WINDOWS = (3, 5, 7, 11)
DEFAULT_BLOCK_WINDOWS = (16, 32, 64, 128, 256, 384, 512)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def engineer_spatial_features(
    wafer: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_SPATIAL_WINDOWS,
) -> pd.DataFrame:
    """Add pre-test spatial features for one wafer while preserving row order."""
    _require_columns(wafer, ("wafer_id", "die_row", "die_col", "old_label"))
    if wafer.empty:
        return wafer.copy()
    if wafer["wafer_id"].nunique(dropna=False) != 1:
        raise ValueError("engineer_spatial_features expects exactly one wafer")
    if wafer.duplicated(["die_row", "die_col"]).any():
        raise ValueError("Duplicate die coordinates found within a wafer")
    if any(window < 1 or window % 2 == 0 for window in windows):
        raise ValueError("Spatial windows must be positive odd integers")

    result = wafer.copy()
    row = result["die_row"].to_numpy(dtype=np.int64)
    col = result["die_col"].to_numpy(dtype=np.int64)
    row_min, row_max = int(row.min()), int(row.max())
    col_min, col_max = int(col.min()), int(col.max())

    shifted_row = row - row_min
    shifted_col = col - col_min
    n_rows = row_max - row_min + 1
    n_cols = col_max - col_min + 1

    valid = np.zeros((n_rows, n_cols), dtype=np.float32)
    old_fail = np.zeros_like(valid)
    valid[shifted_row, shifted_col] = 1.0
    old_fail[shifted_row, shifted_col] = (
        result["old_label"].to_numpy(dtype=np.int8) == 1
    )

    row_span = max(n_rows - 1, 1)
    col_span = max(n_cols - 1, 1)
    row_norm = shifted_row / row_span
    col_norm = shifted_col / col_span
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0
    radial_grid = np.sqrt(
        (np.arange(n_rows)[:, None] - center_row) ** 2
        + (np.arange(n_cols)[None, :] - center_col) ** 2
    )
    radial_valid_max = float(radial_grid[shifted_row, shifted_col].max())
    if radial_valid_max > 0:
        radial_grid /= radial_valid_max

    # Padding makes the distance transform see the area immediately outside the
    # reconstructed wafer footprint as invalid rather than as a distant border.
    padded_valid = np.pad(valid.astype(bool), 1, constant_values=False)
    edge_distance_grid = distance_transform_edt(padded_valid)[1:-1, 1:-1]
    edge_max = float(edge_distance_grid[shifted_row, shifted_col].max())
    if edge_max > 0:
        edge_distance_grid /= edge_max

    result["spatial_row_norm"] = row_norm.astype(np.float32)
    result["spatial_col_norm"] = col_norm.astype(np.float32)
    result["spatial_radius"] = radial_grid[shifted_row, shifted_col].astype(np.float32)
    result["spatial_edge_distance"] = edge_distance_grid[
        shifted_row, shifted_col
    ].astype(np.float32)

    for window in windows:
        # uniform_filter returns local averages for both arrays.  Their ratio is
        # therefore the old-fail fraction among valid dies in the window.
        fail_average = uniform_filter(
            old_fail, size=window, mode="constant", cval=0.0
        )
        valid_average = uniform_filter(
            valid, size=window, mode="constant", cval=0.0
        )
        density = np.divide(
            fail_average,
            valid_average,
            out=np.zeros_like(fail_average),
            where=valid_average > 0,
        )
        result[f"spatial_old_fail_density_w{window}"] = density[
            shifted_row, shifted_col
        ].astype(np.float32)

    if old_fail.sum() > 0:
        nearest_grid = distance_transform_edt(~old_fail.astype(bool))
        nearest = nearest_grid[shifted_row, shifted_col]
        diagonal = max(float(np.hypot(n_rows, n_cols)), 1.0)
        nearest = nearest / diagonal
    else:
        nearest = np.ones(len(result), dtype=np.float64)

    result["spatial_nearest_old_fail"] = nearest.astype(np.float32)
    result["spatial_wafer_old_fail_rate"] = np.float32(
        old_fail.sum() / max(valid.sum(), 1.0)
    )
    result["spatial_wafer_die_count"] = np.float32(valid.sum())
    result["spatial_wafer_rows"] = np.float32(n_rows)
    result["spatial_wafer_cols"] = np.float32(n_cols)
    return result


def _safe_autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _circular_scan(values: np.ndarray, window: int) -> tuple[float, float, float]:
    """Return maximum, minimum, and normalized max location for a circular scan."""
    size = len(values)
    if size == 0:
        return 0.0, 0.0, 0.0
    window = min(max(int(window), 1), size)
    extended = np.concatenate((values, values[: window - 1]))
    cumulative = np.concatenate(([0.0], np.cumsum(extended, dtype=np.float64)))
    sums = cumulative[window : window + size] - cumulative[:size]
    scaled = sums / np.sqrt(window)
    max_index = int(np.argmax(scaled))
    return float(scaled[max_index]), float(scaled.min()), max_index / size


def block_feature_names(
    windows: tuple[int, ...] = DEFAULT_BLOCK_WINDOWS,
) -> list[str]:
    names = [
        "block_length",
        "block_mean",
        "block_std",
        "block_median",
        "block_mad",
        "block_min",
        "block_max",
        "block_q01",
        "block_q05",
        "block_q10",
        "block_q90",
        "block_q95",
        "block_q99",
        "block_skew",
        "block_kurtosis",
        "block_frac_gt_2",
        "block_frac_gt_3",
        "block_frac_lt_m2",
        "block_frac_lt_m3",
        "block_top_1pct_mean_z",
        "block_top_5pct_mean_z",
        "block_bottom_1pct_mean_z",
        "block_autocorr_lag1",
        "block_autocorr_lag5",
        "block_autocorr_lag20",
    ]
    for window in windows:
        names.extend(
            (
                f"block_scan_max_w{window}",
                f"block_scan_min_w{window}",
                f"block_scan_abs_w{window}",
                f"block_scan_location_w{window}",
            )
        )
    return names


def engineer_one_block_signal(
    value: str | np.ndarray,
    windows: tuple[int, ...] = DEFAULT_BLOCK_WINDOWS,
) -> dict[str, float]:
    """Convert one space-separated block signal into compact robust features."""
    if isinstance(value, str):
        readings = np.fromstring(value, sep=" ", dtype=np.float32)
    else:
        readings = np.asarray(value, dtype=np.float32).reshape(-1)
    readings = readings[np.isfinite(readings)]
    if readings.size == 0:
        return {name: np.nan for name in block_feature_names(windows)}

    median = float(np.median(readings))
    mad = float(np.median(np.abs(readings - median)))
    robust_scale = 1.4826 * mad
    if robust_scale <= 1e-8:
        robust_scale = float(readings.std())
    if robust_scale <= 1e-8:
        robust_scale = 1.0
    z = (readings.astype(np.float64) - median) / robust_scale

    quantiles = np.quantile(readings, (0.01, 0.05, 0.10, 0.90, 0.95, 0.99))
    centered = readings.astype(np.float64) - float(readings.mean())
    std = float(readings.std())
    if std > 1e-8:
        normalized = centered / std
        skew = float(np.mean(normalized**3))
        kurtosis = float(np.mean(normalized**4) - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0

    top_1_count = max(1, int(np.ceil(0.01 * len(z))))
    top_5_count = max(1, int(np.ceil(0.05 * len(z))))
    ordered = np.sort(z)
    output: dict[str, float] = {
        "block_length": float(len(readings)),
        "block_mean": float(readings.mean()),
        "block_std": std,
        "block_median": median,
        "block_mad": mad,
        "block_min": float(readings.min()),
        "block_max": float(readings.max()),
        "block_q01": float(quantiles[0]),
        "block_q05": float(quantiles[1]),
        "block_q10": float(quantiles[2]),
        "block_q90": float(quantiles[3]),
        "block_q95": float(quantiles[4]),
        "block_q99": float(quantiles[5]),
        "block_skew": skew,
        "block_kurtosis": kurtosis,
        "block_frac_gt_2": float(np.mean(z > 2.0)),
        "block_frac_gt_3": float(np.mean(z > 3.0)),
        "block_frac_lt_m2": float(np.mean(z < -2.0)),
        "block_frac_lt_m3": float(np.mean(z < -3.0)),
        "block_top_1pct_mean_z": float(ordered[-top_1_count:].mean()),
        "block_top_5pct_mean_z": float(ordered[-top_5_count:].mean()),
        "block_bottom_1pct_mean_z": float(ordered[:top_1_count].mean()),
        "block_autocorr_lag1": _safe_autocorrelation(z, 1),
        "block_autocorr_lag5": _safe_autocorrelation(z, 5),
        "block_autocorr_lag20": _safe_autocorrelation(z, 20),
    }

    for window in windows:
        maximum, minimum, location = _circular_scan(z, window)
        output[f"block_scan_max_w{window}"] = maximum
        output[f"block_scan_min_w{window}"] = minimum
        output[f"block_scan_abs_w{window}"] = max(abs(maximum), abs(minimum))
        output[f"block_scan_location_w{window}"] = location
    return output


def engineer_block_features(
    values: pd.Series,
    windows: tuple[int, ...] = DEFAULT_BLOCK_WINDOWS,
) -> pd.DataFrame:
    """Vectorize block feature extraction over a Series, preserving its index."""
    rows = [engineer_one_block_signal(value, windows) for value in values]
    return pd.DataFrame(rows, index=values.index, dtype=np.float32)


def engineer_wafer_features(
    wafer: pd.DataFrame,
    include_block: bool = True,
    spatial_windows: tuple[int, ...] = DEFAULT_SPATIAL_WINDOWS,
    block_windows: tuple[int, ...] = DEFAULT_BLOCK_WINDOWS,
) -> pd.DataFrame:
    """Construct all leakage-safe features for one wafer."""
    result = engineer_spatial_features(wafer, windows=spatial_windows)
    if include_block:
        _require_columns(result, ("block_readings",))
        block = engineer_block_features(result["block_readings"], block_windows)
        result = pd.concat((result, block), axis=1)
    return result.drop(columns=["block_readings"], errors="ignore")

