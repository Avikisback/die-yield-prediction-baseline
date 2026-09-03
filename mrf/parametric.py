"""Remove the per-wafer process gradient from each parametric measurement.

The generator builds every feature as

    x = N(base_mean, base_std)
        + radial_map * U(-1, 1) * 0.30 * base_std
        + linear_map * U(-1, 1) * 0.15 * base_std
        + fail_density * 0.20 * fail_shift
        + fail_shift * (1 for a hard failure, U(0.05, 0.25) for a marginal one)

with ``fail_shift`` between 0.1 and 0.5 ``base_std``.  The two gradient
coefficients are redrawn for every (wafer, feature) pair, so across wafers they
behave as noise with a standard deviation of roughly 0.17 ``base_std`` -- the
same order as the signal for a hard failure and several times larger than the
signal for a marginal one.  No model fitted on raw feature values can remove a
coefficient that changes wafer to wafer.

Fitting the gradient inside each wafer and keeping the residual removes it
exactly, because the gradient lies in the span of ``[1, xn, yn, radius]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrf.spatial import coordinate_basis

PREFIX = "dz_"


def detrended_names(feature_columns) -> list[str]:
    return [f"{PREFIX}{name}" for name in feature_columns]


def detrend_wafer(
    wafer: pd.DataFrame,
    feature_columns: list[str],
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-die standardised residuals and the fitted gradient norms.

    The gradient is fitted on ``old_label == 0`` dies only.  Pre-test failures
    carry the full ``fail_shift`` on every feature, so leaving them in would
    drag the fit toward the defect signature the residual is meant to expose.
    ``old_label`` is pre-test information, so this is available at predict time.
    """
    if basis is None:
        basis = coordinate_basis(wafer)
    values = wafer.loc[:, feature_columns].to_numpy(dtype=np.float64)
    clean = wafer["old_label"].to_numpy(dtype=np.int8) == 0
    # A wafer that is almost all pre-test failures cannot support a 7-term fit.
    if clean.sum() < 4 * basis.shape[1]:
        clean = np.ones(len(wafer), dtype=bool)

    coefficients, *_ = np.linalg.lstsq(basis[clean], values[clean], rcond=None)
    residual = values - basis @ coefficients

    reference = residual[clean]
    centre = np.median(reference, axis=0)
    scale = 1.4826 * np.median(np.abs(reference - centre), axis=0)
    fallback = reference.std(axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)

    standardised = (residual - centre) / scale
    # ``base_std`` is a per-feature constant, so the fitted coefficients divided
    # by it are directly comparable across wafers and make a usable summary of
    # how strong this wafer's gradients were.
    gradient_strength = np.linalg.norm(coefficients[1:4] / scale, axis=0)
    return standardised.astype(np.float32), gradient_strength


def summary_columns(standardised: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap order statistics over the 500 detrended measurements per die.

    A die that fails on many parameters at once looks different from one that
    drifts on a handful, and a linear model over the individual z-scores cannot
    express that.
    """
    absolute = np.abs(standardised)
    ordered = np.sort(standardised, axis=1)
    width = standardised.shape[1]
    top = max(1, int(round(0.05 * width)))
    return {
        "dstat_mean": standardised.mean(axis=1),
        "dstat_std": standardised.std(axis=1),
        "dstat_mean_abs": absolute.mean(axis=1),
        "dstat_max": ordered[:, -1],
        "dstat_min": ordered[:, 0],
        "dstat_top5pct_mean": ordered[:, -top:].mean(axis=1),
        "dstat_bottom5pct_mean": ordered[:, :top].mean(axis=1),
        "dstat_frac_gt_2": (standardised > 2.0).mean(axis=1),
        "dstat_frac_lt_m2": (standardised < -2.0).mean(axis=1),
        "dstat_rms": np.sqrt((standardised.astype(np.float64) ** 2).mean(axis=1)),
    }
