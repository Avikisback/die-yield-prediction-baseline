"""Scan statistics over the 2,000 sub-die block readings.

``generate_block_readings`` builds each die's signal as white noise that is
partially smoothed with a 5-tap kernel, and then, for a failing die only, adds
about 100 isolated positive spikes whose positions are clustered around one
random seed with a spread of ``0.05 * k``.  The detector that matches that
description is a circular scan with a Gaussian window; the width that maximises
signal-to-noise is close to the cluster's own spread.

Two details decide whether the scan works:

* the window must wrap, because the cluster is placed modulo ``k``; and
* the output must be divided by a *constant* gain, not by each die's own
  spread.  A heavily smoothed series has only a handful of effective degrees of
  freedom, so its own robust spread is both unstable and inflated by the very
  bump the scan is looking for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

READINGS = 2000
GAUSSIAN_WIDTHS = (25, 50, 100, 150, 200, 300, 400)
BOXCAR_WIDTHS = (64, 256, 512)
LOCATION_WIDTHS = (50, 150)
SMOOTHING_KERNEL = 5
SMOOTHING_MIX = 0.4


def _colouring(k: int, kernel: int, mix: float) -> np.ndarray:
    """Transfer function of the generator's ``0.6*u + 0.4*boxcar(u)`` step."""
    freq = np.fft.fftfreq(k)
    offsets = np.arange(-(kernel // 2), kernel // 2 + 1)
    response = np.cos(2 * np.pi * np.outer(freq, offsets)).sum(axis=1) / kernel
    return (1.0 - mix) + mix * response


class ScanBank:
    """Pre-computed circular filters and their exact null-hypothesis gains."""

    def __init__(
        self,
        k: int = READINGS,
        gaussian_widths: tuple[int, ...] = GAUSSIAN_WIDTHS,
        boxcar_widths: tuple[int, ...] = BOXCAR_WIDTHS,
    ) -> None:
        self.k = k
        self.gaussian_widths = gaussian_widths
        self.boxcar_widths = boxcar_widths
        colouring = _colouring(k, SMOOTHING_KERNEL, SMOOTHING_MIX)
        power = colouring**2
        self._filters: dict[str, np.ndarray] = {}
        self._gains: dict[str, float] = {}

        full = np.fft.fftfreq(k)
        for width in gaussian_widths:
            response = np.exp(-2 * (np.pi * full * width) ** 2)
            self._store(f"gauss{width}", response, power)
        for width in boxcar_widths:
            kernel = np.zeros(k)
            kernel[:width] = 1.0
            response = np.fft.fft(np.roll(kernel, -(width // 2))).real
            self._store(f"box{width}", response, power)

    def _store(self, name: str, response: np.ndarray, power: np.ndarray) -> None:
        # Variance of the filtered series when the input is a unit-variance draw
        # from the generator's coloured base process.
        gain = float(np.sqrt((power * response**2).sum() / power.sum()))
        self._filters[name] = response[: self.k // 2 + 1].copy()
        self._gains[name] = max(gain, 1e-9)

    @property
    def names(self) -> list[str]:
        return list(self._filters)

    def scan(self, standardised: np.ndarray) -> dict[str, np.ndarray]:
        """Return scan maxima (and cluster locations) for a batch of dies."""
        spectrum = np.fft.rfft(standardised, axis=1)
        output: dict[str, np.ndarray] = {}
        for name, response in self._filters.items():
            filtered = np.fft.irfft(spectrum * response[None, :], n=self.k, axis=1)
            filtered /= self._gains[name]
            output[f"block_scan_{name}"] = filtered.max(axis=1).astype(np.float32)
            width = name.replace("gauss", "")
            if name.startswith("gauss") and int(width) in LOCATION_WIDTHS:
                output[f"block_cluster_at_{name}"] = (
                    filtered.argmax(axis=1) / self.k
                ).astype(np.float32)
        return output


def parse(values: pd.Series, k: int = READINGS) -> np.ndarray:
    """Turn the space-separated reading strings into one dense array."""
    out = np.empty((len(values), k), dtype=np.float32)
    for position, item in enumerate(values):
        if isinstance(item, str):
            row = np.fromstring(item, sep=" ", dtype=np.float32)
        else:
            row = np.asarray(item, dtype=np.float32).reshape(-1)
        if len(row) != k:
            padded = np.full(k, np.nan, dtype=np.float32)
            padded[: min(k, len(row))] = row[:k]
            row = padded
        out[position] = row
    return out


def _tail_statistics(z: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(z, axis=1)
    width = z.shape[1]
    one = max(1, width // 100)
    five = max(1, width // 20)
    centred = z - z.mean(axis=1, keepdims=True)
    spread = z.std(axis=1)
    safe = np.where(spread > 1e-9, spread, 1.0)
    normalised = centred / safe[:, None]
    return {
        "block_mean": z.mean(axis=1),
        "block_std": spread,
        "block_skew": (normalised**3).mean(axis=1),
        "block_kurtosis": (normalised**4).mean(axis=1) - 3.0,
        "block_q95": ordered[:, int(0.95 * width)],
        "block_q99": ordered[:, int(0.99 * width)],
        "block_top1pct_mean": ordered[:, -one:].mean(axis=1),
        "block_top5pct_mean": ordered[:, -five:].mean(axis=1),
        "block_bottom1pct_mean": ordered[:, :one].mean(axis=1),
        "block_frac_gt_2": (z > 2.0).mean(axis=1),
        "block_frac_gt_3": (z > 3.0).mean(axis=1),
        "block_frac_lt_m2": (z < -2.0).mean(axis=1),
        # The defect signature is one-sided, so the gap between the upper and
        # lower tail separates a real cluster from a die that is simply noisy.
        "block_tail_asymmetry": ordered[:, -one:].mean(axis=1)
        + ordered[:, :one].mean(axis=1),
    }


def engineer(values: pd.Series, bank: ScanBank | None = None) -> pd.DataFrame:
    """Compute every ``block_*`` column for one wafer's dies."""
    bank = bank or ScanBank()
    readings = parse(values, bank.k)
    centre = np.median(readings, axis=1, keepdims=True)
    spread = 1.4826 * np.median(np.abs(readings - centre), axis=1, keepdims=True)
    fallback = readings.std(axis=1, keepdims=True)
    spread = np.where(spread > 1e-6, spread, fallback)
    spread = np.where(spread > 1e-6, spread, 1.0)
    z = ((readings - centre) / spread).astype(np.float64)

    columns: dict[str, np.ndarray] = {}
    columns.update(_tail_statistics(z))
    columns.update(bank.scan(z))
    columns["block_raw_median"] = centre.ravel()
    columns["block_raw_scale"] = spread.ravel()
    frame = pd.DataFrame(
        {name: np.asarray(value, dtype=np.float32) for name, value in columns.items()},
        index=values.index,
    )
    return frame
