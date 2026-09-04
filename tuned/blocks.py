"""A likelihood-ratio detector for the 2,000 sub-die block readings.

``generate_block_readings`` gives every die white noise partially smoothed by a
5-tap kernel, and gives a failing die about 100 extra positive spikes clustered
around one uniformly random seed.  The spikes are added *after* the smoothing,
so signal and noise do not share a spectrum, and the seed is drawn per die and
is not recoverable from anything else.

That description names the optimal statistic.  Write ``u`` for the readings
after the smoothing has been divided back out, ``sigma`` for the noise scale,
and ``q`` for the chance that a block ``d`` away from the seed is one of the
anomalous ones.  Conditional on the seed sitting at ``t``,

    log LR(t) = sum_p log[ 1 + q(p - t) * (r(u_p) - 1) ]
    r(u)      = N(u; mu, sigma^2 + tau^2) / N(u; 0, sigma^2)

and for the ``q`` values that arise here the leading term is a plain circular
correlation of ``r(u) - 1`` against ``q`` -- one FFT per die, every seed at
once.  The seed itself is then integrated out rather than maximised over:

    log LR = log mean_t exp(lambda * c(t))

which is the Bayes statistic for a uniformly drawn seed.

Three details matter more than they look:

* **Divide by a constant, never by the die's own spread.**  The scan is smooth,
  so a die's own spread across shifts is unstable and is inflated by the very
  bump being looked for.
* **Whiten first.**  ``r`` is a per-sample transform and the derivation assumes
  independent samples; the readings are smoothed, so they are not.  Dividing the
  spectrum by the estimated response restores it.
* **``r`` is nonlinear.**  A matched filter is its linear approximation, and
  replacing the filter by ``r - 1`` is where most of the gain sits.

Nothing here reads ``config.yaml``: the noise spectrum is estimated from the
readings themselves, and the remaining constants are covered by a small bank of
values the head is free to weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

READINGS = 2000
WHITENING = (0.0, 1.5)
# Amplitudes are multiples of the *noise* standard deviation at that whitening
# level, not raw reading units.  The generator's spike is 4.5 against a noise
# spread of about 10.5, so the interesting range is a fraction of a sigma; a
# bank stated in raw units would be testing for spikes seven times too large.
AMPLITUDES = (0.2, 0.45, 0.8, 1.5)
ENVELOPES = (90, 150)
LAMBDAS = (1.0, 1.8, 3.2)
SPIKE_SPREAD = 0.30          # tau/mu, the only shape constant that is assumed
CHI2_MEDIAN = 0.454936423    # median of a chi-square with one degree of freedom

GLOBAL_NAMES = (
    "block_level",
    "block_median",
    "block_scale",
    "block_skew",
    "block_kurtosis",
    "block_frac_gt_2",
    "block_frac_gt_3",
    "block_top1pct",
    "block_bottom1pct",
)


def parse(values: pd.Series, k: int = READINGS) -> np.ndarray:
    """Dense float array from the space-separated reading strings.

    Short rows are padded with the row's own mean rather than with NaN, because
    a single NaN would poison every FFT-based statistic for that die.  For rows
    of the expected length the padding never runs.
    """
    out = np.empty((len(values), k), dtype=np.float64)
    for position, item in enumerate(values):
        if isinstance(item, str):
            row = np.fromstring(item, sep=" ", dtype=np.float64)
        else:
            row = np.asarray(item, dtype=np.float64).reshape(-1)
        row = row[np.isfinite(row)]
        if len(row) >= k:
            out[position] = row[:k]
        else:
            filler = float(row.mean()) if len(row) else 0.0
            out[position, : len(row)] = row
            out[position, len(row):] = filler
    return out


@dataclass
class NoiseModel:
    """Readings-only description of the null: level, scale and spectrum.

    Estimated from every die in the split, labelled or not.  Failing dies are a
    few percent of the population and carry the anomaly on about 5% of their own
    blocks, so their effect on a median-based level and on an averaged
    periodogram is far below the precision any of this needs.
    """

    k: int
    level: float
    scale: float
    response: np.ndarray = field(repr=False)

    @classmethod
    def fit(cls, readings: np.ndarray) -> "NoiseModel":
        k = int(readings.shape[1])
        level = float(np.median(readings))
        centred = readings - level
        scale = float(1.4826 * np.median(np.abs(centred)))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(centred.std()) or 1.0
        power = (np.abs(np.fft.rfft(centred, axis=1)) ** 2).mean(axis=0)
        response = np.sqrt(power / power.mean())
        return cls(k=k, level=level, scale=scale,
                   response=np.maximum(response, 1e-3))

    def to_dict(self) -> dict:
        return {"k": self.k, "level": self.level, "scale": self.scale,
                "response": self.response.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "NoiseModel":
        return cls(k=int(payload["k"]), level=float(payload["level"]),
                   scale=float(payload["scale"]),
                   response=np.asarray(payload["response"], dtype=np.float64))


def _envelope(k: int, width: float) -> np.ndarray:
    offset = np.arange(k)
    offset = np.minimum(offset, k - offset)
    envelope = np.exp(-0.5 * (offset / width) ** 2)
    return envelope / envelope.sum()


def scan_names(whitening=WHITENING, amplitudes=AMPLITUDES,
               envelopes=ENVELOPES, lambdas=LAMBDAS) -> list[str]:
    names: list[str] = []
    for gamma in whitening:
        for amplitude in amplitudes:
            for width in envelopes:
                stem = f"block_lr_g{gamma:g}_a{amplitude:g}_w{width}"
                names.append(stem + "_max")
                names.extend(f"{stem}_m{value:g}" for value in lambdas)
    return names


def feature_names(**kwargs) -> list[str]:
    return scan_names(**kwargs) + list(GLOBAL_NAMES)


def engineer(
    readings: np.ndarray,
    noise: NoiseModel,
    whitening=WHITENING,
    amplitudes=AMPLITUDES,
    envelopes=ENVELOPES,
    lambdas=LAMBDAS,
    reference: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Block statistics for a batch of dies.

    ``reference`` holds the null centre and spread of every scan so that a batch
    scored later is normalised by the same constants as the batch the head was
    fitted on.  Passing ``None`` estimates them here and returns them.
    """
    k = noise.k
    centred = (readings - noise.level) / noise.scale
    spectrum = np.fft.rfft(centred, axis=1)
    reference = {} if reference is None else dict(reference)
    columns: dict[str, np.ndarray] = {}
    envelope_fft = {w: np.conj(np.fft.rfft(_envelope(k, w))) for w in envelopes}

    for gamma in whitening:
        u = np.fft.irfft(spectrum / noise.response[None, :] ** gamma, n=k, axis=1)
        sigma2 = float(np.median(u ** 2)) / CHI2_MEDIAN
        sigma = np.sqrt(sigma2)
        for ratio in amplitudes:
            amplitude = ratio * sigma
            tau2 = (SPIKE_SPREAD * amplitude) ** 2
            total = sigma2 + tau2
            density_ratio = np.sqrt(sigma2 / total) * np.exp(
                -0.5 * ((u - amplitude) ** 2 / total - u ** 2 / sigma2)
            )
            evidence = np.fft.rfft(density_ratio - 1.0, axis=1)
            for width in envelopes:
                scan = np.fft.irfft(
                    evidence * envelope_fft[width][None, :], n=k, axis=1
                )
                stem = f"block_lr_g{gamma:g}_a{ratio:g}_w{width}"
                key = stem + "__null"
                if key not in reference:
                    reference[key] = (float(scan.mean()),
                                      float(scan.std()) or 1.0)
                centre, spread = reference[key]
                scan = (scan - centre) / spread
                columns[stem + "_max"] = scan.max(axis=1).astype(np.float32)
                for value in lambdas:
                    peak = (value * scan).max(axis=1)
                    columns[f"{stem}_m{value:g}"] = (
                        np.log(np.exp(value * scan - peak[:, None]).mean(axis=1))
                        + peak
                    ).astype(np.float32)

    ordered = np.sort(centred, axis=1)
    one = max(1, k // 100)
    spread = centred.std(axis=1)
    safe = np.where(spread > 1e-9, spread, 1.0)
    normalised = (centred - centred.mean(axis=1, keepdims=True)) / safe[:, None]
    columns.update(
        {
            "block_level": centred.mean(axis=1).astype(np.float32),
            "block_median": np.median(centred, axis=1).astype(np.float32),
            "block_scale": spread.astype(np.float32),
            "block_skew": (normalised ** 3).mean(axis=1).astype(np.float32),
            "block_kurtosis": ((normalised ** 4).mean(axis=1) - 3.0).astype(np.float32),
            "block_frac_gt_2": (centred > 2.0).mean(axis=1).astype(np.float32),
            "block_frac_gt_3": (centred > 3.0).mean(axis=1).astype(np.float32),
            "block_top1pct": ordered[:, -one:].mean(axis=1).astype(np.float32),
            "block_bottom1pct": ordered[:, :one].mean(axis=1).astype(np.float32),
        }
    )
    return columns, reference
