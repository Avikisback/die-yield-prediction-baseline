"""Recover each wafer's failure rate from its own unlabelled scores.

``generate_die_features`` draws ``wafer_base_rate`` from an exponential
distribution and every die's hazard is proportional to it, so the per-wafer rate
spans roughly 0% to 25% and is statistically independent of anything visible
before the test.  No amount of feature engineering can predict it.

It can, however, be *measured* after the fact.  The scores on one wafer are a
two-component mixture -- passing dies drawn from one distribution, failing dies
from another -- and the mixing weight is the wafer's failure rate.  Estimating
that weight uses only the scores of the wafer being predicted, never its
labels, so it is available for the unlabelled validation split.

The estimate then enters the model as a prior shift, which is the textbook
correction when the class balance at predict time differs from training:

    logit_new = logit_old + logit(rate_wafer) - logit(rate_overall)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

GRID_POINTS = 512
BANDWIDTH = 0.25
MIN_RATE = 1e-4
MAX_RATE = 0.5


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


@dataclass
class MixtureReference:
    """Score densities for passing and failing dies, held per wafer.

    Keeping each wafer's contribution separately makes a leave-one-wafer-out
    reference cheap, which is what stops a wafer's own labels from leaking into
    the estimate of its own rate.
    """

    grid: np.ndarray
    pass_density: dict
    fail_density: dict
    centre: float
    scale: float
    overall_rate: float

    @classmethod
    def fit(cls, score, label, wafer, bandwidth: float = BANDWIDTH):
        score = np.asarray(score, dtype=np.float64)
        centre = float(score.mean())
        scale = float(score.std()) or 1.0
        z = (score - centre) / scale
        grid = np.linspace(z.min() - 1.0, z.max() + 1.0, GRID_POINTS)
        label = np.asarray(label).astype(bool)
        wafer = np.asarray(wafer)

        pass_density = {}
        fail_density = {}
        for key in np.unique(wafer):
            here = wafer == key
            pass_density[key] = cls._kernel(z[here & ~label], grid, bandwidth)
            fail_density[key] = cls._kernel(z[here & label], grid, bandwidth)
        return cls(
            grid=grid,
            pass_density=pass_density,
            fail_density=fail_density,
            centre=centre,
            scale=scale,
            overall_rate=float(label.mean()),
        )

    @staticmethod
    def _kernel(values, grid, bandwidth):
        """Unnormalised Gaussian kernel sum; counts are kept so wafers pool."""
        if values.size == 0:
            return np.zeros_like(grid)
        delta = (grid[:, None] - values[None, :]) / bandwidth
        return np.exp(-0.5 * delta**2).sum(axis=1) / (bandwidth * np.sqrt(2 * np.pi))

    def densities(self, exclude=None):
        keys = [k for k in self.pass_density if k != exclude]
        f0 = np.sum([self.pass_density[k] for k in keys], axis=0)
        f1 = np.sum([self.fail_density[k] for k in keys], axis=0)
        f0 = f0 / max(f0.sum(), 1e-12)
        f1 = f1 / max(f1.sum(), 1e-12)
        return f0 + 1e-12, f1 + 1e-12

    def estimate_rate(self, score, exclude=None) -> float:
        """Maximum-likelihood mixing weight for one wafer's scores."""
        f0, f1 = self.densities(exclude)
        z = (np.asarray(score, dtype=np.float64) - self.centre) / self.scale
        d0 = np.interp(z, self.grid, f0)
        d1 = np.interp(z, self.grid, f1)

        def negative_log_likelihood(rate):
            return -np.log(rate * d1 + (1.0 - rate) * d0).sum()

        result = minimize_scalar(
            negative_log_likelihood, bounds=(MIN_RATE, MAX_RATE), method="bounded"
        )
        return float(result.x)


def estimate_rates(score, wafer, reference, leave_out: bool):
    """Recovered failure rate for every wafer, from that wafer's scores alone."""
    score = np.asarray(score, dtype=np.float64)
    wafer = np.asarray(wafer)
    return {
        str(key): reference.estimate_rate(
            score[wafer == key], exclude=key if leave_out else None
        )
        for key in np.unique(wafer)
    }


def prior_shift(score, probability, wafer, reference, leave_out: bool,
                alpha: float = 1.0, rates=None):
    """Re-weight predictions wafer by wafer using the recovered rate.

    ``alpha`` shrinks the offset toward zero.  The full correction assumes the
    recovered rate is exact; it is not.  A wafer of ~950 dies at a 2% rate
    carries about 19 failures, so its rate is known to well under a factor of
    two -- and in log-odds a factor of two near 2% is a shift of 0.7, comparable
    to the whole spread of the die-level score.  Shrinking trades a little of
    the between-wafer signal for immunity to that estimation error, and the
    amount to shrink is chosen on training out-of-fold scores only.
    """
    probability = np.asarray(probability, dtype=np.float64)
    wafer = np.asarray(wafer)
    if rates is None:
        rates = estimate_rates(score, wafer, reference, leave_out)
    adjusted = _logit(probability).copy()
    baseline = _logit(reference.overall_rate)
    for key, rate in rates.items():
        here = wafer == key
        adjusted[here] += alpha * (_logit(rate) - baseline)
    return 1.0 / (1.0 + np.exp(-adjusted)), rates
