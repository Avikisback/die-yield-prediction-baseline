"""Recover each wafer's failure rate as a posterior, not a point estimate.

``generate_die_features`` draws ``wafer_base_rate`` from an exponential
distribution and multiplies every die's hazard by it.  It is independent of the
wafer map, of the pre-test failures and of every measurement taken before the
test, so no feature predicts it -- but the wafer's own dies measure it, because
each die's evidence likelihood ratio says how surprising that die looks if the
rate were ``r``:

    log L(r) = sum_i log[ 1 + pi_i(r) * (LR_i - 1) ],  pi_i(r) = min(r * h_i, 0.4)

with ``h_i`` the pre-test hazard shape and ``LR_i`` the die's own evidence.  No
label enters, so the same estimate is available on the unlabelled split.

``mrf.calibrate`` solves a related problem -- a two-component mixture on the
score, maximised, then shrunk toward zero by a hand-picked ``alpha``.  Two
things are different here.

**The hazard shape is used.**  Dies do not share a failure probability; a die
at the wafer edge next to a defect cluster is several times more likely to fail
than one in a clean centre.  Mixing with a single weight throws that away.

**The rate is integrated out, not maximised.**  A wafer of ~950 dies at a 2%
rate carries about 19 failures, so its rate is genuinely uncertain, and in
log-odds a factor-of-two error there is a shift of 0.7 -- comparable to the
whole spread of the die-level evidence.  ``alpha`` exists to blunt that.
Carrying the exponential prior through and reporting

    P(fail_i) = integral P(fail_i | r) p(r | wafer) dr

shrinks for the same reason ``alpha`` does, but by an amount the wafer's own
sample size decides rather than a constant chosen once for every wafer.  A
quiet wafer with little evidence keeps the population prior; a wafer with a
hundred obvious failures moves all the way.

The prior scale is fitted on the training wafers only and is a single number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = np.exp(np.linspace(np.log(2e-4), np.log(0.25), 240))
DEFAULT_SCALE = 0.02


def observed_rates(shape: np.ndarray, label: np.ndarray,
                   wafer: np.ndarray) -> dict[str, float]:
    """Per-wafer rate implied by the labels: failures over summed hazard shape.

    The 0.4 cap on ``pi_i`` is ignored here; it binds only when the rate is
    above roughly 10%, which happens on a handful of wafers and moves this
    moment estimate by less than the estimate's own noise.
    """
    shape = np.asarray(shape, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    wafer = np.asarray(wafer)
    out: dict[str, float] = {}
    for key in np.unique(wafer):
        here = wafer == key
        out[str(key)] = float(label[here].sum()) / max(float(shape[here].sum()), 1e-9)
    return out


def fit_prior_scale(rates) -> float:
    """Maximum-likelihood exponential scale over the training wafers' rates."""
    values = np.asarray(list(rates.values()), dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0:
        return DEFAULT_SCALE
    return float(max(values.mean(), 1e-4))


@dataclass
class RatePosterior:
    """Grid posterior over one wafer's base rate.

    The grid is log-spaced, so the quadrature weight of each point is its own
    spacing.  Leaving that out puts far too much mass on the crowded low end and
    biases every posterior mean downward -- by a factor of four on a wafer whose
    dies say nothing.
    """

    grid: np.ndarray
    log_weight: np.ndarray

    @property
    def weight(self) -> np.ndarray:
        w = np.exp(self.log_weight - self.log_weight.max()) * np.gradient(self.grid)
        return w / w.sum()

    @property
    def mean(self) -> float:
        return float(self.weight @ self.grid)

    @property
    def mode(self) -> float:
        return float(self.grid[int(np.argmax(self.log_weight))])


def _clip_prior(rate_grid: np.ndarray, shape: np.ndarray, cap: float) -> np.ndarray:
    """``pi_i(r)`` for every grid rate and die: shape (grid, dies)."""
    return np.minimum(rate_grid[:, None] * shape[None, :], cap)


def wafer_posterior(evidence_logit: np.ndarray, shape: np.ndarray,
                    prior_scale: float, grid: np.ndarray = GRID,
                    cap: float = 0.4) -> RatePosterior:
    """Posterior over the base rate from one wafer's dies, using no labels."""
    evidence_logit = np.asarray(evidence_logit, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.float64)
    ratio = np.exp(np.clip(evidence_logit, -30.0, 30.0))
    pi = _clip_prior(grid, shape, cap)
    # log sum over dies of log(1 + pi * (LR - 1)); the pass-density factor is
    # common to every rate and drops out of the normalisation.
    inner = np.maximum(1.0 + pi * (ratio[None, :] - 1.0), 1e-300)
    log_likelihood = np.log(inner).sum(axis=1)
    log_prior = -grid / max(prior_scale, 1e-6)
    return RatePosterior(grid=grid, log_weight=log_likelihood + log_prior)


def posterior_probabilities(evidence_logit: np.ndarray, shape: np.ndarray,
                            wafer: np.ndarray, prior_scale: float,
                            grid: np.ndarray = GRID, cap: float = 0.4):
    """Per-die posterior failure probability with the wafer rate integrated out.

    Returns the probabilities and, for the report, each wafer's posterior mean
    rate.
    """
    evidence_logit = np.asarray(evidence_logit, dtype=np.float64)
    shape = np.asarray(shape, dtype=np.float64)
    wafer = np.asarray(wafer)
    probability = np.empty(len(shape), dtype=np.float64)
    rates: dict[str, float] = {}
    for key in np.unique(wafer):
        here = wafer == key
        posterior = wafer_posterior(evidence_logit[here], shape[here],
                                    prior_scale, grid, cap)
        weight = posterior.weight
        ratio = np.exp(np.clip(evidence_logit[here], -30.0, 30.0))
        pi = _clip_prior(grid, shape[here], cap)
        per_rate = pi * ratio[None, :] / np.maximum(
            1.0 + pi * (ratio[None, :] - 1.0), 1e-300
        )
        probability[here] = weight @ per_rate
        rates[str(key)] = posterior.mean
    return probability, rates


def offsets(shape: np.ndarray, wafer: np.ndarray, rates, cap: float = 0.4,
            fallback: float = DEFAULT_SCALE) -> np.ndarray:
    """``logit(pi_i)`` for a dictionary of per-wafer rates."""
    shape = np.asarray(shape, dtype=np.float64)
    wafer = np.asarray(wafer)
    rate = np.array([float(rates.get(str(key), fallback)) for key in wafer])
    pi = np.clip(rate * shape, 1e-9, min(cap, 1 - 1e-9))
    return np.log(pi / (1.0 - pi))
