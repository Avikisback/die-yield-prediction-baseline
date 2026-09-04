"""An additive log-odds head with a fixed offset.

The generator makes the three resolutions conditionally independent given the
label, so the posterior is additive in log-odds:

    logit P(fail) = logit(prior) + logLR_parametric + logLR_block

Two consequences drive the design of this module.

**The offset is not a coefficient.**  ``logit(prior)`` enters with weight
exactly one; it is known, not fitted.  Ordinary logistic regression has nowhere
to put a term like that, so the fit here is a small penalised likelihood solved
directly.

**The evidence terms are not linear.**  Take the parametric channel.  65% of
failures receive only 5-25% of the shift, so the fail-class score is a two
component mixture and its log-likelihood ratio

    log[ 0.35 * phi(s - d) + 0.65 * phi(s - u*d) ] - log phi(s)

has slope ``u*d`` over most of the range and slope ``d`` in the upper tail --
a factor of five apart.  A single fitted coefficient has to average the two and
is wrong at both ends.  Each channel therefore enters through a cubic B-spline
with a second-difference roughness penalty, which recovers the kink from the
data without being told where it is.

The penalty is on second differences rather than on coefficient size, so a
straight line costs nothing and only curvature is paid for; a channel that
really is linear stays linear.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

DEGREE = 3


def spline_knots(x: np.ndarray, n_bases: int = 8) -> np.ndarray:
    """Interior knots at quantiles, boundary knots at the full observed range.

    The boundary has to be the range and not, say, the 0.1% and 99.9% quantiles.
    Beyond the last knot ``spline_design`` clamps, so a boundary inside the data
    would give every die above it an identical parametric term -- and the dies
    above the 99.9% quantile here are almost all real failures whose ordering is
    exactly what the top of the ranking is made of.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    interior = int(max(n_bases - DEGREE - 1, 0))
    if interior:
        quantiles = np.linspace(0, 1, interior + 2)[1:-1]
        inner = np.unique(np.clip(np.quantile(x, quantiles),
                                  lo + 1e-9, hi - 1e-9))
    else:
        inner = np.empty(0)
    return np.concatenate(([lo] * (DEGREE + 1), inner, [hi] * (DEGREE + 1)))


def spline_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """B-spline design matrix, with values outside the knot range clamped.

    Clamping rather than extrapolating matters: a held-out die can score beyond
    anything seen in training, and a cubic extrapolated past its last knot can
    swing by orders of magnitude.  Clamping makes the fitted function flat out
    there, which is the conservative reading of "no evidence either way".
    """
    x = np.clip(np.asarray(x, dtype=np.float64), knots[0], knots[-1])
    return np.asarray(
        BSpline.design_matrix(x, knots, DEGREE, extrapolate=False).todense()
    )


def difference_penalty(n: int, order: int = 2) -> np.ndarray:
    """``D.T @ D`` for the ``order``-th difference operator on ``n`` bases."""
    d = np.eye(n)
    for _ in range(order):
        d = np.diff(d, axis=0)
    return d.T @ d


@dataclass
class Block:
    """One additive term: a channel name, its knots, and where it sits in beta."""

    name: str
    knots: np.ndarray = field(repr=False)
    start: int = 0
    width: int = 0
    linear: bool = False


class AdditiveHead:
    """Penalised logistic with an offset and one smooth term per channel.

    ``smooth`` columns get a spline expansion, ``linear`` columns enter as they
    are.  Fitting minimises

        sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ]
        + (lam_smooth/2) * sum_j beta_j' P beta_j + (lam_linear/2) * ||beta_lin||^2

    with ``eta = offset + intercept + X beta``.
    """

    def __init__(self, smooth: tuple[str, ...], linear: tuple[str, ...] = (),
                 n_bases: int = 8, lam_smooth: float = 20.0,
                 lam_linear: float = 10.0):
        self.smooth = tuple(smooth)
        self.linear = tuple(linear)
        self.n_bases = int(n_bases)
        self.lam_smooth = float(lam_smooth)
        self.lam_linear = float(lam_linear)

    # ------------------------------------------------------------------ setup
    def _design(self, frame) -> np.ndarray:
        parts = []
        for block in self.blocks_:
            column = np.asarray(frame[block.name], dtype=np.float64)
            if block.linear:
                parts.append(((column - self.centre_[block.name])
                              / self.scale_[block.name])[:, None])
            else:
                parts.append(spline_design(column, block.knots))
        return np.hstack(parts) if parts else np.zeros((len(frame), 0))

    def _penalty(self, n_columns: int) -> np.ndarray:
        penalty = np.zeros((n_columns, n_columns))
        for block in self.blocks_:
            slot = slice(block.start, block.start + block.width)
            if block.linear:
                penalty[slot, slot] = self.lam_linear * np.eye(block.width)
            else:
                penalty[slot, slot] = self.lam_smooth * difference_penalty(block.width)
        return penalty

    # -------------------------------------------------------------------- fit
    def fit(self, frame, y, offset, sample_weight=None) -> "AdditiveHead":
        y = np.asarray(y, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        weight = (np.ones_like(y) if sample_weight is None
                  else np.asarray(sample_weight, dtype=np.float64))

        self.blocks_: list[Block] = []
        self.centre_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        cursor = 0
        for name in self.smooth:
            column = np.asarray(frame[name], dtype=np.float64)
            knots = spline_knots(column, self.n_bases)
            width = len(knots) - DEGREE - 1
            self.blocks_.append(Block(name, knots, cursor, width, linear=False))
            cursor += width
        for name in self.linear:
            column = np.asarray(frame[name], dtype=np.float64)
            self.centre_[name] = float(np.mean(column))
            self.scale_[name] = float(np.std(column)) or 1.0
            self.blocks_.append(Block(name, np.empty(0), cursor, 1, linear=True))
            cursor += 1

        x = self._design(frame)
        penalty = self._penalty(cursor)
        total = float(weight.sum())

        def objective(theta):
            intercept, beta = theta[0], theta[1:]
            eta = offset + intercept + x @ beta
            # log(1 + exp(eta)) evaluated without overflowing for large |eta|
            softplus = np.logaddexp(0.0, eta)
            loss = float(np.sum(weight * (softplus - y * eta))) / total
            probability = 1.0 / (1.0 + np.exp(-eta))
            residual = weight * (probability - y) / total
            gradient = np.concatenate(([residual.sum()], x.T @ residual))
            loss += 0.5 * float(beta @ penalty @ beta) / total
            gradient[1:] += (penalty @ beta) / total
            return loss, gradient

        start = np.zeros(cursor + 1)
        result = minimize(objective, start, jac=True, method="L-BFGS-B",
                          options={"maxiter": 800, "maxfun": 2000})
        if not result.success:
            # Silently returning a half-optimised head would show up as a
            # mysterious drop in one fold and nowhere else.
            warnings.warn(f"additive head did not converge: {result.message}",
                          RuntimeWarning, stacklevel=2)
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        self.converged_ = bool(result.success)
        self.n_iter_ = int(result.nit)
        return self

    # ------------------------------------------------------------- prediction
    def evidence(self, frame) -> np.ndarray:
        """Log-odds contributed by the measurements, excluding the offset."""
        return self.intercept_ + self._design(frame) @ self.coef_

    def decision(self, frame, offset) -> np.ndarray:
        return np.asarray(offset, dtype=np.float64) + self.evidence(frame)

    def predict_proba(self, frame, offset) -> np.ndarray:
        eta = self.decision(frame, offset)
        return 1.0 / (1.0 + np.exp(-eta))

    def partial(self, frame) -> dict[str, np.ndarray]:
        """Per-term contribution to the log-odds; sums to ``evidence`` exactly."""
        out: dict[str, np.ndarray] = {}
        for block in self.blocks_:
            column = np.asarray(frame[block.name], dtype=np.float64)
            beta = self.coef_[block.start: block.start + block.width]
            if block.linear:
                basis = ((column - self.centre_[block.name])
                         / self.scale_[block.name])[:, None]
            else:
                basis = spline_design(column, block.knots)
            out[block.name] = basis @ beta
        return out

    def curve(self, name: str, grid: np.ndarray) -> np.ndarray:
        """The fitted shape of one smooth term, for plotting and for the report."""
        block = next(b for b in self.blocks_ if b.name == name)
        beta = self.coef_[block.start: block.start + block.width]
        if block.linear:
            return ((np.asarray(grid) - self.centre_[name]) / self.scale_[name]) * beta[0]
        return spline_design(np.asarray(grid), block.knots) @ beta
