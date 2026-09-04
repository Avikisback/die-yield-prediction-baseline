"""Collapse each resolution to the one number that carries its evidence.

For the 500 parametric measurements that number is exact rather than merely
convenient.  The generator draws every measurement independently given the label
and shifts the failing ones by a per-feature constant, so the family is a
location family along a single known direction and the projection onto it is a
*sufficient* statistic: no other function of the 500 values adds anything.  That
the failing class is a mixture -- 65% of failures get only 5-25% of the shift --
does not change that, because the mixture lies along the same direction.  What
the sufficient statistic's log-likelihood ratio then looks like is the head's
problem, not this module's, and it is the reason the head has to be nonlinear.

The projection is the diagonal (naive-Bayes) discriminant

    w_f = (mean_fail_f - mean_pass_f) / var_f

which needs one mean difference per feature.  ``mrf.models.DiagonalScore``
already does that; what is added here is that the direction may be estimated
from the pre-test failures as well.  Dies with ``old_label == 1`` carry the same
``fail_shift`` under the same 65% marginal rule, and ``old_label`` is available
before the test, so they are extra labelled positives that cost nothing and
roughly double the sample the direction is measured from.

Every pass over the measurements is chunked and row-masked rather than sliced.
The matrix is 190,000 x 500 float32 -- 380 MB -- and fifteen folds that each
take a copy will not fit on a 16 GB laptop beside everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_VARIANCE = 1e-12
CHUNK = 20_000


def _moments(x: np.ndarray, mask: np.ndarray, chunk: int = CHUNK):
    """Count, mean and variance over the rows ``mask`` selects, in float64."""
    total = np.zeros(x.shape[1], dtype=np.float64)
    square = np.zeros(x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        block = x[start:stop][mask[start:stop]]
        if not len(block):
            continue
        block = np.asarray(block, dtype=np.float64)
        total += block.sum(axis=0)
        square += (block ** 2).sum(axis=0)
        count += len(block)
    if count == 0:
        raise ValueError("No rows selected")
    mean = total / count
    return count, mean, np.maximum(square / count - mean ** 2, 0.0)


def project(x: np.ndarray, centre: np.ndarray, weights: np.ndarray,
            chunk: int = CHUNK) -> np.ndarray:
    """``(x - centre) @ weights`` for every row, without copying ``x``."""
    out = np.empty(len(x), dtype=np.float64)
    shift = float(centre @ weights)
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        out[start:stop] = np.asarray(x[start:stop], dtype=np.float64) @ weights
    return out - shift


@dataclass
class DiagonalScore:
    """Sufficient statistic for a shift along a single estimated direction."""

    weights: np.ndarray = field(repr=False)
    centre: np.ndarray = field(repr=False)
    scale: float
    separation: np.ndarray = field(repr=False)
    n_positive: int
    n_negative: int

    @classmethod
    def fit(cls, x: np.ndarray, positive: np.ndarray,
            negative: np.ndarray) -> "DiagonalScore":
        """Fit from two disjoint row masks over the full measurement matrix."""
        if positive.sum() == 0 or negative.sum() == 0:
            raise ValueError("Both classes must be present")
        n_pos, mean_pos, _ = _moments(x, positive)
        n_neg, mean_neg, var_neg = _moments(x, negative)
        centre = (n_pos * mean_pos + n_neg * mean_neg) / (n_pos + n_neg)
        variance = np.where(var_neg > MIN_VARIANCE, var_neg, 1.0)
        difference = mean_pos - mean_neg
        weights = difference / variance
        raw = project(x, centre, weights)
        scale = float(raw[negative].std()) or 1.0
        return cls(weights=weights, centre=centre, scale=scale,
                   separation=difference / np.sqrt(variance),
                   n_positive=int(n_pos), n_negative=int(n_neg))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return project(x, self.centre, self.weights) / self.scale


def fit_channel(x: np.ndarray, label: np.ndarray, old_label: np.ndarray,
                rows: np.ndarray, use_old_fails: bool = True) -> DiagonalScore:
    """Fit the parametric direction on the rows ``rows`` selects.

    Positives are post-test failures among eligible dies, plus -- when asked for
    -- every ``old_label == 1`` die.  Negatives are the eligible dies that
    passed.
    """
    label = np.asarray(label).astype(bool)
    old_label = np.asarray(old_label).astype(bool)
    rows = np.asarray(rows).astype(bool)
    positive = label & ~old_label
    if use_old_fails:
        positive = positive | old_label
    negative = (~label) & (~old_label)
    return DiagonalScore.fit(x, positive & rows, negative & rows)
