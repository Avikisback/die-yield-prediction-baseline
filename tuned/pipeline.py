"""The fitted model: prior, two evidence channels, and the wafer-rate loop.

    logit P(fail_i) = logit(pi_i)  +  f_param(s_i)  +  f_block(b_i)
    pi_i            = min(rate_w * h_i, 0.4)

``h_i`` is the pre-test hazard shape, ``rate_w`` the wafer's own base rate, and
the two ``f`` are smooth functions of one sufficient statistic each.  Every term
is estimated: the direction behind ``s`` and the weights behind ``b`` from the
training labels, the shapes of the two ``f`` from the same, and ``rate_w`` from
the wafer's own dies with no labels at all.

The one structural liberty is that ``h_i`` is allowed a correction.  The
generator's bracket ``1 + 3*density + 1.5*radius`` is reconstructed from the die
list rather than from the wafer-map array, so it can be slightly off; a smooth
term in ``log h`` plus a few spatial covariates absorbs the difference.  The
correction belongs to the *prior*: it multiplies ``h`` before the rate is
recovered, instead of being mistaken for something a die's own measurements
said.  Getting that split wrong double-counts the neighbourhood, because the
generator also leaks the local defect density into every parametric measurement.

The head is fitted **once**, against the population rate, and the per-wafer rate
is applied only when scoring.  That is not a shortcut: the evidence functions
describe what a die's own measurements say, which does not depend on the prior
they are later added to.  Re-fitting the head against offsets derived from each
wafer's own unlabelled rate estimate lets the wafer level feed back into the head
that produced it, and the ablations measure the cost -- a second pass is worse on
12 of 15 folds, a third is worse on 15 of 15.

Interfaces take the 500 measurements as a bare float32 matrix plus a row mask,
never a sliced copy: fifteen folds of 190,000 x 500 copies do not fit in memory
beside everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tuned import channels, head as head_module, waferrate

BLOCK_PREFIX = "block_"
HAZARD_SHAPE = "haz_shape"
LOG_SHAPE = "haz_log_shape"
GENERATOR_DENSITY = "haz_density_w5"
PRIOR_LINEAR = (
    "haz_density_w3",
    "haz_density_w7",
    "haz_density_w11",
    "haz_radius",
    "haz_edge_distance",
    "haz_nearest_old_fail",
    "haz_wafer_old_fail_rate",
)
CAP = 0.4
# A coarse rate grid is enough while searching for the evidence level: the
# search is over a smooth scalar and the fine grid is only needed once, when the
# probabilities themselves are computed.
PIN_GRID = np.exp(np.linspace(np.log(2e-4), np.log(0.25), 96))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


@dataclass
class Fusion:
    """Fit and score the additive posterior."""

    use_block: bool = True
    use_parametric: bool = True
    use_old_fails: bool = True
    subtract_density: bool = True
    evidence_level: str = "match_count"
    use_rate: bool = True
    use_hazard: bool = True
    correct_prior: bool = True
    smooth_evidence: bool = True
    n_bases: int = 8
    lam_smooth: float = 20.0
    lam_linear: float = 10.0
    block_C: float = 0.05
    rounds: int = 1
    random_state: int = 42
    diagonal_: channels.DiagonalScore | None = field(default=None, repr=False)
    block_model_: Pipeline | None = field(default=None, repr=False)
    head_: head_module.AdditiveHead | None = field(default=None, repr=False)

    # ----------------------------------------------------------------- pieces
    @staticmethod
    def block_columns(frame) -> list[str]:
        return sorted(c for c in frame.columns if c.startswith(BLOCK_PREFIX))

    def _design(self, frame: pd.DataFrame, x: np.ndarray,
                rows: np.ndarray) -> pd.DataFrame:
        """The head's inputs for the rows ``rows`` selects."""
        index = np.flatnonzero(rows)
        out = pd.DataFrame(index=index)
        if self.use_parametric:
            score = self.diagonal_.transform(x)[rows]
            if self.density_slope_:
                # The generator adds ``fail_density * 0.2 * fail_shift`` to every
                # measurement, failing or not, so the diagonal score carries a
                # neighbourhood offset that is not evidence about this die.  The
                # prior already counts the neighbourhood; leaving the offset in
                # counts it a second time, and an additive head cannot subtract
                # it because the true form is f(s - c*density), not f(s) - g(density).
                score = score - self.density_slope_ * frame.loc[
                    rows, GENERATOR_DENSITY].to_numpy(dtype=np.float64)
            out["parametric_score"] = score
        if self.use_block:
            block = frame.loc[rows, self.block_names_].replace(
                [np.inf, -np.inf], np.nan)
            out["block_score"] = _logit(self.block_model_.predict_proba(block)[:, 1])
        out[LOG_SHAPE] = frame.loc[rows, LOG_SHAPE].to_numpy(dtype=np.float64)
        for name in self.prior_linear_:
            out[name] = frame.loc[rows, name].to_numpy(dtype=np.float64)
        return out

    def _shape(self, frame, rows) -> np.ndarray:
        """The pre-test hazard shape, or a flat one for the single-channel rows.

        Turning it off is what makes ``block_only`` and ``parametric_only``
        comparable to the same rows in ``mrf``: those fit a classifier on one
        block of features and nothing else, with no spatial prior underneath.
        """
        if not self.use_hazard:
            return np.ones(int(np.sum(rows)), dtype=np.float64)
        return frame.loc[rows, HAZARD_SHAPE].to_numpy(dtype=np.float64)

    def _prior_terms(self) -> list[str]:
        return [LOG_SHAPE, *self.prior_linear_] if self.correct_prior else []

    def _split(self, design) -> tuple[np.ndarray, np.ndarray]:
        """Split the head's log-odds into (prior correction, evidence).

        The two sum back to ``head.evidence`` exactly, so nothing is quietly
        dropped.  Which side the *constant* falls on is decided by
        ``_pin_level``; it changes no fitted probability but it changes every
        recovered wafer rate.
        """
        parts = self.head_.partial(design)
        zero = np.zeros(len(design))
        evidence = sum((parts[name] for name in self.evidence_terms_), zero)
        correction = self.head_.intercept_ + sum(
            (parts[name] for name in self._prior_terms()), zero)
        return correction + self.evidence_offset_, evidence - self.evidence_offset_

    def _pin_level(self, raw, correction, shape, wafer, target) -> float:
        """Where the constant the head cannot allocate on its own should sit.

        The head sees ``logit(prior) + intercept + prior terms + evidence terms``
        and only the total is identified: a constant can move between the prior
        side and the evidence side without changing a single fitted probability.
        The rate step is not indifferent to it, because it reads the evidence as
        a likelihood ratio.  So something outside the head has to decide.

        ``match_count`` -- the default -- bisects until the posterior predicts
        the number of failures the training wafers actually had.  That is one
        aggregate over 150,000 dies, and it is the identity that has to hold:
        a model whose probabilities do not sum to the failures it has seen is
        not reporting probabilities.

        ``none`` leaves the intercept on the prior side.  It ranks *better* --
        by 0.0012 average precision, on 13 of 15 folds -- and it is not shipped,
        because of what the ranking is bought with: its probabilities sum to 46%
        of the actual failures, and the per-wafer rates it recovers correlate
        with the drawn ones at 0.70 instead of 0.93.  Leaving the intercept out
        of the evidence quietly shrinks the wafer-rate term, which is the same
        trade ``mrf.calibrate`` makes deliberately with ``alpha = 0.5``.  A
        shrunk wafer effect does rank slightly better here; it is still an
        accident of where a constant landed rather than a decision, and it costs
        the one claim this model makes that a fab would act on.

        ``mean_exp`` uses the textbook identity -- a likelihood ratio averages
        to one over the passing class -- and is the worst of the three, because
        ``exp(evidence)`` reaches ``e^16`` on the clearest failures and the
        sample mean is then decided by a handful of them.
        """
        if self.evidence_level == "none":
            return 0.0
        if self.evidence_level == "mean_exp":
            passing = target == 0
            return float(np.log(np.mean(np.exp(np.clip(raw[passing], -30, 30)))))

        observed = float(target.sum())

        def predicted(level: float) -> float:
            probability, _ = waferrate.posterior_probabilities(
                raw - level, shape * np.exp(correction + level), wafer,
                self.prior_scale_, grid=PIN_GRID, cap=CAP)
            return float(probability.sum())

        low, high = -6.0, 6.0
        if predicted(low) < observed:
            return low
        if predicted(high) > observed:
            return high
        for _ in range(10):
            middle = 0.5 * (low + high)
            if predicted(middle) > observed:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    # -------------------------------------------------------------------- fit
    def fit(self, frame: pd.DataFrame, x: np.ndarray, rows: np.ndarray) -> "Fusion":
        rows = np.asarray(rows).astype(bool)
        label = frame["label"].to_numpy(dtype=np.int8)
        old_label = frame["old_label"].to_numpy(dtype=np.int8)
        wafer = frame["wafer_id"].astype(str).to_numpy()

        self.block_names_ = self.block_columns(frame) if self.use_block else []
        self.prior_linear_ = tuple(n for n in PRIOR_LINEAR if n in frame.columns)
        self.evidence_terms_ = tuple(
            name for name, enabled in (("parametric_score", self.use_parametric),
                                       ("block_score", self.use_block)) if enabled)

        fit_rows = rows & (old_label == 0)
        self.density_slope_ = 0.0
        if self.use_parametric:
            self.diagonal_ = channels.fit_channel(
                x, label, old_label, rows, use_old_fails=self.use_old_fails)
            if self.subtract_density and GENERATOR_DENSITY in frame.columns:
                # Fitted on passing dies only, so the failures' own shift cannot
                # drag the slope and take real signal with it.
                clean = fit_rows & (label == 0)
                density = frame.loc[clean, GENERATOR_DENSITY].to_numpy(
                    dtype=np.float64)
                raw = self.diagonal_.transform(x)[clean]
                spread = float(density.var())
                if spread > 1e-12:
                    self.density_slope_ = float(
                        np.cov(density, raw, bias=True)[0, 1] / spread)
        if self.use_block:
            # ``generate_block_readings`` injects the anomaly signature into every
            # die whose final label is 1, pre-test failures included, so those
            # dies are extra labelled positives for this channel exactly as they
            # are for the parametric one.  There are more of them than there are
            # new failures.
            block_rows = fit_rows
            if self.use_old_fails:
                block_rows = fit_rows | (rows & (old_label == 1))
            block = frame.loc[block_rows, self.block_names_].replace(
                [np.inf, -np.inf], np.nan)
            self.block_model_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("classifier", LogisticRegression(
                        C=self.block_C, solver="lbfgs", max_iter=3_000,
                        random_state=self.random_state)),
                ]
            ).fit(block, label[block_rows])

        design = self._design(frame, x, fit_rows)
        target = label[fit_rows].astype(np.float64)
        wafer_fit = wafer[fit_rows]
        shape_fit = self._shape(frame, fit_rows)

        self.prior_scale_ = waferrate.fit_prior_scale(
            waferrate.observed_rates(shape_fit, target, wafer_fit))
        self.overall_rate_ = float(target.sum() / shape_fit.sum())
        rates = {str(key): self.overall_rate_ for key in np.unique(wafer_fit)}
        self.evidence_offset_ = 0.0

        smooth = self.evidence_terms_ if self.smooth_evidence else ()
        linear = () if self.smooth_evidence else self.evidence_terms_
        if self.correct_prior:
            smooth = smooth + (LOG_SHAPE,)
            linear = linear + self.prior_linear_

        for _ in range(max(self.rounds, 1)):
            offset = waferrate.offsets(shape_fit, wafer_fit, rates, cap=CAP,
                                       fallback=self.overall_rate_)
            self.head_ = head_module.AdditiveHead(
                smooth=smooth, linear=linear, n_bases=self.n_bases,
                lam_smooth=self.lam_smooth, lam_linear=self.lam_linear,
            ).fit(design, target, offset)
            parts = self.head_.partial(design)
            raw = sum((parts[name] for name in self.evidence_terms_),
                      np.zeros(len(design)))
            correction = self.head_.intercept_ + sum(
                (parts[name] for name in self._prior_terms()),
                np.zeros(len(design)))
            self.evidence_offset_ = self._pin_level(
                raw, correction, shape_fit, wafer_fit, target)
            if not self.use_rate:
                break
            correction, evidence = self._split(design)
            _, rates = waferrate.posterior_probabilities(
                evidence, shape_fit * np.exp(correction), wafer_fit,
                self.prior_scale_, cap=CAP)

        self.fitted_rates_ = rates
        return self

    # -------------------------------------------------------------- inference
    def score_frame(self, frame: pd.DataFrame, x: np.ndarray,
                    rows: np.ndarray) -> pd.DataFrame:
        """Per-die channel scores, evidence log-odds and corrected hazard."""
        rows = np.asarray(rows).astype(bool)
        design = self._design(frame, x, rows)
        correction, evidence = self._split(design)
        out = pd.DataFrame(index=design.index)
        for name in self.evidence_terms_:
            out[name] = design[name].to_numpy()
        out["evidence"] = evidence
        out["hazard"] = self._shape(frame, rows) * np.exp(correction)
        return out

    def predict_proba(self, frame: pd.DataFrame, x: np.ndarray,
                      rows: np.ndarray) -> np.ndarray:
        scored = self.score_frame(frame, x, rows)
        wafer = frame.loc[rows, "wafer_id"].astype(str).to_numpy()
        if self.use_rate:
            probability, self.last_rates_ = waferrate.posterior_probabilities(
                scored["evidence"].to_numpy(), scored["hazard"].to_numpy(),
                wafer, self.prior_scale_, cap=CAP)
            return probability
        prior = np.clip(self.overall_rate_ * scored["hazard"].to_numpy(), 1e-9, CAP)
        self.last_rates_ = {str(key): self.overall_rate_ for key in np.unique(wafer)}
        return 1.0 / (1.0 + np.exp(
            -(np.log(prior / (1 - prior)) + scored["evidence"].to_numpy())))
