"""Model factories.

The generator adds a per-feature constant to failing dies, so the true decision
function is linear in standardised measurements: a penalised logistic regression
is the matched model here, not an approximation to it.  Gradient boosting is
kept for the ablation because it is the obvious thing to reach for, and it is
worth showing on the record that it does worse on this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(backend: str, random_state: int = 42, C: float = 0.02):
    if backend == "linear":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C, solver="lbfgs", max_iter=3_000, random_state=random_state
                    ),
                ),
            ]
        )
    if backend == "stacked":
        return Stacked(C=C, random_state=random_state)
    if backend == "boost":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=random_state,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=1_500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown backend: {backend}")


def positive_weight(y, mode: str = "sqrt") -> float:
    """Up-weight the minority class without letting it dominate the fit.

    ``balanced`` (negatives/positives, about 23 here) drags the boundary well
    past the point that maximises fail-class F1.  ``sqrt`` keeps the ranking
    sharp and leaves the operating point to explicit threshold selection.
    """
    y = np.asarray(y)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        raise ValueError("No failures present")
    ratio = negatives / positives
    return {"none": 1.0, "sqrt": float(np.sqrt(ratio)), "balanced": float(ratio)}[mode]


def fit(model, x, y, weight: float):
    weights = np.where(np.asarray(y) == 1, weight, 1.0).astype(np.float64)
    if isinstance(model, Pipeline):
        model.fit(x, y, classifier__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


class DiagonalScore(BaseEstimator, TransformerMixin):
    """Collapse the parametric block to one likelihood-ratio score.

    Conditional on the label, the generator draws each measurement independently,
    so the optimal combination is the diagonal (naive-Bayes) discriminant

        score_i = sum_f (mean_fail_f - mean_pass_f) / var_f * x_if

    which needs one mean difference per feature instead of a 500 x 500 covariance.
    With only ~6,500 failures in the training set that difference in statistical
    efficiency is worth more than any amount of tuning on the joint fit, and each
    coefficient stays readable as the measured pass/fail separation of one
    parameter.
    """

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(bool)
        centre = np.nanmean(X, axis=0)
        X = np.where(np.isfinite(X), X, centre)
        variance = np.nanvar(X, axis=0)
        variance = np.where(variance > 1e-12, variance, 1.0)
        self.weights_ = (X[y].mean(axis=0) - X[~y].mean(axis=0)) / variance
        self.centre_ = centre
        self.sd_ = np.sqrt(variance)
        # Cohen's d per parameter: the readable form of the same quantity.
        self.separation_ = (X[y].mean(axis=0) - X[~y].mean(axis=0)) / self.sd_
        scores = (X - centre) @ self.weights_
        self.scale_ = float(scores.std()) or 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, self.centre_)
        return (((X - self.centre_) @ self.weights_) / self.scale_)[:, None]


class Stacked(BaseEstimator, ClassifierMixin):
    """Diagonal score over the parametric block, logistic over everything else.

    Reducing 500 correlated-in-estimation coefficients to a single well-measured
    score leaves the logistic stage with a few dozen parameters, which is the
    number the 6,519 available failures can actually support.
    """

    def __init__(self, parametric_prefix=("dz_", "feature_"), C=1.0, random_state=42):
        self.parametric_prefix = parametric_prefix
        self.C = C
        self.random_state = random_state

    def _split(self, X):
        columns = list(X.columns)
        wide = [c for c in columns if c.startswith(tuple(self.parametric_prefix))]
        rest = [c for c in columns if c not in set(wide)]
        return wide, rest

    def fit(self, X, y, sample_weight=None):
        wide, rest = self._split(X)
        self.wide_, self.rest_ = wide, rest
        self.score_ = DiagonalScore().fit(X.loc[:, wide].to_numpy(), y)
        self.head_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=self.C, solver="lbfgs", max_iter=3_000,
                    random_state=self.random_state)),
            ]
        )
        self.head_.fit(self._design(X), y, classifier__sample_weight=sample_weight)
        self.classes_ = self.head_.classes_
        return self

    def _design(self, X):
        parametric = self.score_.transform(X.loc[:, self.wide_].to_numpy())
        frame = pd.DataFrame(parametric, columns=["parametric_score"], index=X.index)
        if self.rest_:
            frame = pd.concat([frame, X.loc[:, self.rest_]], axis=1)
        return frame

    def predict_proba(self, X):
        return self.head_.predict_proba(self._design(X))

    def predict(self, X):
        return self.head_.predict(self._design(X))
