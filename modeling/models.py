"""Model factories with a LightGBM-first and scikit-learn fallback strategy."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    backend: str
    positive_weight: float
    random_state: int


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return (
            "lightgbm"
            if importlib.util.find_spec("lightgbm") is not None
            else "hist_gradient_boosting"
        )
    if requested == "lightgbm" and importlib.util.find_spec("lightgbm") is None:
        raise RuntimeError(
            "LightGBM was requested but is not installed. Install it or use "
            "--backend hist_gradient_boosting."
        )
    return requested


def make_model(config: ModelConfig):
    backend = resolve_backend(config.backend)
    if backend == "linear":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1_000,
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if backend == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=config.random_state,
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
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown backend: {backend}")


def positive_sample_weights(y: np.ndarray, positive_weight: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    return np.where(y == 1, positive_weight, 1.0).astype(np.float32)


def fit_model(model, x, y, positive_weight: float):
    weights = positive_sample_weights(y, positive_weight)
    if isinstance(model, Pipeline):
        model.fit(x, y, classifier__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model

