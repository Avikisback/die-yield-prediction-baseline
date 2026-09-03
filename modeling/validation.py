"""Grouped validation, threshold selection, and hackathon metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray


def eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the scored population: dies that passed the pre-test."""
    if "old_label" not in frame:
        raise ValueError("old_label is required to identify eligible dies")
    return frame.loc[frame["old_label"].eq(0)].copy()


def repeated_stratified_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    repeats: int = 3,
    random_state: int = 42,
) -> list[Fold]:
    """Build reproducible grouped folds and assert there is no wafer overlap."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Number of unique groups must be at least n_splits")

    folds: list[Fold] = []
    placeholder = np.zeros((len(y), 1), dtype=np.float32)
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold_number, (train_index, validation_index) in enumerate(
            splitter.split(placeholder, y, groups)
        ):
            train_groups = set(groups[train_index])
            validation_groups = set(groups[validation_index])
            if train_groups.intersection(validation_groups):
                raise AssertionError("A wafer appeared in both sides of a fold")
            folds.append(
                Fold(repeat, fold_number, train_index, validation_index)
            )
    return folds


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select the fail-class F1 maximizing threshold from validation predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.nanargmax(f1))])


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate eligible-die metrics using failure as the positive class."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "pass_precision": float(precision[0]),
        "pass_recall": float(recall[0]),
        "pass_f1": float(f1[0]),
        "fail_precision": float(precision[1]),
        "fail_recall": float(recall[1]),
        "fail_f1": float(f1[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics

