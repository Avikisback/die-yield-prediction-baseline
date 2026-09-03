"""Attribution and visual diagnostics.

The problem statement asks for actionable root cause rather than a score, so
every predicted failure has to come with the measurements, the neighbourhood and
the block pattern that produced it.

Both models are linear in standardised inputs, which means an attribution is not
an approximation of the model -- it *is* the model.  For a die with standardised
inputs ``z`` the log-odds decompose exactly:

    logit(p) = intercept + sum_j coefficient_j * z_j

so the terms can be added up per resolution (parametric / spatial / block) and
read off per feature with no sampling, no surrogate and no error bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

GROUPS = {
    "parametric": ("dz_", "dstat_", "feature_"),
    "spatial": ("spatial_",),
    "block": ("block_",),
}


def _parts(model):
    if not isinstance(model, Pipeline):
        raise TypeError("Attribution requires a linear pipeline")
    return (model.named_steps["imputer"], model.named_steps["scale"],
            model.named_steps["classifier"])


def standardise(model, x: pd.DataFrame) -> np.ndarray:
    imputer, scaler, _ = _parts(model)
    return scaler.transform(imputer.transform(x))


def _flat_contributions(model, x):
    _, _, classifier = _parts(model)
    return (standardise(model, x) * classifier.coef_[0][None, :],
            float(classifier.intercept_[0]))


def _stacked_contributions(model, x):
    """Unfold the two-stage model back to one term per original column.

    The head is linear in the parametric score and the parametric score is
    linear in the measurements, so the composition is still exactly additive:

        contribution_f = head_coef_0 * w_f * (x_f - centre_f)
                         / (score_scale * head_scale_0)
    """
    design = model._design(x)
    _, scaler, classifier = _parts(model.head_)
    standardised = standardise(model.head_, design)
    head = standardised * classifier.coef_[0][None, :]

    values = np.zeros((len(x), len(x.columns)), dtype=np.float64)
    index = {c: i for i, c in enumerate(x.columns)}
    score = model.score_
    gain = classifier.coef_[0][0] / (score.scale_ * scaler.scale_[0])
    wide = x.loc[:, model.wide_].to_numpy(dtype=np.float64)
    wide = np.where(np.isfinite(wide), wide, score.centre_)
    contribution = (wide - score.centre_) * score.weights_ * gain
    for position, name in enumerate(model.wide_):
        values[:, index[name]] = contribution[:, position]
    for position, name in enumerate(model.rest_, start=1):
        values[:, index[name]] = head[:, position]

    # Whatever the parametric centring leaves behind is a constant, so it
    # belongs with the intercept rather than with any one measurement.
    intercept = float(classifier.intercept_[0]) - float(
        classifier.coef_[0][0] * scaler.mean_[0] / scaler.scale_[0])
    return values, intercept


def contributions(model, x: pd.DataFrame):
    """Exact per-feature log-odds contributions for every row of ``x``."""
    if hasattr(model, "score_") and hasattr(model, "head_"):
        return _stacked_contributions(model, x)
    return _flat_contributions(model, x)


def group_contributions(model, x: pd.DataFrame) -> pd.DataFrame:
    """Collapse the contributions to one column per resolution."""
    values, intercept = contributions(model, x)
    columns = list(x.columns)
    out = {}
    for name, prefixes in GROUPS.items():
        index = [i for i, c in enumerate(columns) if c.startswith(prefixes)]
        out[f"contrib_{name}"] = values[:, index].sum(axis=1) if index else 0.0
    frame = pd.DataFrame(out, index=x.index)
    frame["contrib_intercept"] = intercept
    frame["logit"] = frame.sum(axis=1)
    return frame


def top_drivers(model, x: pd.DataFrame, rows, k: int = 8) -> pd.DataFrame:
    """The ``k`` measurements that moved each selected die the most."""
    subset = x.iloc[rows]
    values, _ = contributions(model, subset)
    columns = np.array(x.columns)
    records = []
    for position, row in enumerate(rows):
        order = np.argsort(-np.abs(values[position]))[:k]
        for rank, j in enumerate(order, start=1):
            records.append(
                {
                    "row": int(row),
                    "rank": rank,
                    "feature": columns[j],
                    "contribution": float(values[position, j]),
                    "value": float(subset.iloc[position, j]),
                }
            )
    return pd.DataFrame(records)


def global_importance(model, columns) -> pd.DataFrame:
    """Effect of a one-standard-deviation move in each input, on the log-odds."""
    if hasattr(model, "score_") and hasattr(model, "head_"):
        _, scaler, classifier = _parts(model.head_)
        score = model.score_
        gain = classifier.coef_[0][0] / (score.scale_ * scaler.scale_[0])
        lookup = dict(zip(model.wide_, score.weights_ * score.sd_ * gain))
        lookup.update(zip(model.rest_, classifier.coef_[0][1:]))
        coefficient = np.array([lookup.get(c, 0.0) for c in columns])
    else:
        _, _, classifier = _parts(model)
        coefficient = classifier.coef_[0]
    resolution = []
    for name in columns:
        for group, prefixes in GROUPS.items():
            if name.startswith(prefixes):
                resolution.append(group)
                break
        else:
            resolution.append("other")
    return pd.DataFrame(
        {
            "feature": list(columns),
            "resolution": resolution,
            "coefficient": coefficient,
            "abs_coefficient": np.abs(coefficient),
        }
    ).sort_values("abs_coefficient", ascending=False)


def wafer_layer(frame: pd.DataFrame, wafer_id: str, column: str) -> np.ndarray:
    """Lay one column back out on the wafer grid, with gaps as NaN."""
    here = frame.loc[frame["wafer_id"].astype(str) == str(wafer_id)]
    rows = here["die_row"].to_numpy(dtype=int)
    cols = here["die_col"].to_numpy(dtype=int)
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    grid[rows, cols] = here[column].to_numpy(dtype=float)
    return grid


def true_shift_directions(config_path: str = "config.yaml") -> pd.DataFrame:
    """The generator's own fail_shift per feature, in units of base_std.

    Used only to check what the fitted model recovered.  It is never an input to
    any model.
    """
    import yaml

    from generate_data import generate_feature_definitions

    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    definitions = generate_feature_definitions(config)
    return pd.DataFrame(
        {
            "feature": [d["name"] for d in definitions],
            "true_shift_in_sd": [d["fail_shift"] / d["base_std"] for d in definitions],
        }
    )


def find_block_readings(csv_path, keys, chunksize: int = 4000) -> dict:
    """Pull the raw 2,000-reading strings for a few named dies."""
    wanted = {(str(w), int(r), int(c)) for w, r, c in keys}
    found = {}
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=["wafer_id", "die_row", "die_col", "block_readings"],
    ):
        for row in chunk.itertuples(index=False):
            key = (str(row.wafer_id), int(row.die_row), int(row.die_col))
            if key in wanted and key not in found:
                found[key] = np.fromstring(row.block_readings, sep=" ", dtype=np.float64)
        if len(found) == len(wanted):
            break
    return found
