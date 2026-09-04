"""Generate the dataset one wafer at a time, and record the latent variables.

Two problems with running ``generate_data.py`` as supplied on a normal laptop:

* it holds every wafer's dataframe in memory and concatenates at the end, which
  for 160 training wafers is several gigabytes of block-reading strings; and
* the quantities that decide the answer -- each wafer's ``wafer_base_rate``,
  each die's ``new_fail_prob``, and which failures were drawn as *marginal* --
  are computed and thrown away, so there is no way to ask how well a model could
  possibly have done.

This module writes the same CSVs wafer by wafer, and alongside them a
``ground_truth.parquet`` holding those latents.

**The output is the same dataset.**  The random draws are taken in exactly the
order ``generate_data.generate_die_features`` takes them, so the stream from
``default_rng(42)`` is identical and so is every value.  ``tests`` checks a
wafer generated both ways for exact equality rather than taking that on trust.

The ground truth is used only by ``tuned.ceiling``, which measures the best
score any model could reach.  No model in this repository reads it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import generate_data as reference

CSV_NAMES = {"train": "train.csv", "test": "test.csv", "validation": "validation.csv"}


def die_features_with_truth(wafer_map, config, rng):
    """``generate_data.generate_die_features`` with the latents returned too.

    The body below is that function's, unchanged in every line that touches
    ``rng``.  Only the return statement differs.
    """
    feature_defs = config["features"]
    n_features = len(feature_defs)
    rows, cols = wafer_map.shape
    window_size = config["neighborhood_window"]
    radial_strength = config["radial_gradient_strength"]
    linear_strength = config["linear_gradient_strength"]
    neigh_influence = config["neighborhood_influence"]
    marginal_frac = config["marginal_fail_fraction"]
    new_fail_rate = config.get("new_fail_rate", 0.02)

    radial_map = reference.compute_radial_map(wafer_map)
    linear_map = reference.compute_linear_gradient(wafer_map, rng)
    fail_density = reference.compute_neighborhood_fail_density(wafer_map, window_size)

    old_label_map = (wafer_map == 2).astype(int)

    wafer_base_rate = rng.exponential(scale=new_fail_rate)
    wafer_base_rate = min(wafer_base_rate, 0.25)

    passing_mask = wafer_map == 1

    new_fail_prob = np.zeros((rows, cols), dtype=float)
    new_fail_prob[passing_mask] = wafer_base_rate
    new_fail_prob += fail_density * wafer_base_rate * 3.0
    new_fail_prob += radial_map * wafer_base_rate * 1.5
    new_fail_prob = np.clip(new_fail_prob, 0, 0.4)
    new_fail_prob[~passing_mask] = 0

    random_draw = rng.random((rows, cols))
    new_fail_mask = (random_draw < new_fail_prob) & passing_mask

    label_map = old_label_map | new_fail_mask.astype(int)

    fail_positions = np.argwhere(label_map == 1)
    n_fail = len(fail_positions)
    n_marginal = int(n_fail * marginal_frac)
    if n_fail > 0:
        marginal_indices = rng.choice(n_fail, size=n_marginal, replace=False)
        marginal_mask = np.zeros((rows, cols), dtype=bool)
        for idx in marginal_indices:
            marginal_mask[fail_positions[idx][0], fail_positions[idx][1]] = True
    else:
        marginal_mask = np.zeros((rows, cols), dtype=bool)

    features_grid = np.zeros((rows, cols, n_features))
    marginal_multiplier = np.zeros(n_features)

    for f_idx, fdef in enumerate(feature_defs):
        base_mean = fdef["base_mean"]
        base_std = fdef["base_std"]
        fail_shift = fdef["fail_shift"]

        base = rng.normal(base_mean, base_std, size=(rows, cols))
        radial_coeff = rng.uniform(-1, 1) * radial_strength * base_std
        linear_coeff = rng.uniform(-1, 1) * linear_strength * base_std
        base += radial_map * radial_coeff
        base += linear_map * linear_coeff
        base += fail_density * neigh_influence * fail_shift

        fail_mask = label_map == 1
        non_marginal_fail = fail_mask & (~marginal_mask)
        base[non_marginal_fail] += fail_shift
        multiplier = rng.uniform(0.05, 0.25)
        base[marginal_mask] += fail_shift * multiplier
        marginal_multiplier[f_idx] = multiplier

        features_grid[:, :, f_idx] = base

    truth = {
        "wafer_base_rate": float(wafer_base_rate),
        "radial_map": radial_map,
        "fail_density": fail_density,
        "new_fail_prob": new_fail_prob,
        "marginal_mask": marginal_mask,
        "marginal_multiplier": marginal_multiplier,
    }
    return features_grid, old_label_map, label_map, truth


def _wafer_rows(wafer_id, wafer_map, config, rng, feature_names):
    grid, old_label_map, label_map, truth = die_features_with_truth(
        wafer_map, config, rng)
    frame = reference.wafer_to_dataframe(
        wafer_id, wafer_map, grid, old_label_map, label_map, feature_names)
    frame["block_readings"] = reference.generate_block_readings(
        len(frame), frame["label"].values, config, rng)
    row = frame["die_row"].to_numpy()
    col = frame["die_col"].to_numpy()
    latent = pd.DataFrame(
        {
            "wafer_id": wafer_id,
            "die_row": row,
            "die_col": col,
            "wafer_base_rate": np.float32(truth["wafer_base_rate"]),
            "true_radius": truth["radial_map"][row, col].astype(np.float32),
            "true_density": truth["fail_density"][row, col].astype(np.float32),
            "true_fail_prob": truth["new_fail_prob"][row, col].astype(np.float32),
            "is_marginal": truth["marginal_mask"][row, col],
        }
    )
    return frame, latent, truth["marginal_multiplier"]


def run(config_path: Path, output_dir: Path, ground_truth: Path) -> None:
    config = reference.load_config(str(config_path))
    n_train = config["num_wafers_train"]
    n_test = config["num_wafers_test"]
    feature_names = [f["name"] for f in config["features"]]
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config["seed"])
    labelled, none_wafers = reference.load_wm811k(config["wm811k_path"])
    maps, ids = reference.select_wafers(
        labelled, none_wafers, n_train + n_test, config["target_fail_rate"], rng)

    splits = {"train": (maps[:n_train], ids[:n_train]),
              "test": (maps[n_train:], ids[n_train:])}
    paths = {name: output_dir / CSV_NAMES[name] for name in ("train", "test")}
    for path in list(paths.values()) + [output_dir / CSV_NAMES["validation"]]:
        path.unlink(missing_ok=True)

    latents: list[pd.DataFrame] = []
    multipliers: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for split, (split_maps, split_ids) in splits.items():
        for number, (wafer_map, wafer_id) in enumerate(zip(split_maps, split_ids), 1):
            frame, latent, multiplier = _wafer_rows(
                wafer_id, wafer_map, config, rng, feature_names)
            header = not paths[split].exists()
            frame.to_csv(paths[split], mode="a", header=header, index=False)
            if split == "test":
                validation = frame.drop(columns=["label"])
                validation.to_csv(output_dir / CSV_NAMES["validation"], mode="a",
                                  header=header, index=False)
            latents.append(latent)
            multipliers[wafer_id] = multiplier
            if number % 20 == 0:
                print(f"  {split} {number}/{len(split_ids)} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)

    ground_truth.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(latents, ignore_index=True).to_parquet(ground_truth, index=False)
    np.savez_compressed(
        ground_truth.with_suffix(".multipliers.npz"), **multipliers)
    print(f"Wrote {paths['train']}, {paths['test']}, "
          f"{output_dir / CSV_NAMES['validation']} and {ground_truth} "
          f"in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("input"))
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("input/ground_truth.parquet"))
    args = parser.parse_args()
    run(args.config, args.output_dir, args.ground_truth)


if __name__ == "__main__":
    main()
