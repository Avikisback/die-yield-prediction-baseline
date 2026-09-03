"""
==============================================================
Synthetic Die-Level Feature Generator for Hackathon
==============================================================
Generates realistic die-level parametric features on top of
WM-811K wafer maps for multi-resolution die yield prediction.

Usage:
    python generate_data.py                        # uses defaults from config.yaml
    python generate_data.py --num_wafers 500       # override total wafers
    python generate_data.py --csv                  # also export CSV

Prerequisites:
    Download WM-811K dataset (LSWMD.pkl) from Kaggle:
    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    Place it at: data/LSWMD.pkl (or update config.yaml)
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-generate feature definitions from compact config
    config["features"] = generate_feature_definitions(config)
    return config


def generate_feature_definitions(config):
    """
    Auto-generate feature definitions (name, base_mean, base_std, fail_shift)
    deterministically from config parameters. Supports scaling to 1000+ features.
    """
    n = config["num_features"]
    fg = config["feature_generation"]
    # Use a dedicated RNG so feature defs are always the same regardless of data seed
    feat_rng = np.random.default_rng(config["seed"] + 1000)

    mean_range = fg["base_mean_range"]
    cv_range = fg["cv_range"]
    shift_frac_range = fg["fail_shift_fraction_range"]
    log_scale = fg.get("log_scale_mean", True)

    features = []
    for i in range(n):
        # Sample base_mean
        if log_scale:
            log_low = np.log10(max(mean_range[0], 1e-6))
            log_high = np.log10(mean_range[1])
            base_mean = 10 ** feat_rng.uniform(log_low, log_high)
        else:
            base_mean = feat_rng.uniform(mean_range[0], mean_range[1])

        # Randomly make some features negative (e.g. threshold voltages)
        if feat_rng.random() < 0.2:
            base_mean = -base_mean

        # Sample CV → compute std
        cv = feat_rng.uniform(cv_range[0], cv_range[1])
        base_std = abs(base_mean) * cv

        # Sample fail_shift as fraction of std, with random sign
        shift_frac = feat_rng.uniform(shift_frac_range[0], shift_frac_range[1])
        sign = feat_rng.choice([-1, 1])
        fail_shift = sign * shift_frac * base_std

        features.append({
            "name": f"feature_{i + 1}",
            "base_mean": float(base_mean),
            "base_std": float(base_std),
            "fail_shift": float(fail_shift),
        })

    return features


def load_wm811k(pkl_path):
    """
    Load WM-811K dataset and return list of labeled wafer maps.
    Each wafer map is a 2D numpy array: 0=no die, 1=pass, 2=fail.
    """
    if not os.path.exists(pkl_path):
        print(f"ERROR: WM-811K file not found at '{pkl_path}'")
        print("Please download LSWMD.pkl from:")
        print("  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map")
        print(f"And place it at: {pkl_path}")
        sys.exit(1)

    print(f"Loading WM-811K from {pkl_path}...")

    try:
        df = pd.read_pickle(pkl_path)
    except (ModuleNotFoundError, UnicodeDecodeError):
        # WM-811K pickle was saved with old pandas that used pandas.indexes.* modules
        # and Python 2 byte strings. Shim removed modules and use latin1 encoding.
        import types
        import pandas.core.indexes.base
        import pandas.core.indexes.range

        indexes_pkg = types.ModuleType('pandas.indexes')
        indexes_pkg.__path__ = []  # make it a package
        sys.modules['pandas.indexes'] = indexes_pkg
        sys.modules['pandas.indexes.base'] = pandas.core.indexes.base
        sys.modules['pandas.indexes.range'] = pandas.core.indexes.range

        # pandas.core.indexes.numeric was removed in pandas 2.0+
        try:
            import pandas.core.indexes.numeric
            sys.modules['pandas.indexes.numeric'] = pandas.core.indexes.numeric
        except ModuleNotFoundError:
            sys.modules['pandas.indexes.numeric'] = pandas.core.indexes.base
            sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base

        with open(pkl_path, 'rb') as f:
            df = pickle.load(f, encoding='latin1')

    # Filter to only labeled wafers with failures present
    labeled = df[df["failureType"].apply(
        lambda x: isinstance(x, np.ndarray) and len(x) > 0 and x[0][0] != "none" and x[0][0] != ""
    )].copy()

    # Also include some "none" (all-pass) wafers to maintain low fail rate
    none_wafers = df[df["failureType"].apply(
        lambda x: isinstance(x, np.ndarray) and len(x) > 0 and x[0][0] == "none"
    )].copy()

    print(f"  Found {len(labeled)} labeled-failure wafers, {len(none_wafers)} none-type wafers")

    return labeled, none_wafers


def select_wafers(labeled_df, none_df, num_wafers, target_fail_rate, rng):
    """
    Select a mix of labeled-failure and none wafers to approximate target fail rate.
    Uses target_fail_rate to determine the proportion of failure-pattern wafers.
    Returns list of 2D wafer map arrays.
    """
    # Failure-pattern wafers typically have ~10-20% failing dies internally.
    # To achieve an overall target_fail_rate (e.g. 3%), we need:
    #   fraction_failure_wafers * avg_fail_rate_per_failure_wafer ≈ target_fail_rate
    # Assuming avg ~12% fail rate within failure-pattern wafers:
    avg_internal_fail_rate = 0.12
    fraction_failure_wafers = min(target_fail_rate / avg_internal_fail_rate, 0.8)
    fraction_failure_wafers = max(fraction_failure_wafers, 0.05)  # at least 5%

    num_failure_wafers = min(int(num_wafers * fraction_failure_wafers), len(labeled_df))
    num_none_wafers = min(num_wafers - num_failure_wafers, len(none_df))
    # total = num_failure_wafers + num_none_wafers

    failure_idx = rng.choice(len(labeled_df), size=num_failure_wafers, replace=False)
    none_idx = rng.choice(len(none_df), size=num_none_wafers, replace=False)

    wafer_maps = []
    wafer_ids = []

    for i, idx in enumerate(failure_idx):
        wm = labeled_df.iloc[idx]["waferMap"]
        wafer_maps.append(wm)
        wafer_ids.append(f"W_F_{i:04d}")

    for i, idx in enumerate(none_idx):
        wm = none_df.iloc[idx]["waferMap"]
        wafer_maps.append(wm)
        wafer_ids.append(f"W_N_{i:04d}")

    # Shuffle
    order = rng.permutation(len(wafer_maps))
    wafer_maps = [wafer_maps[i] for i in order]
    wafer_ids = [wafer_ids[i] for i in order]

    return wafer_maps, wafer_ids


def compute_radial_map(wafer_map):
    """Compute normalized radial distance from center for each die position."""
    rows, cols = wafer_map.shape
    cy, cx = rows / 2.0, cols / 2.0
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    radial = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_r = radial.max()
    if max_r > 0:
        radial = radial / max_r
    return radial


def compute_linear_gradient(wafer_map, rng):
    """Compute a random linear gradient across the wafer (simulates process chamber asymmetry)."""
    rows, cols = wafer_map.shape
    angle = rng.uniform(0, 2 * np.pi)
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    # Normalize to [-1, 1]
    yn = (y_coords - rows / 2) / (rows / 2)
    xn = (x_coords - cols / 2) / (cols / 2)
    gradient = np.cos(angle) * xn + np.sin(angle) * yn
    return gradient


def compute_neighborhood_fail_density(wafer_map, window_size):
    """Compute local fail density in a neighborhood window around each die."""
    fail_mask = (wafer_map == 2).astype(float)
    valid_mask = (wafer_map > 0).astype(float)

    # Use uniform filter as a fast local average
    fail_sum = uniform_filter(fail_mask, size=window_size, mode='constant', cval=0.0)
    valid_sum = uniform_filter(valid_mask, size=window_size, mode='constant', cval=0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(valid_sum > 0, fail_sum / valid_sum, 0.0)

    return density


def compute_zone_features(wafer_map, features_grid, zone_rows, zone_cols):
    """
    Compute zone-level aggregated features.
    Returns zone_means, zone_vars, zone_yield per die (broadcast from zone).
    """
    rows, cols = wafer_map.shape
    n_features = features_grid.shape[2]

    zone_mean_grid = np.zeros_like(features_grid)
    zone_var_grid = np.zeros_like(features_grid)
    zone_yield_grid = np.zeros((rows, cols), dtype=float)

    row_edges = np.linspace(0, rows, zone_rows + 1, dtype=int)
    col_edges = np.linspace(0, cols, zone_cols + 1, dtype=int)

    for zr in range(zone_rows):
        for zc in range(zone_cols):
            r_start, r_end = row_edges[zr], row_edges[zr + 1]
            c_start, c_end = col_edges[zc], col_edges[zc + 1]

            zone_mask = wafer_map[r_start:r_end, c_start:c_end] > 0
            zone_features = features_grid[r_start:r_end, c_start:c_end]

            if zone_mask.sum() > 0:
                valid_features = zone_features[zone_mask]
                z_mean = valid_features.mean(axis=0)
                z_var = valid_features.var(axis=0)

                zone_pass = (wafer_map[r_start:r_end, c_start:c_end] == 1).sum()
                zone_total = zone_mask.sum()
                z_yield = zone_pass / zone_total
            else:
                z_mean = np.zeros(n_features)
                z_var = np.zeros(n_features)
                z_yield = 1.0

            zone_mean_grid[r_start:r_end, c_start:c_end] = z_mean
            zone_var_grid[r_start:r_end, c_start:c_end] = z_var
            zone_yield_grid[r_start:r_end, c_start:c_end] = z_yield

    return zone_mean_grid, zone_var_grid, zone_yield_grid


def generate_die_features(wafer_map, config, rng):
    """
    Generate synthetic parametric features for all dies in a wafer.

    Two-stage labeling:
    - old_label: from WM-811K wafer map (pre-test failures)
    - label: post-test failures (old fails + newly failed dies)

    Returns arrays for the die-level dataframe.
    """
    feature_defs = config["features"]
    n_features = len(feature_defs)
    rows, cols = wafer_map.shape
    window_size = config["neighborhood_window"]
    radial_strength = config["radial_gradient_strength"]
    linear_strength = config["linear_gradient_strength"]
    neigh_influence = config["neighborhood_influence"]
    marginal_frac = config["marginal_fail_fraction"]
    new_fail_rate = config.get("new_fail_rate", 0.02)  # mean of per-wafer fail rate distribution

    # Spatial maps
    radial_map = compute_radial_map(wafer_map)
    linear_map = compute_linear_gradient(wafer_map, rng)
    fail_density = compute_neighborhood_fail_density(wafer_map, window_size)

    # old_label: from WM-811K map (1=pass, 2=fail → 0/1 where fail=1)
    old_label_map = (wafer_map == 2).astype(int)

    # Generate new post-test failures among currently passing dies
    # Each wafer gets a RANDOM base fail rate (drawn from exponential distribution)
    # so some wafers have high new-fail counts and others have near zero
    wafer_base_rate = rng.exponential(scale=new_fail_rate)
    wafer_base_rate = min(wafer_base_rate, 0.25)  # cap at 25%

    passing_mask = (wafer_map == 1)
    n_passing = passing_mask.sum()

    # Probability of new failure: wafer-specific base + spatial proximity + edge effect
    new_fail_prob = np.zeros((rows, cols), dtype=float)
    new_fail_prob[passing_mask] = wafer_base_rate
    # Increase probability near existing failures
    new_fail_prob += fail_density * wafer_base_rate * 3.0
    # Increase probability at wafer edges
    new_fail_prob += radial_map * wafer_base_rate * 1.5
    # Clamp
    new_fail_prob = np.clip(new_fail_prob, 0, 0.4)
    # Only apply to passing dies
    new_fail_prob[~passing_mask] = 0

    # Sample new failures
    random_draw = rng.random((rows, cols))
    new_fail_mask = (random_draw < new_fail_prob) & passing_mask

    # Final label: old fails + new fails
    label_map = old_label_map | new_fail_mask.astype(int)

    # Determine which failing dies are "marginal" (hard to separate)
    # Apply marginal logic to ALL fails in label_map (both old and new)
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

    # Generate features
    features_grid = np.zeros((rows, cols, n_features))

    for f_idx, fdef in enumerate(feature_defs):
        base_mean = fdef["base_mean"]
        base_std = fdef["base_std"]
        fail_shift = fdef["fail_shift"]

        # Base feature: all dies get a draw from the pass distribution
        base = rng.normal(base_mean, base_std, size=(rows, cols))

        # Add spatial gradients (process variation)
        radial_coeff = rng.uniform(-1, 1) * radial_strength * base_std
        linear_coeff = rng.uniform(-1, 1) * linear_strength * base_std
        base += radial_map * radial_coeff
        base += linear_map * linear_coeff

        # Neighborhood influence: dies near failures get slight shift
        base += fail_density * neigh_influence * fail_shift

        # Apply fail shift to failing dies (based on final label)
        fail_mask = (label_map == 1)
        # Non-marginal fails get full shift
        non_marginal_fail = fail_mask & (~marginal_mask)
        base[non_marginal_fail] += fail_shift

        # Marginal fails get only a tiny shift (nearly indistinguishable)
        base[marginal_mask] += fail_shift * rng.uniform(0.05, 0.25)

        features_grid[:, :, f_idx] = base

    # Compute zone features (kept for reference but not included in output)
    # zone_means, zone_vars, zone_yield = compute_zone_features(
    #     wafer_map, features_grid, config["zone_rows"], config["zone_cols"]
    # )

    return features_grid, old_label_map, label_map


def wafer_to_dataframe(wafer_id, wafer_map, features_grid,
                       old_label_map, label_map, feature_names):
    """Convert a single wafer's data into rows for the dataset DataFrame."""
    valid_mask = wafer_map > 0  # Only real dies

    die_rows, die_cols = np.where(valid_mask)
    n_dies = len(die_rows)

    records = {
        "wafer_id": [wafer_id] * n_dies,
        "die_row": die_rows,
        "die_col": die_cols,
    }

    # Die-level features
    for f_idx, fname in enumerate(feature_names):
        records[fname] = features_grid[die_rows, die_cols, f_idx]


    # Old label (pre-test: from WM-811K wafer map)
    records["old_label"] = old_label_map[die_rows, die_cols]

    # Label (post-test: old fails + new fails)
    records["label"] = label_map[die_rows, die_cols]

    return pd.DataFrame(records)


def generate_block_readings(n_dies, label_array, config, rng):
    """
    Generate block-level readings for each die.
    Each die gets an array of k float values representing sub-die test readings.

    For passing dies: all blocks have normal readings.
    For failing dies: a small fraction of blocks show anomalous shift (sparse defect signal).

    Returns list of space-separated string representations.
    """
    br_config = config.get("block_readings", {})
    k = config.get("num_block_readings", 2000)
    base_mean = br_config.get("base_mean", 100.0)
    base_std = br_config.get("base_std", 15.0)
    fail_shift_frac = br_config.get("fail_shift_fraction", 0.3)
    anomalous_frac = br_config.get("anomalous_block_fraction", 0.05)
    corr_kernel = br_config.get("block_correlation_kernel", 5)

    fail_shift = fail_shift_frac * base_std
    readings_list = []

    for i in range(n_dies):
        # Base readings for all blocks
        readings = rng.normal(base_mean, base_std, size=k)

        # Add spatial correlation between nearby blocks
        if corr_kernel > 1:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(readings, size=corr_kernel, mode='nearest')
            readings = 0.6 * readings + 0.4 * smooth

        # For failing dies: inject anomalous readings in a sparse subset of blocks
        if label_array[i] == 1:
            n_anomalous = max(1, int(k * anomalous_frac))
            # Choose anomalous block positions (clustered — pick a seed and spread)
            seed_pos = rng.integers(0, k)
            anomalous_positions = set()
            anomalous_positions.add(seed_pos)
            while len(anomalous_positions) < n_anomalous:
                # Cluster around seed with some spread
                offset = int(rng.normal(0, k * 0.05))
                pos = (seed_pos + offset) % k
                anomalous_positions.add(pos)

            for pos in anomalous_positions:
                # Shift can be positive or negative, with some randomness
                readings[pos] += rng.normal(fail_shift, fail_shift * 0.3)

        # Round to 2 decimal places and convert to string
        readings_str = " ".join(f"{v:.2f}" for v in readings)
        readings_list.append(readings_str)

    return readings_list


def generate_dataset(wafer_maps, wafer_ids, config, rng):
    """Generate full dataset for a list of wafers."""
    feature_names = [f["name"] for f in config["features"]]
    all_dfs = []

    for i, (wmap, wid) in enumerate(zip(wafer_maps, wafer_ids)):
        if (i + 1) % 100 == 0:
            print(f"  Processing wafer {i + 1}/{len(wafer_maps)}...")

        features_grid, old_label_map, label_map = \
            generate_die_features(wmap, config, rng)

        df = wafer_to_dataframe(
            wid, wmap, features_grid,
            old_label_map, label_map, feature_names
        )

        # Generate block-level readings for dies in this wafer
        block_readings = generate_block_readings(
            len(df), df["label"].values, config, rng
        )
        df["block_readings"] = block_readings

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def print_summary(df, split_name):
    """Print dataset summary statistics."""
    n_wafers = df["wafer_id"].nunique()
    n_dies = len(df)
    n_fail = df["label"].sum()
    fail_rate = n_fail / n_dies * 100

    print(f"\n{'=' * 50}")
    print(f"  {split_name} Dataset Summary")
    print(f"{'=' * 50}")
    print(f"  Wafers:     {n_wafers}")
    print(f"  Total dies: {n_dies:,}")
    print(f"  Failed:     {n_fail:,} ({fail_rate:.2f}%)")
    print(f"  Passed:     {n_dies - n_fail:,} ({100 - fail_rate:.2f}%)")

    # Feature overlap: compute mean difference / pooled std for each feature
    feature_cols = [c for c in df.columns if not c.startswith("zone_") and
                    c not in ("wafer_id", "die_row", "die_col", "label", "neighborhood_fail_density")]
    if len(feature_cols) > 0:
        print(f"\n  Feature Separability (Cohen's d):")
        pass_df = df[df["label"] == 0]
        fail_df = df[df["label"] == 1]
        for col in feature_cols[:5]:  # Show top 5
            p_mean, p_std = pass_df[col].mean(), pass_df[col].std()
            f_mean, f_std = fail_df[col].mean(), fail_df[col].std()
            pooled_std = np.sqrt((p_std ** 2 + f_std ** 2) / 2)
            d = abs(f_mean - p_mean) / pooled_std if pooled_std > 0 else 0
            print(f"    {col:20s}: d = {d:.4f} (low = high overlap)")
        print(f"    ... ({len(feature_cols)} features total)")
    print(f"{'=' * 50}\n")


def save_outputs(train_df, test_df, wafer_maps_dict, config, export_csv=False):
    """Save datasets in configured formats. Also generates a validation set (no labels)."""
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = config.get("output_formats", ["parquet", "pkl"])

    # Create validation version of test (no labels — only inputs)
    val_df = test_df.drop(columns=["label"])

    if "parquet" in formats:
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        val_df.to_parquet(output_dir / "validation.parquet", index=False)
        print(f"  Saved: {output_dir}/train.parquet, test.parquet, validation.parquet")

    if "pkl" in formats:
        train_df.to_pickle(output_dir / "train.pkl")
        test_df.to_pickle(output_dir / "test.pkl")
        val_df.to_pickle(output_dir / "validation.pkl")
        with open(output_dir / "wafer_maps.pkl", "wb") as f:
            pickle.dump(wafer_maps_dict, f)
        print(f"  Saved: {output_dir}/train.pkl, test.pkl, validation.pkl, wafer_maps.pkl")

    if export_csv or "csv" in formats:
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        val_df.to_csv(output_dir / "validation.csv", index=False)
        print(f"  Saved: {output_dir}/train.csv, test.csv, validation.csv")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic die-level features for hackathon")
    parser.add_argument("--num_wafers", type=int, default=None,
                        help="Total number of wafers (overrides config train+test)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--csv", action="store_true",
                        help="Also export CSV format")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine wafer counts
    if args.num_wafers:
        n_train = int(args.num_wafers * 0.8)
        n_test = args.num_wafers - n_train
    else:
        n_train = config["num_wafers_train"]
        n_test = config["num_wafers_test"]

    total_needed = n_train + n_test

    # Initialize RNG with seed
    rng = np.random.default_rng(config["seed"])

    # Load WM-811K
    labeled_df, none_df = load_wm811k(config["wm811k_path"])

    # Select wafers for train and test
    print(f"\nSelecting {n_train} train + {n_test} test wafers...")
    all_maps, all_ids = select_wafers(
        labeled_df, none_df, total_needed, config["target_fail_rate"], rng
    )

    train_maps = all_maps[:n_train]
    train_ids = all_ids[:n_train]
    test_maps = all_maps[n_train:]
    test_ids = all_ids[n_train:]

    # Store wafer maps dict for pkl output
    wafer_maps_dict: dict = {str(wid): wmap for wid, wmap in zip(all_ids, all_maps)}

    # Generate features
    print(f"\nGenerating train features ({n_train} wafers)...")
    train_df = generate_dataset(train_maps, train_ids, config, rng)

    print(f"\nGenerating test features ({n_test} wafers)...")
    test_df = generate_dataset(test_maps, test_ids, config, rng)

    # Print summaries
    print_summary(train_df, "TRAIN")
    print_summary(test_df, "TEST")

    # Save
    print("Saving outputs...")
    save_outputs(train_df, test_df, wafer_maps_dict, config, export_csv=args.csv)

    print("\nDone! Dataset generation complete.")


if __name__ == "__main__":
    main()
