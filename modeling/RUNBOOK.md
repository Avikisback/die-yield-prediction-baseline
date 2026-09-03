# Execution runbook

## 1. Generate the supplied data

Place `LSWMD.pkl` in `data/`, then run:

```powershell
python generate_data.py
```

## 2. Build compact, per-wafer feature caches

```powershell
python -m modeling.cache_features input/train.csv cache/train
python -m modeling.cache_features input/test.csv cache/test
python -m modeling.cache_features input/validation.csv cache/validation
```

The cache builder reads in chunks, discards the original 2,000-value strings
after extracting block statistics, and stores `float32` Parquet files.

## 3. Run the fair ablation

For a quick first pass:

```powershell
python -m modeling.run_experiments cache/train results/modeling --repeats 1
```

For the repeated grouped comparison:

```powershell
python -m modeling.run_experiments cache/train results/modeling --repeats 3
```

The experiments include parametric-only, Model A, block-only, and Model B.
Every split is grouped by `wafer_id`; only `old_label == 0` rows are scored.

## 4. Train the selected model and predict

```powershell
python -m modeling.train_predict `
  cache/train `
  cache/validation `
  results/modeling/experiment_summary.json `
  results/submission.csv `
  --experiment model_b_boost
```

Old failures are assigned `predicted_label = 1` directly. The stored OOF
threshold is used for eligible dies. The command also saves a fitted model and
its metadata next to the submission.

## Backend

The experiment code uses LightGBM when installed and otherwise uses
scikit-learn histogram gradient boosting. Install the optional accelerator with:

```powershell
python -m pip install "lightgbm>=4.0.0"
```

Do not tune against `test.csv`: the starter code constructs `validation.csv`
from the same test rows with only `label` removed.

