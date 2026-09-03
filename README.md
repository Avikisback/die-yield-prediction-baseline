# Die Yield Prediction — Baseline

This private repository contains the first reproducible baseline for the
multi-resolution die-yield prediction task. It compares:

- **Model A:** 500 parametric die features plus leakage-safe spatial context.
- **Model B:** Model A features plus compact statistics extracted from each
  die's 2,000 block readings.

Only dies with `old_label == 0` are used for fitting and evaluation. Splits are
grouped by `wafer_id`, so dies from one wafer never appear in both training and
validation folds. Old failures are assigned `predicted_label = 1` only when the
final all-die submission is constructed.

## Baseline results

| Model | Repeated grouped CV AP | Repeated grouped CV fail F1 | Holdout AP | Holdout fail F1 |
|---|---:|---:|---:|---:|
| Model A — die + spatial | 0.5137 | 0.5268 | 0.5320 | 0.5297 |
| Model B — die + spatial + block | **0.5758** | **0.5537** | **0.5934** | **0.5600** |

The holdout F1 values use thresholds selected only from training out-of-fold
predictions. Model B improves holdout average precision by **0.0614** and fail
F1 by **0.0303** over Model A.

Detailed results are in:

- `results/cross_validation/experiment_summary.csv`
- `results/holdout/holdout_summary.csv`

## Repository contents

- `modeling/features.py` — spatial features and robust multiscale block scans.
- `modeling/run_experiments.py` — repeated stratified group cross-validation.
- `modeling/train_predict.py` — final training and submission generation.
- `modeling/evaluate_holdout.py` — locked-threshold Model A/B comparison.
- `artifacts/model_a_linear.joblib` and `model_b_linear.joblib` — fitted models.
- `artifacts/submission.csv` — baseline all-die predictions.
- `tests/` — feature, split, and pipeline checks.

The generated datasets, WM-811K source file, feature caches, and per-row labeled
holdout/OOF predictions are intentionally excluded.

## Reproduce

Use Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python generate_data.py
python -m modeling.cache_features input/train.csv cache/train
python -m modeling.cache_features input/test.csv cache/test
python -m modeling.cache_features input/validation.csv cache/validation
python -m unittest discover -s tests -v
python -m modeling.run_experiments cache/train results/modeling --repeats 3
```

For the exact prediction and evaluation commands, see
`modeling/RUNBOOK.md`. Place `LSWMD.pkl` at `data/LSWMD.pkl` before generating
data. Do not tune against `test.csv`: the supplied generator creates
`validation.csv` from the same test rows with only `label` removed.

## Scope

This repository is intentionally frozen at the first baseline. Later
generator-aware scoring, GPU tuning, matched block filters, hierarchical models,
and experimental ensembles are not included.
