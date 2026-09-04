# Runbook — generator-matched fusion

Python 3.10+.  Every command below is run from the repository root.

```powershell
python -m pip install -r requirements.txt
```

## 1. Data

Put `LSWMD.pkl` at `data/LSWMD.pkl` (Kaggle: `qingyi/wm811k-wafer-map`), then

```powershell
python -m tuned.genstream
```

This writes `input/train.csv`, `input/test.csv`, `input/validation.csv` and
`input/ground_truth.parquet`.  The three CSVs are **byte-identical** to what
`python generate_data.py` produces — same seed, same draws, same order — but the
wafers are streamed to disk one at a time instead of being concatenated in
memory, which keeps the peak under a gigabyte.  `tests/test_tuned.py` asserts the
random draws match.

`ground_truth.parquet` holds the latents the generator throws away — each
wafer's drawn `wafer_base_rate`, each die's `new_fail_prob`, and which failures
were selected as marginal.  Only `tuned/ceiling.py` reads it.

## 2. Features

```powershell
python -m tuned.cache input/train.csv cache_tuned/train
python -m tuned.cache input/test.csv cache_tuned/test --null cache_tuned/train/block_null.json
python -m tuned.cache input/validation.csv cache_tuned/validation --null cache_tuned/train/block_null.json
```

About nine minutes for the training split.  `--null` reuses the training split's
readings-level, scale, noise spectrum and per-scan null constants, so a block
statistic means the same thing in all three splits.

## 3. Tests

```powershell
python -m unittest discover -s tests -v
```

## 4. Choose the three constants (training wafers only)

```powershell
python -m tuned.select cache_tuned/train results/tuned_selection
```

## 5. Cross-validation

```powershell
python -m tuned.experiments cache_tuned/train results/tuned --repeats 3
```

## 6. Held-out wafers and the submission

```powershell
python -m tuned.final cache_tuned/train cache_tuned/test cache_tuned/validation results/tuned_final
```

Writes `results/tuned_final/submission.csv` with a `predicted_label` for every
die, pre-test failures forced to 1.

## 7. Ceiling and figures

```powershell
python -m tuned.ceiling cache_tuned/train results/tuned_ceiling
python -m tuned.figures cache_tuned/train results results/tuned_figures --mrf-oof results/mrf_rerun/oof_predictions.parquet
python -m tuned.report results --output RESULTS_TUNED.md
```

## 8. Reproducing the previous pipeline on the same folds

```powershell
python -m mrf.cache input/train.csv cache/train
python -m mrf.experiments cache/train results/mrf_rerun --repeats 3
```

Run this if you want the comparison table rebuilt from scratch rather than taken
from `RESULTS.md`; it is what `results/mrf_rerun/` was produced by.
