# Modeling pipeline

This package starts with the two signals that match the supplied generator:

1. pre-test spatial context derived only from coordinates and `old_label`; and
2. robust, circular multiscale scan statistics for the clustered block anomaly.

The cache builder reads the very large CSV in chunks and writes one compact
Parquet file per wafer, avoiding repeated parsing of all 2,000 block readings.

```powershell
python -m modeling.cache_features input/train.csv cache/train
python -m modeling.cache_features input/test.csv cache/test
python -m unittest discover -s tests -v
```

Only dies with `old_label == 0` should be passed to model training and metric
calculation.  Old failures remain in the cache so a final submission can assign
them `predicted_label = 1` without calling the model.

The first four experiments should be run on identical repeated grouped folds:

- parametric-only linear and boosted baselines;
- Model A: parametric plus spatial features;
- block-only scan-feature model; and
- Model B: Model A plus block features.

Do not tune against `test.csv`.  The starter generator creates
`validation.csv` by dropping `label` from the same test rows, so it is not an
independent local holdout.
