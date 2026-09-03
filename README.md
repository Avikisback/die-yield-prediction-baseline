# Multi-Resolution Die Yield Prediction

Predicting which dies pass the pre-test but fail the post-test, from three
resolutions of the same wafer: 500 parametric measurements per die, the
neighbourhood the die sits in, and 2,000 sub-die block readings.

Two models, as the problem statement specifies:

- **Model A** — die-level measurements plus spatial context.
- **Model B** — everything in Model A plus the block readings.

`modeling/` holds the earlier baseline. `mrf/` holds this pipeline. Both are
scored by the *same* code (`modeling/validation.py`) on the *same* wafer-grouped
folds and the same seed-42 dataset, so the comparison is like-for-like.

## Headline

Cross-validation on the 160 training wafers, five wafer-grouped folds repeated
three times, scored by the baseline's own metric code:

| | baseline AP | this | | baseline fail F1 | this | |
|---|--:|--:|--:|--:|--:|--:|
| Model A | 0.5137 | **0.5524** | +7.5% | 0.5268 | **0.5323** | +1.0% |
| Model B | 0.5758 | **0.6209** | +7.8% | 0.5537 | **0.5777** | +4.3% |

On the 40 held-out test wafers, scored once: Model B reaches **AP 0.6302**,
**fail F1 0.5836**, ROC-AUC 0.9210, against the baseline's 0.5934 / 0.5600.

Exact numbers: **[RESULTS.md](RESULTS.md)**. Figures: `results/figures/`.
Full write-up: `results/report.html`.

## What moved, and what didn't

The baseline was already close to the information available in the die
measurements, so most of the obvious ideas do nothing. Three things were
measured rather than assumed:

**1. The die-level ceiling is already reached — this was worth knowing first.**
The generator's per-feature `fail_shift` is a deterministic function of
`config.yaml`, so the exact likelihood-ratio direction can be written down and
scored. That oracle reaches AP 0.5271. The baseline's fitted Model A reaches
0.5137 — 97% of it. No amount of feature engineering or tuning on the parametric
block was going to matter, and measuring that early is what redirected the work.

**2. The 500 measurements are conditionally independent given the label.**
Each is an independent normal draw plus a per-feature constant, so the matched
estimator is the diagonal (naive-Bayes) discriminant: one mean difference per
feature, rather than 500 jointly-fitted coefficients supported by only 6,519
failures. Collapsing the parametric block to that single score and putting a
small logistic head on top of it — spatial and block features alongside — is
both more accurate and about 3x faster to fit.

**3. The per-wafer failure rate is unpredictable, but it is measurable.**
`generate_die_features` draws `wafer_base_rate` from an exponential distribution
and multiplies every die's hazard by it. Nothing visible before the test predicts
it, and it spans roughly 0% to 25%. But the scores on a wafer are a two-component
mixture whose mixing weight *is* that rate, so it can be recovered by maximum
likelihood from the wafer's own unlabelled scores and applied as a prior shift.
On held-out wafers the recovered rate tracks the real one closely, and it is the
single largest contributor to the improvement here.

The offset is shrunk by a factor chosen on training out-of-fold scores. At full
strength the correction assumes the recovered rate is exact; it is not, and
unshrunk it costs more in false positives than it gains in ranking.

### Two ideas that were tried and rejected

Both are kept in the repository, with the evidence, because the negative results
are part of the analysis.

**Per-wafer gradient removal.** Every feature carries a radial and a linear
process gradient whose coefficients are redrawn per wafer, and those gradients
lie exactly in the span of `[1, x, y, radius]`, so regressing them out per wafer
looks obviously correct. It isn't. On real WM-811K maps the new failures are
themselves clustered against radius and against pre-test defect neighbourhoods,
so the fit absorbs signal along with the nuisance. Scored with the generator's
own coefficients, average precision falls from **0.5271 raw to 0.5065
detrended**. Run `mrf/parametric.py` to reproduce; the ablation is in the results
table.

**Matched-filter block scans.** The block anomalies are ~100 positive spikes
clustered around one seed with a spread of `0.05 * k`, which makes a circular
Gaussian scan the matched detector — better founded than the baseline's boxcar
sums. Head-to-head on identical rows, folds and classifier, it scores AP 0.1321
against the baseline's 0.1296: the same information in half the columns. It is
kept because the scan's peak location names *which* block region is anomalous,
which the report needs, but it is not a source of accuracy.

## Interpretability

Both models are linear in standardised inputs, and the stacked model is a linear
head over a linear score, so an attribution is not an approximation of the model
— it is the model. For each die the log-odds decompose exactly:

```
logit(p) = intercept + parametric terms + spatial terms + block terms
```

`tests/test_mrf.py` asserts that decomposition sums back to `predict_proba` for
both model forms, to 1e-6.

That gives, per predicted failure: the measurements that drove it and by how
much, the neighbourhood contribution, and the block-scan peak with the block
range it sits in. Laid back onto the wafer grid, the same decomposition produces
one heatmap per resolution, so a cluster driven by neighbourhood effects looks
visibly different from one driven by a die's own parameters.

As a check that the model learned the real mechanism rather than an artefact, the
fitted parametric coefficients are correlated against the generator's actual
`fail_shift / base_std` (see RESULTS.md).

## Class imbalance

About 4.2% of eligible dies fail, so accuracy is not a usable metric — predicting
"all pass" scores 95.8%. The pipeline treats imbalance explicitly:

- folds are grouped by `wafer_id` and stratified, so no wafer is split across
  training and validation, and every reported failure rate is a real held-out one;
- the minority class is weighted by `sqrt(negatives/positives)`. Full `balanced`
  weighting pushes the boundary well past the point that maximises fail-class F1;
- the decision threshold is chosen on out-of-fold predictions only, never on the
  data it is then scored against;
- results are reported as AP and fail-class F1, with precision and recall shown
  separately, plus an inspection-budget curve: what share of true failures a fab
  catches if it can afford to re-examine 1%, 5% or 10% of dies.

## Layout

```
mrf/
  spatial.py      neighbourhood, radius, edge distance, hazard shape
  parametric.py   per-wafer gradient removal (rejected; kept for the ablation)
  block.py        circular Gaussian scan bank over the 2,000 readings
  cache.py        CSV -> one compact Parquet file per wafer
  models.py       diagonal discriminant, stacked model, backends
  calibrate.py    per-wafer rate recovery and the shrunk prior shift
  experiments.py  wafer-grouped cross-validation over the whole catalogue
  final.py        cross-fitted holdout scoring and the submission
  interpret.py    exact attributions, wafer layers, coefficient recovery
  figures.py      every figure in the report
  report.py       regenerates RESULTS.md from the saved result files
```

## Reproduce

Python 3.10+. Place `LSWMD.pkl` in `data/`
([Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)).

```bash
python -m pip install -r requirements-modeling.txt
python generate_data.py
python -m mrf.cache input/train.csv cache_mrf/train
python -m mrf.cache input/test.csv cache_mrf/test
python -m mrf.cache input/validation.csv cache_mrf/validation
python -m unittest discover -s tests
python -m mrf.experiments cache_mrf/train results/mrf --repeats 3
python -m mrf.final cache_mrf/train cache_mrf/test cache_mrf/validation results/mrf_final
python -m mrf.figures results/mrf_final cache_mrf/test results/figures
python -m mrf.report
```

`validation.csv` is built by the starter generator from the test rows with the
label column dropped, so it is not an independent holdout and nothing is tuned
against it. The 40 test wafers are touched exactly once, by `mrf/final.py`.
