# Multi-Resolution Die Yield Prediction

Predicting which dies pass the pre-test but fail the post-test, from three
resolutions of the same wafer: 500 parametric measurements per die, the
neighbourhood the die sits in, and 2,000 sub-die block readings.

Two models, as the problem statement specifies:

- **Model A** — die-level measurements plus spatial context.
- **Model B** — everything in Model A plus the block readings.

Three pipelines live here, all scored by the *same* code
(`modeling/validation.py`) on the *same* wafer-grouped folds and the same
seed-42 dataset:

| | what it is |
|---|---|
| `modeling/` | the first reproducible baseline |
| `mrf/` | diagonal discriminant, stacked head, per-wafer rate recovery |
| `tuned/` | this pipeline: the posterior the generator implies, every piece estimated |

`mrf/` is **re-run from scratch here**, not quoted: `results/mrf_rerun/`
reproduces its published table to four decimals — `B_stacked_cal` AP 0.6209,
`block_only` ROC-AUC 0.7259, 154,037 eligible dies, 6,519 failures. So the
comparison below is like-for-like and not an artefact of a different
environment or a differently generated dataset.

## Headline

Cross-validation on the 160 training wafers, five wafer-grouped folds repeated
three times:

| | baseline | mrf | this | vs mrf |
|---|--:|--:|--:|--:|
| **Model A** — AP | 0.5137 | 0.5524 | **0.5874** | +6.3% |
| **Model A** — fail F1 | 0.5268 | 0.5323 | **0.5571** | +4.7% |
| **Model B** — AP | 0.5758 | 0.6209 | **0.6564** | +5.7% |
| **Model B** — fail F1 | 0.5537 | 0.5777 | **0.6071** | +5.1% |
| **Model B** — ROC-AUC | — | 0.9132 | **0.9261** | +1.4% |

On the 40 held-out test wafers, scored once, Model B reaches **AP 0.6597**,
**fail F1 0.6041**, ROC-AUC 0.9327 — against mrf's 0.6302 / 0.5836 and the
baseline's 0.5934 / 0.5600.

Exact numbers, every ablation and every paired test:
**[RESULTS_TUNED.md](RESULTS_TUNED.md)**. Figures: `results/tuned_figures/`.

## The model

The generator makes the three resolutions conditionally independent given the
label, so the posterior is not something to approximate with a classifier — it
can be written down:

```
logit P(fail_i) = logit(pi_i) + f_param(s_i) + f_block(b_i)
pi_i            = min(rate_w * h_i, 0.4)
```

- `h_i` is the pre-test hazard shape, `1 + 3*density + 1.5*radius`, exactly what
  `generate_die_features` multiplies the wafer's rate by;
- `rate_w` is the wafer's own base rate, recovered from its **unlabelled** dies;
- `s_i` is the diagonal discriminant over the 500 measurements — a *sufficient*
  statistic for them, not a summary of them;
- `b_i` is a likelihood-ratio scan over the 2,000 block readings;
- `f_param` and `f_block` are smooth functions fitted from the training labels.

Every term is estimated. Nothing reads `config.yaml` except `tuned/ceiling.py`,
whose only job is to say how much better a model that *did* know the
generator's parameters could have done.

## What moved

### 1. The hazard shape was being reconstructed on the wrong grid

`generate_die_features` normalises each die's radius by the largest distance
anywhere in the wafer-map **array** — a corner, where there is no die at all.
`mrf/spatial.py` normalises by the largest distance among the **observed dies**,
about 0.75 of that, so its radius is inflated by a third and the generator's
coefficient of 1.5 becomes an effective 2.0.

Rebuilding the radial map on the array grid makes the reconstruction exact.
`tests/test_tuned.py` checks it against `generate_data.compute_radial_map`
element by element, and the ceiling run reports a correlation of **1.0000**
between the reconstructed log hazard and the generator's own per-die
probability divided by its wafer rate.

There is a second, quieter confirmation. The model is *allowed* to reshape the
hazard with a spline, and turning that freedom off changes average precision by
−0.0003. Given a grid that is already exact, there is nothing left to correct.

### 2. The block channel had a likelihood ratio available and was using a filter

`generate_block_readings` adds ~100 positive spikes to a failing die, clustered
around one uniformly random seed, **after** the 5-tap smoothing. Writing out the
log-likelihood ratio for a cluster seeded at `t`:

```
log LR(t) = sum_p log[ 1 + q(p - t) * (r(u_p) - 1) ]
r(u)      = N(u; mu, sigma^2 + tau^2) / N(u; 0, sigma^2)
```

Three things follow, and all three were measured: `r` is **nonlinear** (a
matched filter is its linear approximation); the seed should be **integrated
out**, not maximised over; and the noise should be **whitened first**, because
`r` is a per-sample transform and the readings are smoothed.

The block channel on its own, same folds, same metric code:

| | AP | ROC-AUC |
|---|--:|--:|
| mrf scan bank | 0.1372 | 0.7259 |
| this | **0.1596** | **0.7461** |

### 3. The wafer rate is a posterior, not a point estimate that needs shrinking

`mrf/calibrate.py` recovers each wafer's rate by maximising a two-component
mixture on the scores, then multiplies the offset by a hand-picked `alpha = 0.5`
because the unshrunk correction costs fail-class F1.

Two things change. The **hazard shape is used** — dies on a wafer do not share a
failure probability, and mixing them with a single weight throws that away. And
the rate is **integrated out** under the exponential prior the generator draws
it from, whose scale is fitted on the training wafers, so the shrinkage is
decided by each wafer's own sample size instead of one constant for all of them.

| | correlation with the truth |
|---|--:|
| recovered rate vs the drawn `wafer_base_rate` | **0.947** |
| implied fail fraction vs the actual, this pipeline | **0.988** |
| implied fail fraction vs the actual, mrf | 0.967 |

The fitted prior scale comes out at 0.0199 against the generator's 0.02.

### 4. One constant in the fit is not identified, and calibration is what decides it

The head sees `logit(prior) + intercept + prior terms + evidence terms`, and
only the total is identified: a constant can move between the prior side and the
evidence side without changing a single fitted probability. The rate step is
*not* indifferent, because it reads the evidence as a likelihood ratio.

This is the one place where the best-ranking option is not the one shipped, and
the results table carries a `predicted/actual` column so the trade is visible:

| where the constant goes | AP | predicted/actual failures |
|---|--:|--:|
| left on the prior side | 0.6578 | 0.583 |
| **so the posterior predicts the observed failure count** | 0.6564 | **0.997** |
| so `E[exp(evidence)] = 1` over passes | 0.6504 | 1.879 |

Leaving the intercept out of the evidence quietly shrinks the wafer-rate term —
the same trade `mrf.calibrate` makes deliberately with `alpha`. It ranks 0.0012
better and its probabilities sum to 58% of the failures actually present. A fab
deciding how many dies to re-inspect needs the second row, so that is what
ships. The textbook identity is worst of the three, because `exp(evidence)`
reaches `e^16` on the clearest failures and its sample mean is then decided by a
handful of them.

### 5. Two smaller things the generator hands over

**Pre-test failures are extra labelled positives.** Dies with `old_label == 1`
carry the same `fail_shift` under the same marginal rule, and the same block
anomaly. They are excluded from scoring, as required, but not from fitting the
two channel directions — worth +0.0011 AP, better on 15 folds out of 15.

**The neighbourhood leaks into every measurement.** `base += fail_density * 0.2
* fail_shift` is applied to every die, failing or not, so the diagonal score
carries a neighbourhood offset that is not evidence about that die. The prior
already counts the neighbourhood; leaving the offset in counts it twice, and an
additive head cannot remove it because the true form is `f(s - c*density)`, not
`f(s) - g(density)`. Fitting `c` on passing dies and subtracting is worth
+0.0015 AP.

## What was measured and did not help

Reported because knowing which plausible ideas are dead is part of the result.

**Smooth per-channel ratios tie a single coefficient.** The failing class is a
mixture — 65% of failures get 5–25% of the shift — so the parametric
log-likelihood ratio is genuinely kinked, and a spline should beat a slope. It
does not: 0.6022 against 0.6023. `results/tuned_figures/channel_shapes.png`
shows why. The kink is real but it sits out in the tail where almost no dies
are; across the mass of the distribution the ratio is very close to straight.
The splines are kept because they are the correct form and cost nothing, not
because they earn their place.

**Refitting the head against the recovered rates does nothing.** One, two and
three passes span 0.0002 AP, and which is best *flips* depending on the setting
in §4 — a knob with no signal. The single pass ships because it is the simplest
and fastest of three indistinguishable options.

**A neural network does not beat the derived block detector.** A 1-D CNN over
the raw 2,000 readings — circular convolutions, a receptive field reaching the
cluster's scale, learnable soft-max pooling over shifts, trained on a GPU over
40,000 simulated dies with wafer-grouped folds — is the tool that would find
structure a derivation missed. On those same dies and folds it reaches ROC-AUC
**0.7423** against the derived statistic's **0.7559**, and the two scores
together (0.7497) are worse than the derived score alone, so the network is not
even carrying anything *different*. The closed form is not merely defensible;
there is nothing left to find. See `tuned/blockcnn.py`.

**Per-wafer gradient removal** was rejected in `mrf/` with evidence and is not
revisited; the analysis there holds.

## How close is this to optimal?

`tuned/ceiling.py` scores the same dies with things the generator knows and no
model can — each wafer's actual drawn rate, and the exact discriminant direction
implied by `config.yaml`.

| | AP | what it is allowed to know |
|---|--:|---|
| true prior only | 0.1285 | the generator's own per-die probability, no measurements |
| oracle direction, population rate | 0.6040 | the generator's `fail_shift`, but no per-wafer rate |
| **this model** | **0.6584** | nothing but the training data |
| Bayes oracle | 0.6609 | true wafer rate *and* the generator's direction |
| true rate, fitted channels | 0.6632 | a perfect wafer rate |

The model is at **99.6% of the Bayes oracle**. Two more measurements say where
the remaining 0.4% is *not*:

- the fitted parametric direction scores ROC-AUC 0.8547 against the generator's
  own direction at 0.8541 — the die measurements are exhausted;
- scored only against failures that received the full shift, the same model
  reaches **AP 0.9990**.

So the entire residual error is the 65% of failures the generator deliberately
makes near-invisible, plus the block cluster's unknowable seed
(`tuned/blocksim.py`: on identical dies, ROC-AUC **0.861** for a detector told
where each cluster was seeded, against **0.742** for the same detector having to
search all 2,000 positions). Neither is a modelling failure. There is no further
accuracy in this dataset to find.

## Interpretability

The model is additive in log-odds by construction, so an attribution is not an
approximation of it — it is it:

```
logit(p) = logit(prior from the wafer rate and the neighbourhood)
         + f_param(the die's 500 measurements, collapsed)
         + f_block(the die's 2,000 sub-die readings, collapsed)
```

`tests/test_tuned.py` asserts the pieces sum back to the head's output exactly.
Because the parametric channel is a single sufficient statistic with one weight
per measurement, the per-measurement contribution is exact too, and the fitted
separations correlate with the generator's actual `fail_shift/base_std` at
**0.9985** — the model learned the real mechanism.

## Class imbalance

About 4.2% of eligible dies fail, so accuracy is not a usable metric —
"all pass" scores 95.8%.

- Folds are grouped by `wafer_id` and stratified; no wafer is split.
- **No class weighting.** The model produces calibrated probabilities, which the
  wafer-rate step needs and §4 is about protecting; re-weighting the fit would
  break that. The operating point is set by an explicit threshold instead.
- The threshold is chosen on out-of-fold predictions only.
- Reported as AP and fail-class F1 with precision and recall shown separately,
  plus an inspection-budget curve: what share of failures a fab catches if it can
  re-examine 1%, 5% or 10% of dies.
- Pre-test failures are excluded from fitting and scoring, and re-attached as
  certain failures only when the submission is assembled.

## Layout

```
tuned/
  hazard.py       the pre-test hazard shape, on the grid the generator used
  blocks.py       likelihood-ratio scan over the 2,000 block readings
  channels.py     the diagonal discriminant, and why it is sufficient
  head.py         additive log-odds head with a fixed offset
  waferrate.py    the per-wafer rate posterior
  pipeline.py     the model: prior, two evidence channels, the rate step
  cache.py        CSV -> one compact Parquet file per wafer
  genstream.py    byte-identical data generation that fits in memory, plus latents
  select.py       the three free constants, chosen on training folds only
  experiments.py  wafer-grouped cross-validation over the catalogue
  final.py        held-out scoring and the submission
  ceiling.py      what any model could have reached, and where the rest goes
  blocksim.py     the block channel's own ceiling, by re-running its generator
  blockcnn.py     the same question asked of a neural network (GPU)
  figures.py      every figure
  report.py       regenerates RESULTS_TUNED.md from the saved result files
```

## Reproduce

Python 3.10+, `LSWMD.pkl` in `data/`
([Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)).
Exact commands: **[tuned/RUNBOOK.md](tuned/RUNBOOK.md)**.

```bash
python -m pip install -r requirements.txt
python -m tuned.genstream
python -m tuned.cache input/train.csv cache_tuned/train
python -m tuned.cache input/test.csv cache_tuned/test --null cache_tuned/train/block_null.json
python -m tuned.cache input/validation.csv cache_tuned/validation --null cache_tuned/train/block_null.json
python -m unittest discover -s tests
python -m tuned.select cache_tuned/train results/tuned_selection
python -m tuned.experiments cache_tuned/train results/tuned --repeats 3
python -m tuned.final cache_tuned/train cache_tuned/test cache_tuned/validation results/tuned_final
```

`python -m tuned.genstream` writes the same three CSVs as `generate_data.py` —
same seed, same draws, byte for byte, asserted by a test — but streams wafers to
disk instead of concatenating them in memory, which keeps the peak under a
gigabyte instead of several.

`validation.csv` is the test rows with the label dropped, so it is not an
independent holdout and nothing was tuned against it. The 40 test wafers are
touched exactly once, by `tuned/final.py`.
