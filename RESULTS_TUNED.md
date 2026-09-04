# Results

Every cross-validation number below uses five wafer-grouped folds repeated three times, the fold builder and metric code in `modeling/validation.py`, and the same seed-42 dataset the earlier baseline and `mrf` were scored on. Only dies with `old_label == 0` are scored.

## Cross-validation, all experiments

| experiment | AP | ROC-AUC | fail F1 | precision | recall | predicted/actual | note |
|---|---:|---:|---:|---:|---:|---:|---|
| B_evidence_level_none | 0.6578 | 0.9268 | 0.6078 | 0.7532 | 0.5094 | 0.583 | ablation: leave the head's constant on the prior side |
| B_no_prior_correction | 0.6566 | 0.9263 | 0.6080 | 0.7618 | 0.5059 | 1.000 | ablation: hazard shape taken as exact |
| B_two_rounds | 0.6566 | 0.9262 | 0.6078 | 0.7685 | 0.5027 | 0.998 | ablation: refit the head against the recovered rates |
| B_three_rounds | 0.6565 | 0.9262 | 0.6081 | 0.7820 | 0.4975 | 0.999 | ablation: refit it twice |
| B_full | 0.6564 | 0.9261 | 0.6071 | 0.7689 | 0.5016 | 0.997 | Model B, smooth ratios plus the recovered wafer rate |
| B_no_old_fails | 0.6553 | 0.9257 | 0.6064 | 0.7654 | 0.5021 | 0.981 | ablation: parametric direction without pre-test fails |
| B_keep_density_offset | 0.6549 | 0.9256 | 0.6050 | 0.7386 | 0.5123 | 0.998 | ablation: leave the generator's density term in the parametric score |
| B_evidence_level_mean_exp | 0.6504 | 0.9232 | 0.6030 | 0.7378 | 0.5099 | 1.879 | ablation: move the constant so E[exp(evidence)] = 1 over passes |
| B_linear | 0.6023 | 0.8981 | 0.5698 | 0.7612 | 0.4553 | 1.002 | Model B, one coefficient per channel (no spline) |
| B_smooth | 0.6022 | 0.8977 | 0.5697 | 0.7516 | 0.4587 | 1.002 | Model B, smooth per-channel log-likelihood ratios |
| A_full | 0.5874 | 0.8978 | 0.5571 | 0.8265 | 0.4202 | 0.997 | Model A, plus the recovered per-wafer rate |
| A_linear | 0.5276 | 0.8545 | 0.5304 | 0.8813 | 0.3794 | 1.002 | Model A, one coefficient per channel (no spline) |
| A_smooth | 0.5275 | 0.8544 | 0.5302 | 0.8804 | 0.3794 | 1.002 | Model A, smooth per-channel log-likelihood ratios |
| parametric_only | 0.5263 | 0.8533 | 0.5298 | 0.8845 | 0.3781 | 1.000 | die measurements alone |
| block_only | 0.1596 | 0.7454 | 0.2217 | 0.1865 | 0.2730 | 1.000 | sub-die readings alone |
| prior_only | 0.0523 | 0.5557 | 0.0939 | 0.0554 | 0.3074 | 1.662 | pre-test hazard and recovered wafer rate, no die evidence |

## The same folds, the same data, the previous pipeline

`mrf` re-run here rather than quoted, so the comparison cannot be an artefact of a different environment.

| experiment | AP | ROC-AUC | fail F1 | precision | recall |
|---|---:|---:|---:|---:|---:|
| B_stacked_cal | 0.6209 | 0.9132 | 0.5777 | 0.7909 | 0.4550 |
| B_stacked | 0.5863 | 0.8904 | 0.5596 | 0.7921 | 0.4326 |
| B_flat | 0.5732 | 0.8829 | 0.5525 | 0.8020 | 0.4214 |
| B_detrended | 0.5657 | 0.8770 | 0.5503 | 0.8697 | 0.4025 |
| A_stacked_cal | 0.5524 | 0.8844 | 0.5323 | 0.8532 | 0.3869 |
| A_stacked | 0.5238 | 0.8526 | 0.5282 | 0.8883 | 0.3758 |
| parametric_only | 0.5128 | 0.8422 | 0.5268 | 0.9408 | 0.3659 |
| A_flat | 0.5116 | 0.8419 | 0.5258 | 0.9108 | 0.3695 |
| block_only | 0.1372 | 0.7259 | 0.1963 | 0.1401 | 0.3278 |

## Held-out test wafers (40 wafers, scored once)

| model | AP | ROC-AUC | fail F1 | precision | recall | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| model_a | 0.5898 | 0.8996 | 0.5412 | 0.7980 | 0.4094 | 0.9706 |
| model_b | 0.6597 | 0.9327 | 0.6041 | 0.7546 | 0.5036 | 0.9721 |

Published baseline holdout for reference: Model A AP 0.5320 / fail F1 0.5297, Model B AP 0.5934 / fail F1 0.5600.

## Ceiling

What the model reaches against what the generator's own knowledge could reach, on the training wafers.  See `tuned/ceiling.py` for what each rung is allowed to see.

| level | AP | ROC-AUC | fail F1 | what it is allowed to know |
|---|---:|---:|---:|---|
| model | 0.6584 | 0.9270 | 0.6104 | the fitted model, all training wafers in-sample |
| true_prior | 0.1285 | 0.7550 | 0.1931 | generator's own per-die probability, no die evidence |
| true_prior_plus_fitted_evidence | 0.6632 | 0.9297 | 0.6132 | true wafer rate, fitted channels: isolates the rate-recovery loss |
| bayes_oracle | 0.6609 | 0.9292 | 0.6113 | true wafer rate and the generator's own parametric direction |
| oracle_direction_only | 0.6040 | 0.8994 | 0.5725 | generator's parametric direction, population rate: isolates estimation |
| model_hard_failures_only | 0.9990 | 1.0000 | 0.9883 | the same model scored only against failures that got the full shift |

## The sub-die channel on its own

Both numbers below come from re-running the generator's own block process (`tuned/blocksim.py`), which is the only way to ask what a detector *could* have reached.

- matched filter evaluated at the **true** cluster seed: **0.8606** ROC-AUC
- the same filter maximised over all 2,000 seeds: 0.7061
- this pipeline's likelihood-ratio bank: **0.7420**
- `mrf`'s scan bank on the same dies: 0.7179

The seed is drawn uniformly per die and correlates with nothing, so the gap between the last row and the first is not a modelling failure -- it is the price of searching 2,000 positions.

A 1-D convolutional network was trained on the raw readings to check whether the derived statistic misses anything (`tuned/blockcnn.py`, GPU):

| detector | ROC-AUC | AP | seconds |
|---|--:|--:|--:|
| derived likelihood ratio | 0.7559 | 0.2929 | 116 |
| 1-D CNN | 0.7423 | 0.2760 | 603 |
| both together | 0.7497 | | |

The network does not beat the closed form, and the two together are worse than the closed form alone, so the network is not carrying anything the derivation missed.

## Diagnostics

- `wafer_rate_recovery_r`: **0.9466**
- `direction_recovery_r`: **0.9985**
- `oracle_score_auc`: **0.8541**
- `fitted_score_auc`: **0.8547**
- `block_score_auc`: **0.7461**
- `block_score_ap`: **0.1600**
- `marginal_fraction_of_failures`: **0.6461**
- `hazard_reconstruction_r`: **1.0000**
- `prior_scale`: **0.0199**
- `overall_rate`: **0.0218**
- `wafer_fraction_recovery_r`: **0.9884**
- `wafer_fraction_mean_ratio`: **0.9988**
- `wafer_rate_mean_ratio`: **1.0363**
- `mrf_wafer_fraction_recovery_r`: **0.9670**

## Selected constants

Chosen on five wafer-grouped folds of the training wafers only (`tuned/select.py`), before the held-out wafers were scored.

- `lam_smooth` = 20.0 (grid [20.0, 2000.0, 50000.0])
- `n_bases` = 6.0 (grid [6, 10])
- `block_C` = 0.02 (grid [0.02, 0.1])

## Figures

- `results/tuned_figures/channel_shapes.png`
- `results/tuned_figures/wafer_rate_recovery.png`
- `results/tuned_figures/precision_recall.png`
- `results/tuned_figures/ceiling.png`
- `results/tuned_figures/block_channel.png`
- `results/tuned_figures/hazard_shape.png`
