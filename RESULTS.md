# Results

All cross-validation numbers use five wafer-grouped folds repeated three times, the fold builder and metric code from `modeling/validation.py`, and the same seed-42 dataset the earlier baseline was scored on. Only dies with `old_label == 0` are fitted or scored.

## Cross-validation, all experiments

| experiment | features | AP | ROC-AUC | fail F1 | precision | recall |
|---|--:|--:|--:|--:|--:|--:|
| **Model B, stacked + wafer rate** | 541 | 0.6209 | 0.9132 | 0.5777 | 0.7909 | 0.4550 |
| Model B, stacked | 541 | 0.5863 | 0.8904 | 0.5596 | 0.7921 | 0.4326 |
| Model B, flat logistic | 541 | 0.5732 | 0.8829 | 0.5525 | 0.8023 | 0.4214 |
| ablation: per-wafer detrending | 551 | 0.5657 | 0.8770 | 0.5503 | 0.8697 | 0.4025 |
| **Model A, stacked + wafer rate** | 514 | 0.5524 | 0.8844 | 0.5323 | 0.8532 | 0.3869 |
| ablation: gradient boosting | 541 | 0.5430 | 0.8674 | 0.5333 | 0.8065 | 0.3984 |
| Model A, stacked | 514 | 0.5238 | 0.8526 | 0.5282 | 0.8883 | 0.3758 |
| die measurements alone | 500 | 0.5128 | 0.8422 | 0.5268 | 0.9408 | 0.3659 |
| Model A, flat logistic | 514 | 0.5116 | 0.8419 | 0.5258 | 0.9118 | 0.3694 |
| block readings alone | 27 | 0.1372 | 0.7259 | 0.1962 | 0.1484 | 0.2895 |

## Against the published baseline (cross-validation)

| model | baseline AP | this AP | gain | baseline F1 | this F1 | gain |
|---|--:|--:|--:|--:|--:|--:|
| Model A | 0.5137 | 0.5524 | +0.0387 | 0.5268 | 0.5323 | +0.0055 |
| Model B | 0.5758 | 0.6209 | +0.0451 | 0.5537 | 0.5777 | +0.0240 |

## Held-out test wafers (40 wafers never used for fitting)

| model | stage | AP | ROC-AUC | fail F1 | precision | recall | accuracy |
|---|--:|--:|--:|--:|--:|--:|--:|
| model_a | uncalibrated | 0.5406 | 0.8706 | 0.5308 | 0.9102 | 0.3746 | 0.9720 |
| model_a | wafer_rate_calibrated | 0.5730 | 0.8914 | 0.5348 | 0.8655 | 0.3870 | 0.9715 |
| model_b | uncalibrated | 0.6024 | 0.9062 | 0.5631 | 0.7989 | 0.4348 | 0.9714 |
| model_b | wafer_rate_calibrated | 0.6302 | 0.9210 | 0.5836 | 0.7787 | 0.4667 | 0.9718 |

Baseline holdout for reference: Model A AP 0.5320 / F1 0.5297, Model B AP 0.5934 / F1 0.5600.

## Diagnostics

- Correlation between the fitted parametric coefficients and the generator's own `fail_shift / base_std`: **0.995**
- Correlation between each held-out wafer's recovered failure rate and its actual rate, using no labels: **0.895**

## Figures

- `block_pattern.png`
- `global_importance.png`
- `model_a_to_b_delta.png`
- `precision_recall.png`
- `wafer_maps.png`
- `wafer_rate_recovery.png`
