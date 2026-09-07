# Phase-6 development benchmark

Protocol: `phase6-v1`. Data: `sha256:a8da6d0b9179304db0d135a39f0609b1e9dbd9232f98810f78b367fbcc73bcd0`.

Scope: full frozen comparison.

These are development comparisons, not an independent profitability claim. 2024 was already inspected during MVP research. The final 2025 holdout remains closed.

Every candidate uses identical records within a horizon/fold. Training and policy selection precede outer evaluation; calibration quarters remain unused. Probabilities are uncalibrated. Naive one-hot references express artificial certainty.

Net values below SUM arithmetic P&L bps on a fixed entry-bid notional across folds. They are NOT account-return percentages or compounded returns. Historical asks are 3-bps spread proxies; stress uses 4.5 bps with identical signals and 0.5 bps slippage per side. Realized-exit drawdown excludes intratrade mark-to-market losses.

A cash policy may legitimately produce zero trades. That is not demonstrated trading profitability. No model is activated for trading by this benchmark.

## Horizon 3 minutes

Eligible labels: 531,882/534,698; missing/incomplete paths dropped: 2,816.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4250 | 0.4234 | 0.4142 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4250 | 0.4233 | 0.4142 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.3162 | 0.3124 | 0.2081 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3241 | 0.3220 | 0.1616 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.3162 | 0.3124 | 0.1947 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.0310 | 0.0221 | 1.9024 | -1232739.01 | -1695887.16 | 308750 | -447941.91 |
| always_down | 0.0308 | 0.0224 | 1.9029 | -1237354.54 | -1700503.04 | 308750 | -450782.68 |
| last_candle_direction | 0.0641 | 0.0504 | 1.8892 | -1232266.25 | -1692063.86 | 306516 | -449100.00 |
| momentum | 0.3451 | 0.3407 | 0.7244 | -439505.86 | -604197.71 | 109789 | -177416.26 |
| mean_reversion | 0.3453 | 0.3432 | 0.7241 | -438839.52 | -603531.31 | 109789 | -173509.23 |
| empirical_priors | 0.3162 | 0.3124 | 0.1829 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 6 minutes

Eligible labels: 530,411/534,698; missing/incomplete paths dropped: 4,287.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4356 | 0.4338 | 0.4939 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4357 | 0.4338 | 0.4939 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2989 | 0.2929 | 0.3840 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3186 | 0.3165 | 0.2879 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.2989 | 0.2929 | 0.3738 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.0572 | 0.0446 | 1.8117 | -615065.44 | -846588.51 | 154341 | -223893.77 |
| always_down | 0.0565 | 0.0445 | 1.8145 | -619709.80 | -851233.22 | 154341 | -226374.09 |
| last_candle_direction | 0.1092 | 0.0910 | 1.8024 | -618094.43 | -848519.88 | 153609 | -224990.90 |
| momentum | 0.3744 | 0.3687 | 0.7822 | -267870.55 | -368261.72 | 66924 | -108055.05 |
| mean_reversion | 0.3767 | 0.3743 | 0.7800 | -267542.26 | -367933.41 | 66924 | -104785.39 |
| empirical_priors | 0.2989 | 0.2929 | 0.3247 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 9 minutes

Eligible labels: 528,940/534,698; missing/incomplete paths dropped: 5,758.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4344 | 0.4338 | 0.5353 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4344 | 0.4339 | 0.5353 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2842 | 0.2768 | 0.4887 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3165 | 0.3133 | 0.3739 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.2842 | 0.2768 | 0.5120 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.0762 | 0.0626 | 1.7412 | -409060.16 | -563306.88 | 102826 | -149175.11 |
| always_down | 0.0748 | 0.0614 | 1.7468 | -413579.59 | -567826.64 | 102826 | -151554.93 |
| last_candle_direction | 0.1398 | 0.1202 | 1.7351 | -413763.90 | -567406.55 | 102423 | -151996.58 |
| momentum | 0.3814 | 0.3746 | 0.8406 | -201117.07 | -276837.88 | 50478 | -80704.38 |
| mean_reversion | 0.3871 | 0.3831 | 0.8354 | -202722.55 | -278443.48 | 50478 | -79181.83 |
| empirical_priors | 0.2842 | 0.2768 | 0.4171 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 12 minutes

Eligible labels: 527,471/534,698; missing/incomplete paths dropped: 7,227.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4294 | 0.4276 | 0.5618 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4293 | 0.4276 | 0.5617 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2721 | 0.2632 | 0.5537 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3192 | 0.3156 | 0.4336 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.2721 | 0.2632 | 0.6190 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.0902 | 0.0754 | 1.6865 | -305068.10 | -420255.86 | 76788 | -111542.42 |
| always_down | 0.0882 | 0.0743 | 1.6945 | -309259.78 | -424447.85 | 76788 | -113627.10 |
| last_candle_direction | 0.1607 | 0.1406 | 1.6842 | -310297.21 | -425360.88 | 76705 | -114156.27 |
| momentum | 0.3824 | 0.3754 | 0.8903 | -166055.65 | -228155.89 | 41398 | -66293.89 |
| mean_reversion | 0.3888 | 0.3838 | 0.8838 | -165141.34 | -227241.52 | 41398 | -64215.40 |
| empirical_priors | 0.2721 | 0.2632 | 0.4786 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 15 minutes

Eligible labels: 526,005/534,698; missing/incomplete paths dropped: 8,693.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4251 | 0.4236 | 0.5791 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4251 | 0.4236 | 0.5791 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2617 | 0.2518 | 0.5968 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3259 | 0.3197 | 0.4769 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.2617 | 0.2517 | 0.7056 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.1007 | 0.0857 | 1.6434 | -243080.88 | -334978.48 | 61262 | -89077.96 |
| always_down | 0.0989 | 0.0850 | 1.6510 | -247034.29 | -338932.18 | 61262 | -91015.14 |
| last_candle_direction | 0.1770 | 0.1568 | 1.6419 | -246205.66 | -338057.00 | 61231 | -90452.93 |
| momentum | 0.3808 | 0.3737 | 0.9323 | -139668.99 | -192245.22 | 35049 | -56301.08 |
| mean_reversion | 0.3887 | 0.3840 | 0.9238 | -140734.27 | -193310.58 | 35049 | -53359.08 |
| empirical_priors | 0.2617 | 0.2517 | 0.5221 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 30 minutes

Eligible labels: 518,711/534,698; missing/incomplete paths dropped: 15,987.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.4013 | 0.3976 | 0.6203 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.4013 | 0.3976 | 0.6203 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2190 | 0.2009 | 0.6828 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3399 | 0.3289 | 0.5832 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.2251 | 0.2113 | 0.9775 | 0.00 | 0.00 | 0 | 0.00 |
| always_up | 0.1318 | 0.1168 | 1.5060 | -119378.01 | -164720.78 | 30227 | -44274.33 |
| always_down | 0.1296 | 0.1173 | 1.5165 | -122447.67 | -167790.66 | 30227 | -45753.92 |
| last_candle_direction | 0.2211 | 0.2039 | 1.5111 | -119664.32 | -164996.62 | 30220 | -44258.37 |
| momentum | 0.3653 | 0.3617 | 1.0749 | -83660.72 | -115544.81 | 21255 | -32521.25 |
| mean_reversion | 0.3733 | 0.3694 | 1.0647 | -86386.23 | -118270.53 | 21255 | -32857.34 |
| empirical_priors | 0.2251 | 0.2113 | 0.6219 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 60 minutes

Eligible labels: 504,370/534,698; missing/incomplete paths dropped: 30,328.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.3703 | 0.3606 | 0.6472 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.3703 | 0.3606 | 0.6472 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2524 | 0.2448 | 0.7028 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.3703 | 0.3638 | 0.6413 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.1808 | 0.1629 | 1.2528 | -21766.88 | -29846.28 | 5386 | -21766.88 |
| always_up | 0.1592 | 0.1464 | 1.3716 | -57317.29 | -79398.39 | 14720 | -21766.88 |
| always_down | 0.1561 | 0.1486 | 1.3881 | -60447.75 | -82529.09 | 14720 | -23273.50 |
| last_candle_direction | 0.2578 | 0.2455 | 1.3820 | -56000.29 | -78081.30 | 14720 | -20591.64 |
| momentum | 0.3365 | 0.3359 | 1.2237 | -47724.71 | -65740.60 | 12010 | -18356.99 |
| mean_reversion | 0.3468 | 0.3444 | 1.2088 | -48358.98 | -66374.92 | 12010 | -18276.26 |
| empirical_priors | 0.1808 | 0.1629 | 0.6651 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Horizon 180 minutes

Eligible labels: 450,193/534,698; missing/incomplete paths dropped: 84,505.

Decision: `keep_champion`; research reference: `reference`; promotion eligible for review: none.

| Model | Mean macro-F1 | Worst macro-F1 | Mean Brier | Total net bps | Stress net bps | Trades | Worst-fold net bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.3310 | 0.3164 | 0.6670 | 0.00 | 0.00 | 0 | 0.00 |
| logistic | 0.3309 | 0.3164 | 0.6670 | 0.00 | 0.00 | 0 | 0.00 |
| ridge | 0.2911 | 0.2831 | 0.6578 | 0.00 | 0.00 | 0 | 0.00 |
| xgboost | 0.2821 | 0.2683 | 0.6415 | 0.00 | 0.00 | 0 | 0.00 |
| most_frequent | 0.1889 | 0.1765 | 1.2082 | -18014.52 | -25250.88 | 4824 | -7339.06 |
| always_up | 0.1889 | 0.1765 | 1.2082 | -18014.52 | -25250.88 | 4824 | -7339.06 |
| always_down | 0.1871 | 0.1786 | 1.2191 | -20579.44 | -27815.99 | 4824 | -8850.00 |
| last_candle_direction | 0.2960 | 0.2894 | 1.2188 | -20855.52 | -28092.09 | 4824 | -9073.34 |
| momentum | 0.2828 | 0.2767 | 1.4142 | -17723.35 | -24095.79 | 4248 | -7486.33 |
| mean_reversion | 0.2880 | 0.2838 | 1.4055 | -16262.30 | -22634.63 | 4248 | -6488.37 |
| empirical_priors | 0.1889 | 0.1765 | 0.6465 | 0.00 | 0.00 | 0 | 0.00 |

Fold order: 2022, 2023, 2024.

- reference: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- logistic: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- ridge: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.
- xgboost: 2022: 0.00 bps / 0 trades; 2023: 0.00 bps / 0 trades; 2024: 0.00 bps / 0 trades.

## Artifact map

- `schedule.json`: frozen time blocks and common embargo.
- `horizon_H/model_table.parquet`: causal features and complete-path targets.
- `horizon_H/test_YEAR/split_audit.json`: identical train/test sample hashes.
- `.../MODEL/selection.json`: all inner candidate metrics and frozen decision policy.
- `.../MODEL/outer_predictions.parquet`: ordered predictions and source lineage.
- `.../MODEL/evaluation.json`: probability, error, session and execution metrics.
- `completion.json`: inventory and hashes; only a succeeded run is valid.
