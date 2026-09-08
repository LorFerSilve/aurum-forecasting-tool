# Fase 8 — implementatiestatus

**Protocol:** `phase8-v1`  
**Release:** blijft `v0.2.0` totdat fase 8–10 samen `v0.3` rechtvaardigen  
**Empirische status:** formele 2020–2024 benchmark geverifieerd; 0/8 neural promotions

## Geïmplementeerd

- compacte single-layer GRU-encoder per geselecteerd timeframe;
- availability-aware gated fusion;
- direction-, return-, range- en realized-volatility-heads;
- hard parameterbudget;
- causale fixed-length candle sequences met gap/stalenessguards;
- expliciete ontbrekende-timeframe-maskering zonder forward fill;
- train-only sequence-imputatie, normalisatie en targetscaling;
- mini-batch AdamW-training, Huber auxiliary losses, gradient clipping en NaN-guards;
- inner-fold early stopping op macro-F1;
- twee vooraf bevroren seeds en outer seed-ensemble;
- CUDA auto-detectie en mixed precision op CUDA;
- deterministische seed/configuratie;
- AMP-overflowherstel via GradScaler met gelogde skipped optimizer steps;
- preflight van alle phase-7 metric/digest-contracten voor alle horizons;
- echte-data CUDA/CPU-smokefits voor zowel korte als lange timeframe-sets vóór de benchmark;
- atomaire neural checkpoints met optimizer/AMP-auditmetadata;
- synthetische intentional-overfit-, train-only-normalisatie- en same-seed CPU-tests;
- exact outer sample-digestcontract tegen de bevroren phase-7 run;
- hergebruik van de bestaande policy- en base/stress-backtestlogica;
- per-seed histories, checkpoints, fusion weights, training/inferencetijden en metrics;
- `phase8 run` / `phase8 validate` CLI;
- clean-Git, dependency-lockpreflight en gesloten-holdoutguards.

## Bewuste v1-beperkingen

De eerste neural baseline gebruikt alleen `1min`, `3min`, `15min` en `1h`. `3h`
wordt niet meegenomen wegens de zeer lage fase-7-dekking; `5min` en `30min` worden
uitgesteld om het initiële model klein en interpreteerbaar te houden. Er is nog geen
Transformer, focal loss, OOF-calibrator, ensemble met klassieke modellen of
future-candle-path. Die complexiteit mag pas na bewijs worden toegevoegd.

## Empirische gate

De canonieke Phase-8 run is
`20260908T020044407464Z-24f4c0d4` op benchmarkcommit `75945fe`.
`phase8 validate` verifieerde 607 artifacts onder `phase8-v1`; de finale holdout
bleef gesloten. Alle acht horizons behouden de frozen Phase-7 champion. Geen horizon
passeert predictive admission of economic promotion.

De neural probabilities verbeteren Brier en log loss op alle horizons, maar mean en
worst-fold macro-F1 zijn overal slechter. Alle geselecteerde policies vallen terug op
cash/no-trade. Zie [het formele verificatierapport](phase8_verification.md) voor de
volledige runtime-, fold-, class-discriminatie- en trainingaudit.

## Reproduceerbaarheidsnotitie

Formele Phase-8 runs weigeren voortaan vóór dataopbouw of training wanneer de
geïnstalleerde kernpackages (`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`,
`pyarrow`) niet overeenkomen met `requirements.lock`. CUDA-local buildtags zoals
`+cu128` zijn toegestaan zolang de publieke PyTorch-versie gelijk is aan de lock.


## Frozen PyTorch runtime

Phase-8 formal runs use public package version `torch==2.11.0`. CI may use the
CPU wheel. NVIDIA development runs use the matching official CUDA 12.8 wheel
`torch==2.11.0+cu128`; the local `+cu128` build tag is accepted by the dependency
guard because the frozen public version remains 2.11.0.


## PyTorch wheel split

The public neural runtime version is frozen separately in
`requirements-neural.lock` as `torch==2.11.0`. The general
`requirements.lock` deliberately does not install PyTorch itself because the
correct wheel source is platform-specific. GitHub CI installs the official CPU
wheel; NVIDIA Windows research runs use the matching official
`2.11.0+cu128` wheel. Formal runtime validation compares the public version
and permits the CUDA local build tag.
