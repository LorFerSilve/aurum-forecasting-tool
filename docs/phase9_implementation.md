# Fase 9 — implementatiestatus

**Protocol:** `phase9-v1`  
**Branch:** `feature/phase-9-future-path`  
**Empirische status:** canonieke benchmark afgerond en gevalideerd; geen championpromotie, direct path behouden als researchoutput

## Reeds geïmplementeerd

- exact vijf-candle 3min path-labelcontract;
- fail-closed segment/gap- en holdoutguards;
- lossless log-bps representatie voor gap/body/upper/lower wick;
- opslag van echte OHLC en timestamps per future step;
- exacte 15min aggregatie van de vijf candles;
- NumPy-reconstructie met gegarandeerde positieve OHLC-invarianten;
- herbruikbare fused phase-8 representation zonder gedrag van de phase-8 outputs te wijzigen;
- direct multi-step future-path quantile head;
- free-running recursive one-step baseline met eigen q50-output als volgende decoder-input;
- q10/q50/q90 ordering by construction;
- niet-negatieve wickquantielen;
- afzonderlijke directe 15min-candle quantile head;
- differentiable Torch-reconstructie;
- pinball loss;
- temporal-consistency Huber loss;
- hard phase-9 parameterbudget;
- train-only path- en aggregate-targetscaling;
- common 15min sample-universe die executable direction/return-labels en path-labels exact joint;
- behoud van dezelfde 15min `sample_id`-semantiek voor latere baseline-pariteit;
- Phase-9 train/predict engine met AdamW, deterministic seeds, AMP-overflowherstel,
  gradient clipping en early stopping op validation macro-F1;
- gecombineerde direction/return/range/volatility/path/aggregate/cumulative/consistency losses;
- atomaire Phase-9 checkpoints met beide normalizers en AMP/optimizer-auditmetadata;
- path-evaluatie voor q50-MAE, q10-q90 coverage, intervalbreedte,
  reconstructed-OHLC-fout en cumulatieve 15min-returnfout;
- expliciete UTC-dtypes voor alle future-path timestamps;
- frozen 2022/2023/2024 nested walk-forward orchestration met hetzelfde 181-minuten-gap;
- direct-vs-recursive vergelijking op exact dezelfde outer rows;
- atomaire fold-artifacts voor predictions, pathquantielen, backtests, histories en checkpoints;
- run-level config/schedule/runtime/dependency/reference snapshots;
- completion manifest met SHA-256 voor alle persisted evidence;
- `phase9 validate` met protocol-, holdout-, outer-year-, gap- en digestguards;
- synthetische end-to-end dry run via `phase9 dry-run`;
- frozen-reference common-sample subset zonder refit of policy-reselectie;
- cryptografisch geverifieerde loader voor de canonieke Phase-7/Phase-8 runs;
- expliciete Phase-7 én Phase-8 run/code/completion-pins in `phase9-v1`;
- fail-closed loading van uitsluitend manifest-authenticated 15min outer predictions;
- behoud van de reeds geselecteerde frozen reference-policies;
- real-data `phase9 preflight` voor data-identiteit, folds, referencepariteit,
  direct/recursive smokefits, quantile-invarianten en checkpoint-integriteit;
- canonieke preflight op commit `6e46605` geslaagd met 525.967 common eligible rows,
  CUDA/RTX 3080, beide varianten en gesloten holdout;
- formele `phase9 run`-orchestration achter een opnieuw uitgevoerde fail-closed preflight;
- run-level aggregatie versus frozen Phase-7/8 common-universe references zonder
  automatische post-hoc promotiedrempel;
- geaggregeerde path high/low/range-fout en path-vs-direct-15min consistency-evidence;
- future-mutation guard: future labels veranderen, historische sequences niet;
- unit tests voor targettiming, gaps, holdout, reconstructie, quantielen, losses,
  train-only scaling, common universe, training, reproduceerbaarheid, orchestration,
  artifacts, dry-run, referencepariteit en budget.

## Formele benchmarkstatus

De canonieke run `20260908T211301827616Z-ad6b573c` is voltooid en met
`phase9 validate` integraal geverifieerd: **225 bestanden**, completion
`sha256:a39fb334233a27c5fea65d1f4269ce3134f24bf9d89c2530b8998f5465a0a776`.

Het formele besluit is:

- Phase-7 `mvp/logistic` blijft de actieve 15m researchchampion;
- Phase-9 direct wordt niet als champion/tradingmodel gepromoveerd;
- de direct path-head mag uitsluitend als optionele research/distributionele output
  worden behouden;
- recursive wordt niet als standaardarchitectuur meegenomen;
- alle inner-selected policies blijven cash/no-trade;
- de finale 2025+ holdout blijft gesloten.

Zie `docs/phase9_verification.md` voor de volledige empirische analyse.

Na de benchmark is in commit `991f5c3` uitsluitend de persisted-configvalidator
gecorrigeerd voor JSON-keyvolgorde. Dit verandert de benchmarkmethodologie of resultaten
niet en vereist geen nieuwe marktbenchmark.

Tijdens de pre-merge review is ook de cumulative path-return evaluatie gecorrigeerd:
de metric gebruikt nu het gereconstrueerde vijf-candle q50-pad in plaats van de directe
aggregate head. De bestaande canonical run hoefde niet opnieuw getraind te worden, omdat
alle benodigde pathquantielen al cryptografisch geverifieerd waren opgeslagen. De
post-benchmark audit geeft mean MAE **6.007010 bps** voor direct en **6.001428 bps** voor
recursive. Dit wijzigt het Phase-9 promotiebesluit niet.

PR #3 is in `main` gemerged als `9a15e04`. De onafhankelijke hercontrole op
2026-09-09 verifieerde opnieuw 225 artefacten en reproduceerde het path-returnauditrapport.

## Monte Carlo-roadmapbesluit

Het document `docs/monte_carlo_integration_proposal.md` is op 2026-09-08 formeel
geaccepteerd en in de hoofdroadmap verwerkt. Dit **wijzigt de bevroren Phase-9 benchmark
niet**: de huidige q10/q50/q90 future-path outputs vormen de distributionele precursor.

Nieuwe uncertainty-contracts en calibratie worden in Phase 11 gebouwd; predictive
Monte Carlo wordt in Phase 12 als decision-policy challenger geëvalueerd; strategy
Monte Carlo/block bootstrap wordt in Phase 13 gebruikt voor robuustheids- en
stresstesting vóór de finale holdout.
