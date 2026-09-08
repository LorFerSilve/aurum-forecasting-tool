# Fase 9 — implementatiestatus

**Protocol:** `phase9-v1`  
**Branch:** `feature/phase-9-future-path`  
**Empirische status:** nog geen formele Phase-9 benchmark; canonieke Phase-8 reference is frozen

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
- future-mutation guard: future labels veranderen, historische sequences niet;
- unit tests voor targettiming, gaps, holdout, reconstructie, quantielen, losses,
  train-only scaling, common universe, training, reproduceerbaarheid, orchestration,
  artifacts, dry-run, referencepariteit en budget.

## Bewust nog niet geopend

Er is nog geen formele `phase9 run` CLI en geen empirische Phase-9 promotion decision.
De canonieke Phase-8 reference is nu frozen in config/protocol. De train/predict-,
recursive baseline-, walk-forward-, artifact-, validator- en common-sample bouwstenen
bestaan al en de synthetic dry run oefent de volledige keten.

De frozen-reference loader en de volledige preflightcode zijn nu aangesloten, maar de
canonieke real-data preflight moet nog lokaal worden uitgevoerd tegen de grote Phase-7-
en Phase-8-runmappen. Alleen een volledig geslaagde `phase9 preflight` mag de laatste
technische blokkade voor het formele `phase9 run`-commando opheffen.

PR #3 blijft tot na de canonical benchmark draft. De finale 2025+ holdout blijft
gesloten en Phase 8 blijft methodologisch frozen.
