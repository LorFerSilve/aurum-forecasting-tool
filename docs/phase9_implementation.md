# Fase 9 — implementatiestatus

**Protocol:** `phase9-v1`  
**Branch:** `feature/phase-9-future-path`  
**Empirische status:** nog geen formele benchmark; canonieke phase-8 reference wordt afgewacht

## Reeds geïmplementeerd

- exact vijf-candle 3min path-labelcontract;
- fail-closed segment/gap- en holdoutguards;
- lossless log-bps representatie voor gap/body/upper/lower wick;
- opslag van echte OHLC en timestamps per future step;
- exacte 15min aggregatie van de vijf candles;
- NumPy-reconstructie met gegarandeerde positieve OHLC-invarianten;
- herbruikbare fused phase-8 representation zonder gedrag van de phase-8 outputs te wijzigen;
- direct multi-step future-path quantile head;
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
- unit tests voor targettiming, gaps, holdout, reconstructie, quantielen, losses,
  train-only scaling, common universe, training, reproduceerbaarheid en budget.

## Bewust nog niet geopend

Er is nog geen formele `phase9 run` CLI en geen empirische promotion decision. Eerst moet
de canonieke phase-8 benchmark volledig gevalideerd en als expliciete reference vastgezet
worden. De train/predict- en evaluatiebouwstenen bestaan al, maar worden nog niet als
formele benchmark geëxposeerd. Daarna worden de outer walk-forward orchestration,
exacte Phase-8/Phase-7 common-sample comparison, recursive one-step baseline en de
formele run/validate-flow toegevoegd.

De lokale `codex/phase9-path-research` branch van een andere agent bestaat momenteel
niet op GitHub. De bijbehorende stash blijft daarom apart bewaard en wordt na de lopende
phase-8 run tegen deze remote branch gereviewd voordat iets ervan wordt overgenomen.
