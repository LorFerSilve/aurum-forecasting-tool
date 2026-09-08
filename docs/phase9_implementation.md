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
- unit tests voor targettiming, gaps, holdout, reconstructie, quantielen, losses en budget.

## Bewust nog niet geopend

Er is nog geen formele `phase9 run` CLI en geen empirische promotion decision. Eerst moet
de canonieke phase-8 benchmark volledig gevalideerd en als expliciete reference vastgezet
worden. Daarna worden training/evaluation, recursive one-step baseline en de formele
walk-forward benchmark toegevoegd.

De lokale `codex/phase9-path-research` branch van een andere agent bestaat momenteel
niet op GitHub. De bijbehorende stash blijft daarom apart bewaard en wordt na de lopende
phase-8 run tegen deze remote branch gereviewd voordat iets ervan wordt overgenomen.
