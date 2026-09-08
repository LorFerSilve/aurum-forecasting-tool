# Verificatie fase 8 — compact multi-timeframe neuraal kernmodel

**Protocol:** `phase8-v1`  
**Canonieke run:** `20260908T020044407464Z-24f4c0d4`  
**Benchmarkcode:** `75945fe70606c8200cebc66678d0e220db5fb0ad`  
**Besluit:** `keep_phase7_champion_all_horizons`

## Integriteit en runtime

De formele run eindigde met status `succeeded` na **1u 45m 26s**. De lokale
`phase8 validate` controleerde **607 bestanden** en accepteerde completion-versie
`sha256:9bce4a80c7e0d65fc40ccd4e1fea3c1ef2d56022f8b60720f3d1a8b54cfbb922`.

De finale holdout bleef gesloten (`holdout_opened=false`). De benchmark gebruikte
dataversie
`sha256:1c8f79be07d8e6162c7ce7b038a67b950320813f647366528c5064c894379254`
en de bevroren Phase-7 reference-run
`20260907T014255645674Z-b2afe281`.

Formele lokale runtime:

- PyTorch: `2.11.0+cu128`;
- CUDA runtime: `12.8`;
- device: `NVIDIA GeForce RTX 3080`;
- deterministic algorithms: ingeschakeld;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- NumPy `2.4.6`;
- pandas `2.3.3`;
- SciPy `1.17.1`;
- scikit-learn `1.9.0`;
- PyArrow `23.0.1`.

De run is uitgevoerd op commit `75945fe`. De twee latere Phase-8 commits op de
feature branch wijzigen alleen PyTorch-2.11 typing/packaging en de platform-specifieke
wheel-lock/CI-installatie. Ze veranderen de modelarchitectuur, labels, folds, losses,
optimizer, seeds, predictionlogica of promotiepoort niet en vereisen daarom geen nieuwe
marktbenchmark.

## Preflight en samplepariteit

De structurele en device-preflight is volledig geslaagd vóór de benchmark. Voor alle
acht horizons en alle drie outer folds kwam de `sample_id` digest exact overeen met
de bevroren Phase-7 reference. De outer jaren bleven 2022, 2023 en 2024 en het
trainingszijdige gap/embargo bleef **181 minuten**.

De echte-data smokefits voor zowel de korte `1min/3min/15min` set als de lange
`3min/15min/1h` set gebruikten CUDA mixed precision, bleven binnen **15.241
parameters**, voerden elk 16 optimizerupdates uit en hadden 0 AMP-skips.

## Predictieve uitkomst

Lagere Brier, log loss en return-MAE zijn beter; hogere macro-F1 is beter.

| Horizon | Neural macro-F1 | Phase-7 macro-F1 | Δ macro-F1 | Δ worst-F1 | Δ Brier | Δ log loss | Δ return-MAE bps | Besluit |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3m | 0.326440 | 0.428295 | -0.101855 | -0.102581 | -0.244440 | -0.437432 | -0.009190 | keep Phase 7 |
| 6m | 0.321088 | 0.435966 | -0.114878 | -0.118068 | -0.206623 | -0.340802 | -0.004721 | keep Phase 7 |
| 9m | 0.324036 | 0.434792 | -0.110756 | -0.119628 | -0.157376 | -0.248257 | -0.007031 | keep Phase 7 |
| 12m | 0.326712 | 0.429891 | -0.103180 | -0.110823 | -0.125524 | -0.193109 | -0.007693 | keep Phase 7 |
| 15m | 0.319270 | 0.425052 | -0.105783 | -0.114531 | -0.101951 | -0.154299 | +0.002924 | keep Phase 7 |
| 30m | 0.337333 | 0.404617 | -0.067284 | -0.069571 | -0.034376 | -0.050299 | -0.012258 | keep Phase 7 |
| 60m | 0.360939 | 0.374892 | -0.013953 | -0.014245 | -0.002436 | -0.003409 | +0.022259 | keep Phase 7 |
| 180m | 0.288102 | 0.332036 | -0.043934 | -0.033284 | -0.022067 | -0.037170 | +0.174437 | keep Phase 7 |

**0/8 horizons** passeren predictive admission. Hoewel de neural probabilities op alle
horizons een lagere gemiddelde Brier en log loss hebben, is mean én worst-fold
macro-F1 overal slechter dan de frozen Phase-7 champion. De vooraf vastgelegde gate
vereist verbetering van directional discrimination zonder regressie in
probability quality; alleen betere probability scores zijn dus onvoldoende.

De sterkste neural horizon is relatief gezien **60m**, maar ook daar blijft
macro-F1 ongeveer 0,014 onder Phase 7 en worst-fold macro-F1 ongeveer 0,014 lager.

## Classgedrag

De lage macro-F1 komt niet door een onstabiele optimizer, maar door zwakke
klassendiscriminatie. Gemiddeld over de outer folds:

| Horizon | Werkelijk neutral | Voorspeld neutral |
|---:|---:|---:|
| 3m | 90,27% | 99,75% |
| 6m | 81,31% | 99,02% |
| 9m | 74,40% | 97,54% |
| 12m | 69,05% | 95,72% |
| 15m | 64,72% | 94,89% |
| 30m | 51,12% | 85,34% |
| 60m | 37,98% | 63,50% |
| 180m | 21,36% | 0,37% |

Voor 3–60m concentreert het model zich te sterk op neutral; bij 180m slaat dit om en
wordt neutral vrijwel nooit voorspeld. De goede Brier/log-loss-score op vooral korte
horizons weerspiegelt daardoor grotendeels sterke probability fit op de dominante
klasse, niet betere directionele scheiding.

## Economische uitkomst

**0/8 horizons** produceren een economic promotion candidate. Voor iedere horizon en
iedere outer fold selecteert de inner policy-search de expliciete cash/no-trade
fallback:

- `confidence_threshold=1.0`;
- `min_expected_net_bps=1e12`;
- 0 uitgevoerde trades;
- 0 exposure;
- 0 base net bps;
- 0 stress net bps.

Er wordt dus geen tradingchampion gecreëerd.

## Trainingstabiliteit

Over de volledige benchmark zijn **144 neural fits** uitgevoerd. De training registreert
in totaal **126.614 geldige optimizerupdates** en **10 recoverable AMP-skips**
(**0,0079%** van het aantal geldige updates). Alle tien skips zitten op de 3m-horizon,
telkens in epoch 1, en worden correct door GradScaler afgehandeld. Geen history bevat
NaN of oneindige loss/metricwaarden.

De geselecteerde finale epochcounts variëren per fold en seed, zoals verwacht bij
inner-fold early stopping. De parametercount blijft in alle fits 15.241, ruim onder
het maximum van 150.000.

## Coverage

Alle samples behouden de 3min-sequence. De 15min availability ligt rond 93–94%.
Voor de lange horizons ligt 1h availability rond 79% en daalt ze voor 180m tot ongeveer
76,6%. Missing hogere timeframes worden niet geforwardfilld; de gated fusion gebruikt
de availability-maskering zoals vooraf vastgelegd.

De eligible labelset daalt logisch met de horizon doordat een volledig toekomstpad
vereist is: van 531.882 samples op 3m naar 450.193 op 180m. Dit verandert de frozen
outer-pariteit niet: per horizon wordt exact dezelfde eligible universe als Phase 7
gebruikt.

## Exitbesluit

Phase 8 voldoet aan haar technische en empirische exitcriteria. De compact GRU-core
traint stabiel, reproduceerbaar en binnen budget, maar **levert geen betere
directionele predictor dan de frozen Phase-7 champions**.

Daarom:

- blijven alle acht Phase-7 champions actief;
- wordt geen neural tradingmodel gepromoveerd;
- blijft release `v0.2.0` de betrouwbare researchbaseline;
- mag de neural core uitsluitend als researchvariant worden meegenomen naar Phase 9,
  waar rijkere future-candle-path supervision wordt onderzocht;
- de finale 2025+ holdout blijft gesloten.

Een negatieve Phase-8-uitkomst is hier het correcte onderzoeksresultaat: extra
modelcomplexiteit wordt niet gepromoveerd zonder vooraf vereiste evidence.
