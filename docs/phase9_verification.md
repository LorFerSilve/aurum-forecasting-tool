# Verificatie fase 9 — directe vijf-candle future path

**Protocol:** `phase9-v1`  
**Canonieke run:** `20260908T211301827616Z-ad6b573c`  
**Benchmarkcode:** `d94763e42d87063b41e4192c64bab46209cbb4b9`  
**Completion manifest:** `sha256:a39fb334233a27c5fea65d1f4269ce3134f24bf9d89c2530b8998f5465a0a776`  
**Validator:** 225 bestanden geverifieerd  
**Besluit:** `keep_phase7_champion_retain_direct_path_research_only`

## Integriteit

De formele Phase-9 benchmark is voltooid en daarna integraal gevalideerd. De run gebruikt
`phase9-v1`, de frozen outer jaren 2022/2023/2024, twee vooraf vastgelegde seeds en
dataversie
`sha256:41c618e346b7be3aed05ffdac634076f0dde422387ecdf9529b0c6702d0ebbfc`.

De finale 2025+ holdout bleef gesloten (`holdout_opened=false`).

De run gebruikt exact de frozen references:

- Phase 7: `20260907T014255645674Z-b2afe281`;
- Phase 8: `20260908T020044407464Z-24f4c0d4`.

De benchmark bevat **525.967** common eligible samples. De drie frozen outer testsets
behouden exact hun vooraf geverifieerde sample-digests.

Na de benchmark is in commit `991f5c3` uitsluitend een validatorbug opgelost waarbij
gesorteerde JSON-keyvolgorde ten onrechte als inhoudelijk configverschil werd behandeld.
Die wijziging verandert geen model, data, weights, labels, folds, losses, seeds, predictions
of benchmarkresultaten en vereist daarom geen nieuwe marktbenchmark.

Tijdens de pre-merge review is daarnaast vastgesteld dat de oorspronkelijk gerapporteerde
`cumulative_15m_return_mae_bps` uit de directe aggregate head kwam in plaats van uit het
gereconstrueerde vijf-candle q50-pad. De modeloutputs zelf waren correct en volledig
opgeslagen. Daarom is de metric zonder retraining opnieuw berekend uit de geverifieerde
`path_quantiles.parquet`-artifacts op exact dezelfde sample IDs. De audit gebruikte de
canonieke completion
`sha256:a39fb334233a27c5fea65d1f4269ce3134f24bf9d89c2530b8998f5465a0a776`,
opende de holdout niet en voerde geen training uit.

## Hoofdvergelijking

Lagere Brier, log loss en MAE zijn beter; hogere macro-F1 is beter.

| Variant | Mean macro-F1 | Worst macro-F1 | Mean Brier | Mean log loss | Mean return-MAE bps | Mean calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Phase-7 frozen champion | 0.425044 | 0.423567 | 0.579089 | 0.984360 | 6.008752 | 0.152802 |
| Phase-8 frozen neural | 0.319263 | 0.309039 | 0.477148 | 0.830081 | 6.011680 | 0.012743 |
| Phase-9 direct | 0.321661 | 0.305940 | 0.476759 | 0.829424 | 6.007854 | 0.009590 |
| Phase-9 recursive | 0.315652 | 0.305154 | 0.476740 | 0.829292 | 6.009282 | 0.010433 |

### Direct versus Phase 7

Phase-9 direct verbetert de probability scores sterk tegenover de klassieke Phase-7
champion, maar verliest duidelijk op directionele discriminatie:

- mean macro-F1: **-0.103383**;
- worst-fold macro-F1: **-0.117627**;
- mean Brier: **-0.102330**;
- mean log loss: **-0.154936**;
- mean return-MAE: **-0.000898 bps**.

De hogere gewone accuracy van Phase 9 verandert dit besluit niet. De 15m-dataset is
sterk neutral-dominant en Phase-9 direct voorspelt de neutral klasse zeer vaak; balanced
accuracy en macro-F1 laten daarom beter zien dat de klassieke champion de drie klassen
duidelijk beter onderscheidt.

### Direct versus Phase 8

De future-path supervision verandert de bestaande neural direction-head slechts marginaal:

- mean macro-F1: **+0.002397**;
- worst-fold macro-F1: **-0.003099**;
- mean Brier: **-0.000390**;
- mean log loss: **-0.000657**;
- mean return-MAE: **-0.003826 bps**.

Per fold is de macro-F1-delta versus Phase 8 ongeveer **+0.00550 / -0.00310 / +0.00479**.
Dat is geen consistente worst-fold verbetering en dus onvoldoende om de path-head in de
champion te promoveren.

## Direct versus recursive

De direct multi-step variant blijft de voorkeursvariant wanneer path-output voor research
nodig is.

Direct heeft:

- hogere mean macro-F1: **0.321661 vs 0.315652**;
- betere aggregate high-MAE: **4.656 vs 4.881 bps**;
- betere aggregate low-MAE: **5.025 vs 5.126 bps**;
- betere aggregate range-MAE: **9.002 vs 9.514 bps**;
- betere mean path/aggregate q10-q90 coverage;
- duidelijk betere path-vs-direct range-consistency:
  **0.919 vs 1.263 bps**.

Recursive heeft een vrijwel identieke Brier/log-loss, een minimaal lagere gemiddelde
per-step median-MAE en na de post-benchmark audit ook een iets betere echte cumulative
five-step path-return MAE (**6.001428 vs 6.007010 bps**). Dat voordeel is echter klein en
weegt niet op tegen de slechtere directionele macro-F1, aggregate high/low/range-fout,
coverage en path-vs-direct consistency. Er is daarom nog steeds geen empirische reden om
recursive als standaardarchitectuur mee te nemen.

## Distributionele pathkwaliteit

Voor de direct variant:

- mean per-component q10-q90 coverage: **78.92%**;
- mean aggregate-component q10-q90 coverage: **79.75%**;
- nominale targetcoverage: **80%**;
- mean per-component interval width: **3.575 log-bps**;
- mean aggregate interval width: **7.858 log-bps**;
- mean per-component median-MAE: **1.138 log-bps**;
- mean aggregate-component median-MAE: **2.539 log-bps**.

De coverage ligt dus dicht bij de nominale 80% en is niet alleen het gevolg van
onbegrensd brede intervallen. De directe path-head levert daarmee een bruikbare
**research-distributie**, ook al levert hij geen bewezen directionele of economische
promotie.

Fysieke high/low/range intervalcoverage wordt bewust niet geclaimd: marginale
componentquantielen definiëren geen joint path-distributie. Een dergelijke claim wordt
pas verantwoord na distributionele calibratie / sampling in latere fasen.

## Pathgeometrie en directionele bruikbaarheid

Directe q50 path-reconstructie geeft gemiddeld:

- aggregate high-MAE: **4.656 bps**;
- aggregate low-MAE: **5.025 bps**;
- aggregate close-MAE: **6.007 bps**;
- aggregate range-MAE: **9.002 bps**;
- corrected cumulative five-step 15m return-MAE: **6.007010 bps**.

De path- en directe 15m-head blijven onderling redelijk coherent:

- close consistency MAE: **0.256 bps**;
- high consistency MAE: **0.392 bps**;
- low consistency MAE: **0.550 bps**;
- range consistency MAE: **0.919 bps**.

Maar de q50 path zelf is geen sterke directionele classifier:

- path-implied mean accuracy: **0.647838**;
- path-implied mean macro-F1: **0.261864**.

De path-output moet daarom niet als vervanging van de bestaande direction-head worden
gebruikt.


## Post-benchmark path-return audit

De gecorrigeerde cumulative five-step q50 path-return MAE is rechtstreeks uit de
persisted pathquantielen berekend:

| Fold | Direct bps | Recursive bps |
|---|---:|---:|
| 2022 | 6.627445 | 6.624825 |
| 2023 | 5.147704 | 5.141810 |
| 2024 | 6.245882 | 6.237647 |
| **Mean** | **6.007010** | **6.001428** |
| **Worst** | **6.627445** | **6.624825** |

De correctie ten opzichte van de eerder opgeslagen aggregate-head metric is klein:
direct verandert per fold met ongeveer +0.00234 / +0.00301 / +0.00190 bps; recursive
met -0.00683 / -0.00394 / -0.00143 bps. De audit verandert daarom geen champion- of
economisch promotiebesluit.

Het auditrapport staat in
`reports/phase9_path_return_audit_20260908T211301827616Z-ad6b573c.json`.

## Economische uitkomst

Zowel direct als recursive selecteren in alle drie outer folds de expliciete
cash/no-trade fallback:

- `confidence_threshold=1.0`;
- `min_expected_net_bps=1e12`;
- 0 trades;
- 0 exposure;
- 0 base net bps;
- 0 stress net bps.

Er is dus **geen economic promotion candidate** en geen tradingchampion.

## Formeel promotiebesluit

Phase 9 voldoet aan haar technische exitcriteria, maar niet aan de voorwaarden om
future-path learning in de actieve champion te promoveren.

Daarom:

1. **Phase-7 `mvp/logistic` blijft de actieve 15m researchchampion.**
2. Phase-9 direct wordt **niet** als champion of tradingmodel gepromoveerd.
3. De direct path-head mag als **optionele research/distributionele output** worden
   behouden voor latere OOF/distribution-calibration en Monte Carlo-onderzoek.
4. De recursive baseline wordt niet als standaardarchitectuur meegenomen.
5. Latere fasen mogen de Phase-9 path-output niet behandelen als bewezen predictive
   edge; iedere downstream toepassing moet opnieuw haar eigen toelatingspoort halen.
6. De finale 2025+ holdout blijft gesloten.
7. Geen Phase-9 resultaat activeert paper of live trading.

Dit is een gecontroleerde, grotendeels negatieve promotie-uitkomst met één bruikbaar
nevenresultaat: de direct path-head produceert redelijk gekalibreerde marginale
distributionele outputs die als researchinput behouden mogen blijven.
