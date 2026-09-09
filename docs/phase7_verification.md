# Verificatie fase 7 — richer price-only features

**Protocol:** `phase7-v2`  
**Run:** `20260907T014255645674Z-b2afe281`  
**Code:** `3f0a703568224fe9169b1e9f8d61dad131f0005b`  
**Besluit:** `freeze_v0.2_research_benchmark_no_trading_champion`

## Integriteit en reproduceerbaarheid

De formele run eindigde met status `succeeded` na **7u 49m 20s**. De lokale
`phase7 validate` controleerde **11.311 bestanden** en accepteerde completion-versie
`sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f`.
De finale holdout bleef gesloten (`holdout_opened=false`).

**Auditcorrectie 2026-09-08:** de eerder in Git vastgelegde completion-versie bevatte
één verkeerd overgenomen teken (`...7f45953c...`). De oorspronkelijke lokale
`completion.json` heeft CreationTime en LastWriteTime `2026-09-07 11:32:15` en de
volledige `phase7 validate` verifieert opnieuw alle 11.311 artifacts met de hierboven
vermelde versie `...7f45053c...`. Dit corrigeert uitsluitend repository-evidence; de
Phase-7 run, artifacts, resultaten en holdoutstatus zijn niet gewijzigd.

Dataversie:
`sha256:555458e7eafa7dbef51e624b22fc4867d41367e8a5b026d2dad42a085408663d`.  
Feature-spec:
`sha256:590b7ee697163f36431f0cb74fe4e3405629b08e283992111aba785757484a98`.

De benchmark gebruikte 534.698 MVP-geankerde prediction rows en 171 mogelijke features.
Er werden geen rijen verwijderd om sparse hogere timeframes passend te maken; missing
waarden bleven expliciet en werden uitsluitend via train-only mediaanimputatie behandeld.

## Besluit per horizon

De pipeline selecteert model+feature research candidates tegenover `mvp/reference`.
Als attributiecontrole is daarnaast per horizon exact dezelfde vaste `reference`-familie
over alle featurevarianten vergeleken. **De gekozen featurevariant is op alle acht
horizons identiek in beide analyses.** Daardoor wordt de featureconclusie niet gedragen
door alleen een wijziging van model-family.

| Horizon | Behouden featurevariant | Pipeline-family | Δ macro-F1 | Δ worst-F1 | Δ Brier | Fold ΔF1 (2022 / 2023 / 2024) |
|---:|---|---|---:|---:|---:|---|
| 3m | `price_session_regime_multitimeframe` | `logistic` | +0.002936 | +0.003717 | -0.008700 | +0.002453 / +0.001954 / +0.004400 |
| 6m | `price_session_regime_3min` | `logistic` | +0.000309 | +0.000253 | -0.000229 | +0.001855 / -0.001997 / +0.001068 |
| 9m | `price_session_regime_multitimeframe` | `logistic` | +0.000210 | +0.000577 | -0.004263 | -0.000808 / +0.000577 / +0.000860 |
| 12m | `price_session_regime_multitimeframe` | `reference` | +0.000514 | +0.000824 | -0.002756 | -0.001129 / +0.000824 / +0.001846 |
| 15m | `mvp` | `logistic` | +0.000000 | +0.000000 | +0.000000 | +0.000000 / +0.000000 / +0.000000 |
| 30m | `price_session_regime_multitimeframe` | `logistic` | +0.003228 | +0.002182 | -0.001202 | +0.004647 / +0.002182 / +0.002855 |
| 60m | `price_3min` | `reference` | +0.004632 | +0.006743 | -0.000128 | +0.003819 / +0.006743 / +0.003334 |
| 180m | `price_3min` | `logistic` | +0.001065 | +0.001360 | -0.000116 | +0.002931 / +0.001360 / -0.001098 |

De delta's zijn de feature-isolerende `reference`-vergelijking tegen
`mvp/reference`; lagere Brier is beter.

### Interpretatie

- **3m, 30m en 60m** geven het sterkste bewijs: de geselecteerde rijkere features
  verbeteren macro-F1 in alle drie outer jaren en voldoen tegelijk aan de vooraf
  vastgelegde worst-fold- en Brier-poort.
- **6m, 9m, 12m en 180m** passeren de formele phase7-v2-poort, maar de winst is klein
  en minstens één outer jaar verslechtert in macro-F1. De 6m-keuze verbetert Brier
  licht maar verslechtert gemiddelde log loss. Dit zijn onderzoeksverbeteringen,
  geen sterke performanceclaim.
- **15m** behoudt het MVP-featureschema. De enige pipeline-candidate daar is een
  minieme model-familywijziging (`mvp/logistic`), geen featureverbetering.

De featurevarianten voor de v0.2 researchbenchmark zijn dus:

- 3m: `price_session_regime_multitimeframe`
- 6m: `price_session_regime_3min`
- 9m: `price_session_regime_multitimeframe`
- 12m: `price_session_regime_multitimeframe`
- 15m: `mvp`
- 30m: `price_session_regime_multitimeframe`
- 60m: `price_3min`
- 180m: `price_3min`

## Economische uitkomst

Er zijn **0 economic promotion candidates** op alle acht horizons. Voor alle getrainde
`reference`, `logistic`, `ridge` en `xgboost`-aggregaten is het geselecteerde
beleid cash/no-trade: gemiddelde en worst-fold trade count zijn 0. Fase 7 levert dus
een betere price-only **researchbenchmark** op sommige horizons, maar geen tradingchampion.

## Coverage en beperkingen

Over de volledige 171-featurematrix zijn 17.510.302 van 91.433.358 waarden missing
(**19,15%**).

| Timeframe | Beschikbaar |
|---|---:|
| 1min | 100,0% |
| 3min | 100,0% |
| 5min | 96,2% |
| 15min | 83,4% |
| 30min | 72,1% |
| 1h | 59,9% |
| 3h | 15,9% |

Vooral `3h` en `1h` zijn sparse. Hun ontbrekende waarden zijn niet geforwardfilld
en gebruiken geen validation/teststatistieken; ze worden uitsluitend op de relevante
trainingfold mediaan-geïmputeerd. De sterke missingness blijft een operationele en
methodologische beperking van de volledige multi-timeframevariant.

Verder:

- HistData is bid-only; echte historische ask, spread en betrouwbare tick count ontbreken.
- Microstructure is daarom niet gesynthetiseerd.
- Probabilities zijn nog ongekalibreerd; volledige OOF-calibratie hoort bij fase 11.
- Phase7-v2 bevat geen afzonderlijke formele significantietest bovenop de vooraf
  vastgelegde mean/worst-fold/Brier-poorten; zeer kleine delta's worden daarom niet
  als sterk bewijs geïnterpreteerd.
- De finale 2025+ holdout is niet geopend.

## Exitbesluit

Fase 7 voldoet aan haar technische en empirische exitcriteria. Rijkere price-only
features voegen aantoonbaar predictive waarde toe op meerdere development folds, terwijl
15m bewust eenvoudig blijft. De v0.2 price-only benchmark kan worden bevroren, maar
**er wordt geen paper/live tradingmodel gepromoveerd**.

Volgende roadmapfase: **fase 8 — compact multi-timeframe neuraal kernmodel**.
