# Fase 6 — verificatierapport

**Datum:** 2026-09-07  
**Protocol:** `phase6-v1`  
**Status:** afgerond als researchbenchmark; geen challenger gepromoveerd

## Samenvatting

De implementatie van fase 6 is volledig uitgevoerd en de vooraf vastgelegde benchmark
is succesvol afgerond. De run met ID
`20260906T180106245286Z-630162b9` omvat acht horizons (3, 6, 9, 12, 15, 30, 60 en
180 minuten), drie outer walk-forwardjaren (2022–2024), elf model-/baselinevarianten
per combinatie en dus 264 outer evaluaties. Daarvan zijn 96 outer fits echte
getrainde modelkandidaten: vaste logistieke referentie, getunede logistiek, ridge en
XGBoost.

De benchmark selecteerde in alle 96 getrainde gevallen expliciet de cashpolicy. Er
werden daardoor geen outer trades uitgevoerd. De vaste logistieke referentie blijft
per horizon de researchchampion; geen challenger voldoet aan de promotiecriteria en
er wordt geen tradingchampion geactiveerd. Dit is een gecontroleerd negatief
researchresultaat, geen bewijs van winst of verlies op toekomstige marktdata.

## Wat is geïmplementeerd

- Versieerbare labels voor acht horizons met richting, continue arithmetic return,
  toekomstige range en gerealiseerde volatiliteit.
- Chronologische outer walk-forward-splits met twee inner validatieblokken, purge,
  een 181-minutengap en een afzonderlijk gereserveerd calibratieblok.
- Een gesloten finale holdout vanaf 2025-01-01 UTC.
- Gelijke records, preprocessingregels en kosten voor alle kandidaten binnen een
  horizon/fold.
- Baselines, logistische regressie, ridge-regressie en XGBoost met early stopping.
- Execution-aware backtester v1 met prediction-, signal-, order- en filltijd,
  spread-/slippagekosten, no-signal policies, non-overlap en fold-/sessierapportage.
- Een reproduceerbare CLI-flow:
  `benchmark run`, `benchmark validate` en `benchmark reproduce`.

## Resultaat en promotiebesluit

De beleidsselectie gebruikt uitsluitend de inner resultaten. Van de 288 policy-
pogingen waren er 287 onvoldoende qua tradecoverage; 48 hadden bovendien een
negatieve gemiddelde inner netto-uitkomst (de categorieën overlappen). Slechts één
policy had voldoende trades en die was negatief. Er is daarom geen geoorloofde reden
om achteraf thresholds te verlagen op basis van de outer jaren.

Voor de oorspronkelijke 15-minutenhorizon verbetert XGBoost de Brier- en log-loss
ten opzichte van de referentie, maar niet de macro-F1. De return-MAE blijft praktisch
gelijk aan een nul-returnbaseline. De langere horizons tonen geen consistente grote
verbetering en hebben meer labeluitval. De volledige interpretatie staat in de
[model card](model_card_phase6.md).

Besluit voor elke horizon: **`keep_champion`**. Fase 7 mag nieuwe price-only features
onderzoeken, maar moet deze opnieuw onder exact hetzelfde benchmarkcontract tegen de
behouden referentie laten concurreren.

## Reproduceerbaarheid en integriteitscontroles

De eindcontrole is uitgevoerd met:

- 96/96 opnieuw gefitte finale kandidaten met exacte voorspelling-, verwachte-return-
  en klassepariteit;
- 96/96 opnieuw afgeleide policies met exacte selectiepariteit;
- 2.319 geverifieerde benchmarkartefacten;
- matching van rankings en XGBoost-rounds uit de opgeslagen inner validatiescores;
- finale holdout niet geopend;
- oorspronkelijke benchmarkrun niet gewijzigd;
- 27 gecontroleerde MVP-feature-, label- en modelartefacten byte-identiek gebleven;
- de benchmarklabels voor 15 minuten inhoudelijk identiek aan de bestaande MVP-labels
  (526.005 geldige records).

Het volledige reproductierapport staat in
[`reports/phase6_reproduction_20260906T180106245286Z-630162b9.json`](../reports/phase6_reproduction_20260906T180106245286Z-630162b9.json).
De aanvullende selectiecontrole staat in
[`reports/phase6_selection_verification.json`](../reports/phase6_selection_verification.json)
en de MVP-controle in
[`reports/phase6_mvp_preservation.json`](../reports/phase6_mvp_preservation.json).

De reproduceerbaarheidsscope heeft een bewuste grens: alle finale kandidaten zijn
opnieuw gefit, maar niet alle inner kandidaten zijn opnieuw getraind; features en
labels zijn niet opnieuw gegenereerd; de opgeslagen outer backtestresultaten zijn
niet opnieuw berekend. De controle bevestigt dus de ontwikkelartefacten en hun
selectiepariteit, niet een nieuwe onafhankelijke holdoutmeting.

## Verificatiecommando's

De test- en kwaliteitscontroles die bij deze afronding horen zijn:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m build
```

De bestaande benchmarkartefacten, bronhashes, configuratie en dependencyversies zijn
onderdeel van de runinventaris. Zie [roadmap.md](../roadmap.md) voor het bijgewerkte
fasebesluit; de eerstvolgende ontwikkelstap is fase 7.
