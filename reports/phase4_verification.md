# Verificatie fase 4 — end-to-end research-MVP

Datum: 2026-09-04  
Uitkomst: **technisch geslaagd; economisch negatief**

## Volledige runs

Het commando hieronder is tweemaal zonder handmatige tussenstappen op de echte XAU/USD-data
uitgevoerd:

```powershell
.\.venv\Scripts\gold-forecast.exe mvp run --config configs/mvp.yaml
```

Geslaagde run-ID's:

- `20260904T154809220237Z-a2b9c042`;
- `20260904T155009507615Z-73592cb0`.

Beide runs gebruikten modeltabelversie
`sha256:d5ad6e63676b19d9f80c84c31f2f643261b314962e0db11368a937cf7a5e6453`
en modelversie `v0.1.0+09e5c67a0f9d`. Alle negentien inhoudelijke runartefacten buiten het
tijd- en run-ID-afhankelijke `run.json` — inclusief configuratiesnapshots, selectie,
modelvergelijking, uurrapport, tien reliability-buckets, backtestbeslissingen, trades,
samenvattingen en voorbeeldvoorspelling — waren per bestand byte-identiek.

## Modelselectie

Zes vooraf begrensde logistische-regressiekandidaten zijn alleen op validation vergeleken:
`C in {0.1, 1, 10}` maal class weight `{none, balanced}`. Op de primaire macro-F1 werd
`C=0.1`, `class_weight=balanced` geselecteerd (`0,424334` validation macro-F1). Het model is
uitsluitend op train gefit. De testset is één keer per run geëvalueerd; de tweede volledige
run diende uitsluitend om de reproduceerbaarheid te controleren en niet om een keuze te
wijzigen.

## Testresultaten op exact 111.098 records

| Model | Dekking | Accuracy | Balanced accuracy | Macro-F1 | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Meest frequente klasse | 1,000000 | 0,628625 | 0,333333 | 0,257323 | 13,385705 | 0,742750 |
| Altijd up | 1,000000 | 0,191210 | 0,333333 | 0,107011 | 29,151762 | 1,617581 |
| Altijd down | 1,000000 | 0,180165 | 0,333333 | 0,101774 | 29,549839 | 1,639669 |
| Laatste candledirection | 1,000000 | 0,187690 | 0,331650 | 0,183497 | 29,278615 | 1,624620 |
| Momentum | 1,000000 | 0,517633 | 0,381635 | 0,381701 | 17,386266 | 0,964734 |
| Mean reversion | 1,000000 | 0,521017 | 0,387735 | 0,387768 | 17,264280 | 0,957965 |
| Logistische regressie | 1,000000 | 0,571738 | 0,422887 | **0,424344** | **0,999740** | **0,590026** |

De logistische regressie is de eerste champion op de vooraf gekozen macro-F1. De hogere
accuracy van de meest-frequente baseline komt door de dominante `neutral`-klasse en gaat
niet samen met gebalanceerde klassedekking. Alle probabilities zijn expliciet
`preliminary` en `uncalibrated`.

## Kostenbewuste backtest

Policy: confidence minstens `0,50`, long/short/no-signal, maximaal één niet-overlappende
positie, entry altijd na prediction time.

| Scenario | Kosten | Signalen | Trades | Hit rate | Gem. netto/trade | Cumulatief netto | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 4,0 bp | 1.480 | 527 | 42,31% | -5,2621 bp | -2.773,10 bp | 2.851,21 bp |
| Stress | 5,5 bp | 1.480 | 527 | 38,52% | -6,7621 bp | -3.563,60 bp | 3.634,21 bp |

Er waren 953 overlappende signalen die conform de policy zijn onderdrukt. Het brutoresultaat
was al negatief (`-665,10 bp`); kosten maken dit verder slechter. Deze configuratie mag dus
niet naar paper trading of echte uitvoering worden gepromoveerd.

## Prediction- en softwarecontroles

- [x] De modelbundle bevat model, preprocessor, vaste klassevolgorde, featureschema, config,
  seed en data-/specversies; hashes worden bij laden gecontroleerd.
- [x] Reload geeft exact dezelfde probabilities als vóór opslag.
- [x] Het prediction-contract valideert UTC, kanssom, argmax, klassevolgorde en alle tien
  verplichte velden.
- [x] De laatste testcandle produceerde als demonstratie `neutral` met kansen
  `0,200229 / 0,614597 / 0,185173`.
- [x] Kapotte, incomplete, dubbele of niet-causale candles/predictions falen gesloten.
- [x] 134 tests, Ruff, mypy strict en `pip check` zijn groen.
- [x] Sdist en wheel bouwen; installatie van de wheel in een nieuwe Python 3.11-omgeving,
  CLI-load en configvalidatie zijn geslaagd.

## Besluit

`promote` voor de softwareketen naar researchrelease `v0.1.0`; `keep champion` voor de
logistische regressie als eerlijke benchmark; `stop` voor tradinginzet. Fase 5 mag de
prijsdata en marktmodellering versterken, maar moet tegen exact dezelfde benchmark worden
vergeleken.
