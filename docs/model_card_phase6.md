# Model card — fase 6, protocol phase6-v1

Status: researchbenchmark; geen challenger gepromoveerd en geen tradingmodel geactiveerd.  
Beoordeelde run: `20260906T180106245286Z-630162b9`, geslaagd op 2026-09-06.  
Datum van deze interpretatie: 2026-09-07.

## Uitkomst en bedoeld gebruik

De volledige vooraf vastgelegde vergelijking omvat acht horizons, drie outer testjaren
en elf modellen/baselines per combinatie: 264 outer evaluaties. De vier getrainde
families — vaste logistische referentie, getunede logistiek, ridge en XGBoost — leveren
samen 96 outer evaluaties. Alle 96 selecteerden expliciet een cashpolicy en voerden
nul trades uit. Netto, kosten, exposure en gerealiseerde drawdown zijn daardoor nul.
Dit toont geen winstgevende strategie aan; het is de uitkomst van de toelatingsregels
voor handelen onder de onderzochte data, modellen en kosten.

Voor elk van de acht horizons blijft de vaste logistische referentie de
researchchampion. Geen challenger voldoet aan alle vooraf vastgelegde promotiecriteria.
Er bestaat nog geen tradingchampion. Deze benchmark is bedoeld voor gecontroleerde
vergelijkingen in volgende researchfasen.

De bron voor alle cijfers is de bewaarde run onder
`reports/benchmark_runs/20260906T180106245286Z-630162b9/`: `summary.json`, de
`horizon_H/test_YEAR/fold_summary.json`-bestanden en de onderliggende
`MODEL/selection.json`-bestanden. Deze model card herinterpreteert de bestaande
resultaten en past geen modellen, thresholds of runartefacten aan.

## Data, features en tijdcontract

Publieke HistData bid-only M1-data uit 2020–2024 vormt de bron: 1.726.589 minuutcandles
en 575.254 volledig opgebouwde 3-minuutcandles. De achttien causale MVP-features blijven
gelijk: recente prijsveranderingen, candlekenmerken, momentum, historische volatiliteit,
afstand tot gemiddelden en cyclische UTC-tijdvelden. Featurebeschikbaarheid volgt de
sluiting van de 3-minuutcandle; rolling windows gaan niet over datagaten.

Entry ligt één minuut na het voorspeltijdstip; exit ligt H minuten na entry. Klassen
gebruiken log-returngrenzen van -6 en +6 bps, inclusief een neutrale zone. Ook
arithmetic return, toekomstige range en gerealiseerde volatiliteit worden als targets
opgeslagen. De labelimplementatie draagt `label_spec_version=v0.2`.

Alleen ononderbroken minuutpaden tellen mee. Van de 534.698 beschikbare
featuretijdstippen blijven per horizon de volgende labels over, vóór de splits:

| Horizon (minuten) | Geldige labels | Ontbrekende/onvolledige paden | Uitval |
|---|---:|---:|---:|
| 3 | 531.882 | 2.816 | 0,53% |
| 6 | 530.411 | 4.287 | 0,80% |
| 9 | 528.940 | 5.758 | 1,08% |
| 12 | 527.471 | 7.227 | 1,35% |
| 15 | 526.005 | 8.693 | 1,63% |
| 30 | 518.711 | 15.987 | 2,99% |
| 60 | 504.370 | 30.328 | 5,67% |
| 180 | 450.193 | 84.505 | 15,80% |

Die uitval kan de geëvalueerde marktperioden selecteren; prestaties op uitgesloten
paden zijn onbekend. Een 24-uurhorizon is uitgesteld zolang een betrouwbare
sluitingskalender ontbreekt. Binnen één horizon/fold gebruiken alle kandidaten exact
dezelfde records; tussen horizons verschilt de dekking.

Outer evaluaties betreffen 2022, 2023 en 2024, met groeiende trainingshistorie vanaf
2020. Voor outer jaar Y zijn april–juni en juli–september van Y-1 de twee inner
validatieblokken. Oktober–december van Y-1 blijft gereserveerd voor latere calibratie.
Deze gereserveerde records worden niet gebruikt voor fit, early stopping of selectie.
Een gemeenschappelijke 181-minutengap en het vereiste dat labels vóór het blokeinde
aflopen, scheiden de blokken. Preprocessing en afgeleide rendementsverwachtingen
gebruiken uitsluitend trainingsrecords.

De finale holdout vanaf 2025-01-01 UTC blijft gesloten. Het reeds besproken MVP-testjaar
2024 is nu expliciet een ontwikkelperiode; het is geen nieuwe onafhankelijke bevestiging.
De drie outer jaaruitkomsten zijn geen drie onafhankelijke statistische experimenten:
trainingshistorie en veel marktstructuren worden gedeeld.

## Modellen en beleidsselectie

De vaste referentie gebruikt logistische regressie met `C=0.1` en
`class_weight=balanced`, opnieuw gefit per outer fold. De drie challengers krijgen
ieder drie configuraties en twee inner folds: logistiek `C=0.1,1,10`, ridge
`alpha=0.1,10,1000` en XGBoost `max_depth=2,3,4`. XGBoost gebruikt maximaal 120 rondes
en early stopping met geduld 10 op uitsluitend inner validatie. Het aantal
configuraties is vergelijkbaar, niet de hoeveelheid rekenwerk.

Classifiers worden geselecteerd op gemiddelde inner macro-F1 met log loss als
tiebreaker; ridge op arithmetic-return-MAE. Voor een classifier wordt de verwachte
return berekend uit zijn kansen en de op training geschatte gemiddelde return per
klasse. Ridge voorspelt rechtstreeks return en gebruikt een normale benadering van
trainingsresiduen om klassekansen af te leiden. Alle kansen zijn ongekalibreerd.
Met name `class_weight=balanced` herweegt de trainingsklassen; de resulterende kansen
zijn niet zonder aanvullende verificatie marktfrequenties. Een lage geaggregeerde
calibration error bewijst evenmin dat zeldzame directionele signalen betrouwbaar zijn.

De zes MVP-baselines en empirische trainingsklassefrequenties blijven afzonderlijk
zichtbaar. Naïeve baselines gebruiken de vaste policy `(confidence=0.5, edge=0)`;
de vier getrainde families kiezen hun policy op inner resultaten. Daarom beschrijven
nettoresultaten het volledige model-plus-policy-systeem. Een verschil in nettoresultaat
tussen een cashmodel en een handelende baseline is niet uitsluitend een modelverschil.
De one-hot-baselines drukken bovendien kunstmatige zekerheid uit; empirische priors
zijn een bruikbaardere eenvoudige referentie voor probability metrics.

Per getrainde familie worden drie policies geprobeerd:
`(confidence, minimum verwachte netto-bps) = (0.5,0), (0.6,2), (0.7,4)`.
Een policy moet minstens twintig trades in ieder inner blok en een positief gemiddelde
van de twee cumulatieve inner nettoreturns behalen. Anders wordt cash vastgelegd als
`confidence_threshold=1` en `min_expected_net_bps=1e12`.

Van de 288 policy-pogingen zijn de volgende categorieën onderling uitsluitend:

| Inner resultaat | Pogingen |
|---|---:|
| Onvoldoende trades, negatieve gemiddelde netto-uitkomst | 47 |
| Onvoldoende trades, precies nul gemiddelde netto-uitkomst | 230 |
| Onvoldoende trades, positieve gemiddelde netto-uitkomst | 10 |
| Voldoende trades, negatieve gemiddelde netto-uitkomst | 1 |
| Voldoende trades, positieve gemiddelde netto-uitkomst | 0 |

Dus 287 pogingen hebben te weinig trades en 48 hebben een negatieve gemiddelde
netto-uitkomst; deze twee tellingen overlappen voor 47 pogingen en mogen niet worden
opgeteld. De ene poging met voldoende trades is XGBoost, horizon 180 minuten, outer
jaar 2022, policy `(0.5,0)`: 46 en 124 inner trades, met -126,71 en -810,63 netto-bps.
Het gemiddelde van de twee cumulatieve foldresultaten is -468,67 bps.

Er zijn **96 expliciet geselecteerde cashpolicies en nul actieve geselecteerde policies
die toevallig geen outer fills krijgen**. Dit onderscheid is gecontroleerd via
`selected_policy` in alle 96 selectieartefacten. Achteraf de drempels verlagen totdat
de outer jaren winst tonen, zou een nieuw experiment op reeds bekeken data zijn.

## De oorspronkelijke horizon van 15 minuten

Onderstaande waarden zijn ongewogen gemiddelden over outer jaren 2022, 2023 en 2024.
De slechtste macro-F1 is het minimum; bij foutmaten is lager beter.

| Model | Gem. macro-F1 | Slechtste macro-F1 | Gem. Brier | Gem. log loss | Gem. return-MAE (bps) |
|---|---:|---:|---:|---:|---:|
| Vaste logistische referentie | 0,425052 | 0,423582 | 0,579064 | 0,984324 | 6,008370 |
| Getunede logistiek | 0,425052 | 0,423582 | 0,579063 | 0,984323 | 6,008370 |
| Ridge | 0,261727 | 0,251841 | 0,596783 | 0,998863 | 6,006565 |
| XGBoost | 0,325881 | 0,319677 | 0,476877 | 0,829383 | 6,007077 |
| Empirische priors | 0,261666 | 0,251714 | 0,522110 | 0,898232 | 6,007078 |
| Altijd nul return voorspellen | — | — | — | — | 6,007048 |

XGBoost geeft betere Brier en log loss dan de referentie en empirische priors, maar
lagere macro-F1 dan de logistische referentie. De gemiddelde accuracy is 65,11% voor
XGBoost, 64,72% voor empirische priors en 59,52% voor de referentie. Deze hogere
accuracy wordt dus vergeleken met een sterke neutrale meerderheidsklasse en mag niet
als overeenkomstig winstpercentage worden gelezen.

De referentie heeft per jaar macro-F1 `0,425102 / 0,423582 / 0,426473`; XGBoost
`0,334546 / 0,319677 / 0,323421`. De slechtste Brier is respectievelijk 0,594970 en
0,515330. Return-MAE is vrijwel gelijk aan altijd nul voorspellen; de kleine verschillen
leveren zonder verdere statistische toets geen bewijs van een bruikbaar handelsvoordeel.
De historische volatiliteitsbaseline heeft gemiddeld 2,481726 bps MAE op het
volatiliteitstarget; de training-mediaanrange heeft 5,610840 bps MAE op het rangetarget.
Die twee fouten hebben andere targets dan return-MAE en rangschikken geen richtingsmodel.

## Langere horizons en promotiebesluit

| Horizon | Model | Gem. macro-F1 | Slechtste macro-F1 | Gem. Brier | Gem. return-MAE (bps) |
|---|---|---:|---:|---:|---:|
| 60 minuten | Referentie | 0,370260 | 0,360643 | 0,647244 | 12,296098 |
| 60 minuten | XGBoost | 0,370295 | 0,363784 | 0,641308 | 12,304354 |
| 180 minuten | Referentie | 0,330967 | 0,316387 | 0,667029 | 22,415632 |
| 180 minuten | XGBoost | 0,282056 | 0,268312 | 0,641451 | 22,463585 |

Op 60 minuten is het gemiddelde macro-F1-verschil slechts circa 0,000035 ten gunste
van XGBoost. De return-MAE van altijd nul voorspellen is 12,292987 bps op 60 minuten
en 22,417276 bps op 180 minuten. De langere horizons tonen daarmee geen consistente
grote verbetering in de directe returnverwachting; tegelijk loopt de labeluitval op.

Het protocol vereist onder andere betere gemiddelde macro-F1, niet slechtere Brier
en slechtste macro-F1, minstens twintig outer trades en positieve base-netto in ieder
outer jaar, positieve gemiddelde stress-netto en geen slechtere slechtste nettofold.
De economische voorwaarden falen voor iedere challenger omdat alle geselecteerde
policies cash blijven. De bewaarde beslissing is voor iedere horizon `keep_champion`;
een beter deelresultaat geeft geen automatische promotie.

## Betekenis van de backtest en verschil met v0.1

De huidige engine rekent arithmetic P&L uit op een vaste entry-bidnotional per trade.
Een long koopt aan een askproxy en verkoopt aan bid; een short verkoopt aan bid en
koopt terug aan een askproxy. De basisspreadproxy is 3 bps, stress is 4,5 bps, met
0,5 bps slippage per kant en zonder commissie/financiering. Werkelijke totale
round-tripkosten kunnen door prijsverhoudingen licht afwijken van exact 4 of 5,5 bps.
Stress gebruikt dezelfde signalen. Signaal/order liggen op prediction time, fills
op de vertraagde entry, en posities overlappen niet binnen een strategie/instrument.

Opgetelde bps zijn geen account-rendementspercentage of samengestelde equitycurve.
De drawdown telt gerealiseerde exit-P&L en mist intratrade mark-to-marketverliezen.
UTC-sessiebuckets zijn vaste uurblokken, geen zomer-/wintertijdgecorrigeerde beurssessies.
Bid-only minuutdata en spreadproxies leveren geen bewijs van werkelijke uitvoerbaarheid.

De eerdere v0.1-simulatie had één vaste 2024-test, confidence `>=0.5`, 527 trades en
-2.773,10 netto-bps volgens de toenmalige signed-logreturnconventie. Fase 6 verandert
de trainingsblokken, beleidsselectie en P&L-conventie. De nuluitkomst nu betekent dat
het nieuwe selectiemechanisme cash kiest; zij bewijst niet dat dezelfde eerdere
strategie verliesloos of winstgevend is geworden.

## Identiteit en verificatiegrens

De volledige run heeft status `succeeded` en liep van `2026-09-06T18:01:06.250Z`
tot `2026-09-06T18:39:24.643Z`. Seed: `20260906`; protocol: `phase6-v1`.
Het runmanifest noemt code `07b9e1a939c92daf950ae073016f393b3caba32d+dirty`.
Daarom zijn de meegearchiveerde bronbestanden en hun afzonderlijke hashes nodig om
de daadwerkelijk gebruikte code te identificeren; alleen het commit-ID is onvoldoende.

| Onderdeel | Versie / SHA-256 |
|---|---|
| Gecombineerde benchmarkdata | `a8da6d0b9179304db0d135a39f0609b1e9dbd9232f98810f78b367fbcc73bcd0` |
| M1-dataset | `ae899269c2f19fe38267bcd069ed951ba08aa1979354eab694cbee9be0e408cd` |
| 3-minuutdataset | `85705596397d199228bf65c392b7a8a2696d7dcf30365e3d5590710dc057ada9` |
| Featuredefinitie | `122383f4ea2e0a1467e2c077a599b70de6aa2a7c279dbb44fae30b78bfe856b8` |
| Rootconfigbestand | `3a04bb1e92872e85d1170d93f24ae85aaf1a4ede07268b5c4c860c630369df2e` |
| Protocolbestand in run | `63a62caeda3b0184e8e62cf4da65769edc9a45842c3599c3b3537ca5245c2b77` |
| Completion-inventarisversie | `ab464b7d61258ad436226fa0248981d86b78bb68ec8a01a4faa62bc7a2999ef7` |
| Completion-bestand | `dc0ec077b81e47603192f6c6518902f751883e9d85e77599aebd4be47959a0a8` |

De bronhashes in die inventaris omvatten onder meer
`source_snapshot/gold_forecasting/benchmark/pipeline.py`
(`b9ca15c2ac84d458d8f62ce62603dd48fa7affdc36db99c07cc6e87c7d9aefa9`),
`benchmark/models.py`
(`cf4d684ebac4f7b5ef3206ac1a79fb6f98cc79d51dad4cae29f631f2f5b8a975`),
`backtesting/v1.py`
(`dd8330e30da431a33e68830879be027777ddb07e40fff856dc36613915d73ddc`)
en `labels/multihorizon.py`
(`2b08974b8fb43694a545c291ddeae71790f2bc0070f52dfe87acdf3a5e675408`).
Alle bron-, model-, voorspelling- en resultaatbestanden staan in `completion.json`.
De datasets behouden daarnaast hun eigen bronmanifesten en per-recordlineage.

Geregistreerde versies: NumPy 2.4.6, pandas 2.3.3, PyArrow 23.0.1,
scikit-learn 1.9.0, SciPy 1.17.1, XGBoost 3.2.0 en joblib 1.6.0.
De volledige `requirements.lock` is meegearchiveerd met SHA-256
`c6bb3e533401cdebf5f3392c7731bfbf40fd429d877dc804e68ad173b433deab`.

De finale kandidaatfits en beleidsselecties zijn inmiddels opnieuw uitgevoerd en exact
geverifieerd (96/96 fits, 96/96 policies). Dit blijft een reproduceerbaarheidscontrole
van de ontwikkelartefacten: alle inner kandidaten zijn niet opnieuw getraind, features
en labels zijn niet opnieuw gegenereerd en de outer backtests zijn niet opnieuw berekend.
De finale holdout is niet geopend. Zie het [fase-6-verificatierapport](phase6_verification.md)
voor de volledige scope en controles.

## Beperkingen en volgende onderzoeksstap

De benchmark beantwoordt welke vooraf gespecificeerde kandidaten en policies onder
deze ontwikkelcondities standhouden. Hij bewijst niet dat één geïsoleerde bottleneck
de oorzaak van alle eerdere verliezen is. De nog zwakke returnvoorspelling,
ongekalibreerde kansen, grove klassegemiddelden voor expected return en bid-only
kostenproxy blijven afzonderlijke onderzoekshypothesen.

Feature-ablaties behoren tot fase 7 en volledige OOF-calibratie tot fase 11. Nieuwe
experimenten moeten hun vergelijking vooraf vastleggen, de bestaande referentie
behouden en prestaties over alle folds inclusief cashuitkomsten rapporteren.
De bekeken outer jaren zijn ontwikkeldata; de finale holdout blijft buiten deze
iteraties. Er is op basis van deze run geen grond voor een winstprognose op korte
of lange termijn.
