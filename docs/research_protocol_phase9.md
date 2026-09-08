# Researchprotocol — fase 9, directe vijf-candle future path

**Protocol:** `phase9-v1`  
**Status:** structureel bevroren; canonieke Phase-8 reference vastgezet vóór de eerste formele Phase-9 benchmark

## Doel en afhankelijkheid

Fase 9 onderzoekt of rijkere supervision uit het volledige pad van de volgende vijf
3min-candles de 15min-voorspelling aantoonbaar verbetert. Het pad is een challenger
bovenop de compacte neural core uit fase 8. Het vervangt de bevroren klassieke
phase-7 fallback niet automatisch.

De klassieke Phase-7 fallback wordt eveneens cryptografisch gepind:

- run: `20260907T014255645674Z-b2afe281`;
- benchmarkcode: `3f0a703568224fe9169b1e9f8d61dad131f0005b`;
- completion manifest:
  `sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f`;
- 15min research champion: `mvp/logistic`.

De Phase-7 completion-pin is op 2026-09-08 administratief gecorrigeerd na herverificatie
van de oorspronkelijke run: de eerdere Git-evidence had één verkeerd overgenomen teken.
De run zelf, zijn 11.311 artifacts en de gesloten holdout zijn ongewijzigd.

De canonieke Phase-8 run is inmiddels formeel vastgesteld en vóór enig Phase-9
marktresultaat in de benchmarkconfig vastgezet:

- run: `20260908T020044407464Z-24f4c0d4`;
- benchmarkcode: `75945fe70606c8200cebc66678d0e220db5fb0ad`;
- completion manifest:
  `sha256:9bce4a80c7e0d65fc40ccd4e1fea3c1ef2d56022f8b60720f3d1a8b54cfbb922`;
- uitkomst: 0/8 neural predictive admissions en 0/8 economic promotions;
- actieve fallback: de frozen Phase-7 champions.

De canonical real-data preflight is op 2026-09-08 volledig geslaagd op commit
`6e466053f3e64aaad1f71b951b6d7a8cb008bda6` met 525.967 common eligible rows.
Het formele `phase9 run`-commando is daarom vrijgegeven, maar voert vóór het openen
van een formele run **opnieuw** dezelfde fail-closed preflight uit en eist daarna exacte
code-, data-, fold-, sample- en frozen-referencepariteit.

De finale holdout vanaf 2025-01-01 UTC blijft gesloten.

## Exacte targettijd

Voor prediction time `t`, die op de 3min UTC-grid ligt:

- de anchor is de volledig gesloten 3min-candle met close exact op `t`;
- stap 1 is de toekomstige candle `[t, t+3min]`;
- stap 2 is `[t+3min, t+6min]`;
- ...
- stap 5 is `[t+12min, t+15min]`.

Alle vijf candles moeten werkelijk bestaan, compleet zijn, dezelfde instrument/source
hebben en in één aaneengesloten 3min-segment liggen. Een gat maakt het volledige
path-label voor die sample ongeldig; er wordt niet geïnterpoleerd of over een gat heen
gesprongen.

## Representatie per candle

Iedere stap wordt lossless gerepresenteerd met vier schaalvrije log-bps-componenten:

1. `gap_log_bps = 10000 * log(open / previous_close)`;
2. `body_log_bps = 10000 * log(close / open)`;
3. `upper_wick_log_bps = 10000 * log(high / max(open, close))`;
4. `lower_wick_log_bps = 10000 * log(min(open, close) / low)`.

Voor stap 1 is `previous_close` de anchor-close op prediction time. Voor latere
stappen is het de close van de voorgaande path-candle.

Deze representatie maakt exacte reconstructie mogelijk:

- open en close blijven strikt positief via exponentiële reconstructie;
- high is altijd minstens max(open, close);
- low is altijd hoogstens min(open, close);
- upper/lower wick blijven niet-negatief.

De echte OHLC-waarden en timestamps van alle vijf candles worden daarnaast expliciet
in het labelartefact bewaard voor audit en evaluatie.

## Direct multi-step, niet autoregressief als standaard

De primary phase-9 challenger voorspelt alle vijf stappen in **één forward pass** uit
de fused historische representation. Voorspelde candle 1 wordt dus niet teruggevoerd
als input voor candle 2. Daardoor ontstaat geen teacher-forcing mismatch en geen
recursieve foutvermenigvuldiging in het primaire ontwerp.

Een recursive one-stepmodel is uitsluitend een benchmarkbaseline. Het krijgt geen
voorkeursstatus en mag de direct multi-step architectuur alleen vervangen als de
vooraf vastgelegde vergelijking dat empirisch rechtvaardigt. De baseline is
free-running: de q50-representatie van de vorige voorspelde candle wordt teruggevoerd
naar een GRUCell voor de volgende stap. Werkelijke future candles worden niet als
decoder-input gebruikt, ook niet tijdens training; er is dus geen teacher-forcing
mismatch.

## Probabilistische output

Voor ieder van de vier componenten en ieder van de vijf stappen worden drie geordende
quantielen geproduceerd:

- q10;
- q50;
- q90.

De architectuur parameteriseert de quantielen intrinsiek geordend. Voor upper/lower
wick is ook q10 niet-negatief.

Daarnaast voorspelt een afzonderlijke directe 15min-head dezelfde vier componenten
voor de geaggregeerde candle over `[t, t+15min]`.

De q50 future path kan altijd naar vijf geldige OHLC-candles worden gereconstrueerd.

## Bestaande Phase-8 outputs blijven behouden

De phase-9 modelvariant behoudt naast de path-heads de bestaande:

- direction logits;
- direct return head;
- range head;
- volatility head;
- availability-aware timeframe fusion.

De bestaande direction/return-targets worden niet stilzwijgend vervangen door het
path-target. Dat is noodzakelijk om de uiteindelijke predictive vergelijking met
phase 8 en de klassieke fallback betekenisvol te houden.

## Losscontract

De eerste phase-9 trainingvariant gebruikt:

- direction loss uit de neural core;
- pinball loss voor de vijf-stappen path quantielen;
- pinball loss voor de directe 15min-candle quantielen;
- Huber temporal-consistency loss tussen de q50 path-aggregatie en de q50 directe
  15min-head;
- de bestaande gradient clipping, AMP-guards, deterministic seeds en train-only
  preprocessing.

Frozen structurele lossgewichten in `configs/phase9.yaml`:

- direction: 1.00;
- path quantile: 1.00;
- direct 15min aggregate quantile: 0.25;
- temporal consistency: 0.10.

Loss-schalen moeten vóór de formele benchmark op train-only data worden genormaliseerd;
er mag geen outer teststatistiek in loss-normalisatie terechtkomen.

## Evaluatiecontract

De formele pipeline moet minstens rapporteren:

- MAE/Huber-fout per path-step en component;
- q10/q90 coverage en intervalbreedte per stap/component;
- cumulatieve path close-returnfout over 15 minuten;
- high/low/rangefout van de geaggregeerde path;
- direction accuracy/macro-F1 van de bestaande comparable direction head;
- path-implied direction accuracy/macro-F1 op de bevroren 6 bps neutral zone;
- consistency tussen reconstructed path en directe 15min-head;
- neural direction/probability metrics versus de canonieke phase-8 neural variant;
- dezelfde metrics versus de frozen phase-7 classical fallback;
- base/stress policy-uitkomsten met exact dezelfde execution assumptions.

De outer jaren, development-only discipline en reserved calibration quarter blijven
ongewijzigd.

## Promotie

Future-path learning wordt alleen behouden als research architecture wanneer het over
meerdere folds aantoonbaar extra waarde levert. Minimaal:

- minstens één vooraf gekozen primaire metric verbetert;
- worst-fold performance blijft aanvaardbaar;
- direction probability quality verslechtert niet betekenisvol;
- intervalcoverage is niet triviaal verkregen door extreem brede intervallen;
- reconstructed candles blijven structureel geldig;
- de extra inferencekosten blijven operationeel begrensd.

Als de path-head geen nuttige verbetering levert, blijft hij een onderzoeksoutput en
wordt de eenvoudigere direction/return-architectuur behouden.

Geen phase-9-uitkomst activeert paper of live trading.

## Canonieke benchmarkuitkomst

De canonieke run `20260908T211301827616Z-ad6b573c` is afgerond en gevalideerd.
Het vooraf vastgelegde promotiecontract resulteert in:

- `keep_phase7_champion`;
- Phase-9 direct: `retain_research_distribution_only`;
- Phase-9 recursive: `do_not_carry_as_default`;
- economic promotion: `false`;
- paper/live activation: `false`;
- finale holdout: gesloten.

Direct verbetert mean macro-F1 slechts marginaal versus de frozen Phase-8 neural
reference (+0,002397), terwijl de worst-fold macro-F1 verslechtert (-0,003099).
Tegenover de Phase-7 champion blijft mean macro-F1 ongeveer 0,103 lager en worst-fold
macro-F1 ongeveer 0,118 lager. Alle policies selecteren cash/no-trade.

De marginale q10-q90 pathcoverage van direct ligt wel dicht bij de nominale 80%
(78,92% per pathcomponent en 79,75% voor aggregate componenten). Daarom mag de direct
path-head als research/distributionele output worden bewaard, maar dit geldt niet als
bewijs van directionele of economische edge. Fysieke high/low/range intervalcoverage
wordt niet afgeleid uit niet-joint marginale quantielen en wordt pas na latere
distributionele calibratie/sampling verantwoord onderzocht.

Zie `docs/phase9_verification.md`.

## Verificatie vóór formele benchmark

- exact vijf toekomstige 3min-candles per geldige sample;
- gap in één future path verwijdert de sample;
- path-timestamps zijn exact t, t+3, t+6, t+9, t+12 en closes tot t+15;
- targetrepresentatie reconstrueert de opgeslagen OHLC exact;
- willekeurige modeloutputs reconstrueren positieve OHLC-invariante candles;
- quantielen zijn per component geordend;
- wickquantielen zijn niet-negatief;
- parameterbudget faalt gesloten;
- temporal-consistency loss is differentiable;
- finale holdout blijft ontoegankelijk;
- formele benchmark start alleen op clean Git en locked dependencies;
- canonical Phase-7/Phase-8 completion manifests en vereiste artifacts zijn intact;
- actuele curated 1min/3min/15min data matcht de frozen Phase-8 source manifests;
- frozen 15min predictions en policies pareren exact op iedere Phase-9 outer universe;
- direct en recursive passeren elk een korte real-data forward/backward/checkpoint-smoke;
- een formele run wordt pas geregistreerd nadat deze volledige preflight opnieuw is geslaagd.


## Common-sample referencepariteit

Phase 9 kan samples verliezen wanneer één van de vijf vereiste toekomstige
3min-candles ontbreekt. Daarom mogen Phase-7/Phase-8 metrics niet rechtstreeks uit
hun oorspronkelijke, grotere outer testset worden overgenomen. Frozen reference
predictions worden zonder refit en zonder policy-reselectie gesubset op exact de
Phase-9 sample-ID's, in Phase-9 volgorde. Prediction timestamps, direction targets,
return targets en sample digest moeten exact overeenkomen; iedere mismatch faalt
gesloten.

Een future-mutation guard controleert bovendien dat het wijzigen van candles ná
prediction time de historische sequence tensors niet verandert, terwijl het
future-path label wel verandert. Daarmee blijft future informatie uitsluitend target.
