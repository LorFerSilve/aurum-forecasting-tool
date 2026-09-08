# Researchprotocol — fase 9, directe vijf-candle future path

**Protocol:** `phase9-v1`  
**Status:** structureel bevroren vóór de eerste formele phase-9 benchmark

## Doel en afhankelijkheid

Fase 9 onderzoekt of rijkere supervision uit het volledige pad van de volgende vijf
3min-candles de 15min-voorspelling aantoonbaar verbetert. Het pad is een challenger
bovenop de compacte neural core uit fase 8. Het vervangt de bevroren klassieke
phase-7 fallback niet automatisch.

Een formele phase-9 benchmark mag pas starten nadat de canonieke phase-8 run is
vastgesteld en als expliciete reference in de phase-9 benchmarkconfig is vastgezet.
Deze implementatiebranch exposeert daarom nog geen formeel `phase9 run`-commando.

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

Een recursive one-stepmodel is uitsluitend een latere benchmarkbaseline. Het krijgt
geen voorkeursstatus en mag de direct multi-step architectuur alleen vervangen als de
vooraf vastgelegde vergelijking dat empirisch rechtvaardigt.

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
- formele benchmark start alleen op clean Git en locked dependencies.
