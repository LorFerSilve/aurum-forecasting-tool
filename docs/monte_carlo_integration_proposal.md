# Voorstel — Monte Carlo-integratie voor Aurum Forecasting Tool

**Status:** ontwerpvoorstel; nog niet activeren als productie- of tradinglogica  
**Repository-review:** main op commit c8549f147a500dd734ead023e18de56d18040332  
**Laatste formele benchmark:** Phase 8 run 20260908T020044407464Z-24f4c0d4  
**Voorgestelde hoofdimplementatie:** Phase 11–13, met voorbereidende contracts in Phase 9  
**Doel:** de forecasting tool niet kunstmatig "nauwkeuriger" laten lijken, maar voorspellingen probabilistisch bruikbaarder maken voor selectiviteit, downside-control, no-tradebeslissingen en robuustheidsanalyse.

---

## 1. Samenvatting van het advies

Monte Carlo kan Aurum betekenisvol verbeteren, maar **niet als vervanging voor het forecastingmodel en niet als truc om de raw next-candle accuracy te verhogen**.

De beste rol is tweeledig:

1. **Predictive Monte Carlo per voorspelling**
   - sample mogelijke toekomstige returns/candle-paths uit een gekalibreerde voorspellingsverdeling;
   - bereken P(net_return > 0), expected net return, downside-quantielen, expected shortfall/CVaR, intervalbreedte en later path-afhankelijke grootheden;
   - voer die outputs als extra evidence naar de Phase-12 decision policy;
   - gebruik ze vooral om twijfelachtige signalen naar **no-trade** te sturen.

2. **Strategy Monte Carlo voor robuustheid**
   - simuleer alternatieve chronologische ontwikkelingspaden via block/stationary bootstrap;
   - behoud temporele afhankelijkheid, volatility clustering, no-overlap en de bestaande executionlogica;
   - schat de verdeling van nettoresultaat, drawdown, coverage, exposure en slechtste scenario's;
   - gebruik dit als Phase-13 stresstest vóór de finale holdout.

De optimale implementatievolgorde is daarom:

~~~text
Phase 9
distributionele future-path outputs
        ↓
Phase 11
echte OOF-store + calibratie
        ↓
Phase 12
per-forecast Monte Carlo risk/decision engine
        ↓
Phase 13
strategy/block-bootstrap Monte Carlo + stresstests
        ↓
Phase 14
éénmalige finale holdout met volledig bevroren MC-policy
        ↓
Phase 15+
runtime-optimalisatie en paper/live inference
~~~

**Belangrijk:** de Monte Carlo-engine mag vóór Phase 11 wel technisch worden voorbereid, maar mag niet als economische selectielaag worden gepromoveerd zolang de gebruikte probabiliteiten/verdelingen niet OOF zijn gekalibreerd.

---

## 2. Relevante huidige staat van de repository

### 2.1 Phase 8 is technisch geslaagd, maar niet economisch gepromoveerd

De huidige main bevat de afgeronde Phase-8 neural benchmark. Het compacte multi-timeframe GRU-model:

- gebruikt direction-, return-, range- en realized-volatility-heads;
- verbetert Brier en log loss ten opzichte van de Phase-7 champions;
- passeert op **0/8 horizons** de vooraf vastgelegde predictive admission;
- produceert op **0/8 horizons** een economic promotion candidate;
- valt in alle outer folds terug op cash/no-trade.

Dat maakt het onwenselijk om nu reeds een Monte Carlo-laag boven ruwe Phase-8-scores te plaatsen en daar economische conclusies uit te trekken.

### 2.2 De probabilities zijn expliciet nog niet gekalibreerd

src/gold_forecasting/inference/prediction.py markeert het huidige publieke prediction contract met:

~~~text
calibration_status = "preliminary"
~~~

De benchmarkcode reserveert calibratiedata maar fit nog geen echte calibrator. De huidige reliability-tabellen zijn diagnostisch; zij zijn geen vervanging voor OOF probability calibration.

De roadmap plaatst de echte OOF-store en calibratie correct in **Phase 11**.

### 2.3 De huidige backtester heeft al een goede integratiehaak

src/gold_forecasting/backtesting/v1.py gebruikt vandaag reeds:

- confidence_threshold;
- min_expected_net_bps;
- direction probabilities;
- expected return;
- ex-ante kosten;
- long/short/no-signal;
- single-non-overlapping execution;
- base- en stresskosten.

De Monte Carlo-laag hoeft deze backtester daarom niet te vervangen.

De logischste evolutie is:

~~~text
huidig:
probability + point expected return + costs
                  ↓
             DecisionPolicy

later:
calibrated probability
+ predictive distribution
+ Monte Carlo risk summary
+ costs
+ OOD/meta evidence
                  ↓
             DecisionPolicy
~~~

### 2.4 Phase 9 is de natuurlijke technische voorloper

De roadmap voorziet in Phase 9 al:

- vijf toekomstige 3min-candles in één forward pass;
- geldige OHLC-reconstructie;
- quantielen of een passende kansverdeling;
- range/volatility;
- cumulatieve return;
- directe 15min-head;
- intervalcoverage.

Dit is precies het soort output waaruit een Monte Carlo-engine later scenario's kan samplen.

### 2.5 De huidige databeperking moet behouden blijven in de simulatie

De historische HistData-bron is bid-only. Er is geen betrouwbare historische ask/spread/tick-countobservatie.

Daarom mag Monte Carlo in de eerste versie **geen fictieve geavanceerde stochastic-spreadmodeling introduceren**.

Gebruik aanvankelijk dezelfde bevroren executionaannames als de backtester:

- base total round-trip cost: 4.0 bps;
- stress total round-trip cost: 5.5 bps;
- constant-spread proxy waar ask ontbreekt.

Monte Carlo moet onzekerheid in de forecast modelleren, niet databronbeperkingen verbergen.

---

## 3. Wat Monte Carlo in Aurum wél en niet moet doen

### 3.1 Wel

Per kandidaat-signaal willen we later vragen kunnen beantwoorden zoals:

- Wat is P(net_return > 0) na kosten?
- Wat is de verwachte netto return?
- Hoe breed is de predictive distribution?
- Wat is het 5%-downsidequantiel?
- Wat is de conditional expected loss in de slechtste 5%?
- Hoe groot is de kans dat de markt eerst een adverse excursion bereikt?
- Hoe vaak is het voorspelde pad intern coherent?
- Is de forecast nog bruikbaar onder de bestaande stresskosten?
- Is een signaal sterk genoeg om te handelen, of is no-trade rationeler?

### 3.2 Niet

Monte Carlo mag niet:

- worden voorgesteld als een methode die zelf predictive edge creëert;
- één slecht model via duizenden samples "betrouwbaar" laten lijken;
- normale returns veronderstellen zonder empirische validatie;
- thresholds fitten op de finale holdout;
- individuele historische trades iid door elkaar schudden en zo tijdsafhankelijkheid negeren;
- stochastic spread/slippage verzinnen zonder betrouwbare data;
- de bestaande no-overlap/executionregels omzeilen;
- een percentage als "kans op winst" tonen zolang het onderliggende contract niet correct gekalibreerd en gevalideerd is.

---

## 4. Voorgestelde architectuur

### 4.1 Predictive Monte Carlo

~~~text
features / sequences
        ↓
champion model(s)
        ↓
raw predictive outputs
        ↓
OOF calibrator / distribution calibrator
        ↓
CalibratedPredictiveDistribution
        ↓
MonteCarloSampler
        ↓
N scenario's
        ↓
MonteCarloForecastSummary
        ↓
Phase-12 decision policy
        ↓
LONG / SHORT / NO TRADE
~~~

### 4.2 Strategy Monte Carlo

~~~text
chronologische OOF decisions/predictions
        ↓
block / stationary bootstrap
        ↓
resampled chronological scenario
        ↓
bestaande backtester v1 opnieuw uitvoeren
        ↓
net P&L / drawdown / coverage / exposure / PF
        ↓
duizenden scenarioresultaten
        ↓
robustness distribution + promotion gate
~~~

De tweede flow moet de **backtester opnieuw uitvoeren** op resampled chronologische blokken. Alleen de finale lijst trade-returns shufflen is niet voldoende, omdat Aurum:

- serial dependence heeft;
- overlappende signalen onderdrukt;
- verschillende horizons kan combineren;
- exposure en kosten aan tijd koppelt;
- regimeclusters kan bevatten.

---

## 5. Implementatie per roadmapfase

### Phase 9 — voorbereiden, nog niet als tradefilter gebruiken

Bouw een algemeen distributioneel contract dat niet aan GRU of één horizon is gekoppeld.

Voorgestelde nieuwe componenten:

~~~text
src/gold_forecasting/uncertainty/
    __init__.py
    contracts.py
    distributions.py
~~~

Conceptuele contracts:

~~~text
CalibratableForecastDistribution
PathDistribution
ReturnDistribution
DistributionMetadata
~~~

Minimaal bewaren:

- prediction time;
- horizon;
- model/version;
- distribution family of quantile representation;
- location/scale of quantile grid;
- path dimensions;
- calibration status;
- sample universe/version.

Als de future-path head quantielen gebruikt, moeten de quantielen:

- monotonisch zijn;
- OOF geëvalueerd worden;
- geldige OHLC-candles reconstrueren;
- gezamenlijke temporele afhankelijkheid niet verliezen.

**Niet** vijf toekomstige candles onafhankelijk van elkaar samplen.

Laat Phase 9 minstens één expliciete distributionele representatie produceren, maar stel de definitieve samplerkeuze uit tot de OOF-resultaten beschikbaar zijn.

Een simpele Gaussian rond de point forecast mag hoogstens als sanity-check dienen, niet als standaardmodel voor XAU/USD.

### Phase 11 — calibratie als harde prerequisite

Monte Carlo wordt pas statistisch zinvol wanneer de inputverdeling betrouwbaar genoeg is.

Breid de Phase-11 OOF-store uit met:

- raw direction probabilities;
- calibrated direction probabilities;
- raw expected return;
- realized return;
- predictive quantiles/distributionparameters;
- realized path;
- predicted range/volatility;
- regime;
- eventfase;
- base/stress kostencontext;
- model/feature/data/fold versions.

Vergelijk op OOF-data minimaal twee eenvoudige return-distributiekandidaten:

1. **Empirical OOF residual distribution**
   - residuals uitsluitend uit voorgaande/OOF-data;
   - per horizon;
   - eventueel grof conditioneren op volatility/regime;
   - shrink naar een grotere pool wanneer een bucket te klein is.

2. **Robuuste parametric residual distribution**
   - bijvoorbeeld Student-t wanneer OOF fit en proper scores dit ondersteunen.

Promoveer niet automatisch de meest complexe variant.

Voor continue distributions toevoegen:

- PIT-histogram / probability integral transform;
- empirical interval coverage;
- interval width/sharpness;
- CRPS of een andere passende proper scoring rule;
- tail coverage;
- calibration per horizon;
- calibration per breed regime;
- stabiliteit over outer folds.

De bestaande Brier/log-loss/ECE blijven gelden voor direction probabilities.

### Phase 12 — echte per-forecast Monte Carlo decision layer

Hier hoort de hoofdimplementatie thuis.

Voorgestelde componenten:

~~~text
src/gold_forecasting/uncertainty/
    monte_carlo.py
    summaries.py

src/gold_forecasting/decision/
    __init__.py
    policy.py
~~~

De bestaande backtesting.v1.DecisionPolicy kan eerst compatibel blijven. Pas nadat de nieuwe policy bewezen beter is, mag de hoofdpolicy uitgebreid of verplaatst worden.

Per forecast minimaal in MonteCarloForecastSummary:

~~~text
draw_count
seed
p_gross_positive
p_net_positive_base
p_net_positive_stress
expected_gross_return_bps
expected_net_return_bps_base
expected_net_return_bps_stress
q05_net_return_bps
q50_net_return_bps
q95_net_return_bps
expected_shortfall_05_bps
predictive_interval_width_bps
distribution_version
calibrator_version
~~~

Wanneer een gezamenlijke Phase-9 candle-path beschikbaar is:

~~~text
max_adverse_excursion distribution
max_favorable_excursion distribution
P(TP before SL)
P(SL before TP)
P(neither)
path consistency metrics
~~~

TP/SL-informatie mag pas een policyinput worden nadat de executionsemantiek voor intrapath highs/lows ondubbelzinnig is vastgelegd.

Conceptuele toekomstige policy-evidence:

~~~text
direction is long/short
AND calibrated directional confidence is sufficient
AND P(net_return > 0) is sufficient
AND expected net return is sufficient
AND downside tail is acceptable
AND OOD status is not blocked
AND meta-labeler accepts
→ signal
otherwise
→ no-trade
~~~

De thresholds moeten uitsluitend op inner/OOF development data worden gekozen.

Bij vergelijking van twee policies of twee modelvarianten moeten waar mogelijk dezelfde Monte Carlo random draws worden gebruikt. Dat vermindert ruis in A/B-vergelijkingen en voorkomt dat een kandidaat "wint" door een toevallig gunstig random seed.

### Phase 13 — strategy Monte Carlo en stresstests

Dit is de tweede grote toepassing.

Financiële returns en modelerrors zijn tijdsafhankelijk. Random individuele trades opnieuw trekken vernietigt:

- autocorrelatie;
- volatility clustering;
- regimeclusters;
- sequences van verliezen;
- tijdelijke modeldegradatie.

Gebruik daarom een **moving-block** of **stationary bootstrap** over chronologische OOF/developmentperioden.

Bij voorkeur resamplen we chronologische decision/prediction records of tijdsblokken en voeren daarna de bestaande backtester opnieuw uit.

Niet alleen:

~~~text
final_trade_returns → random shuffle
~~~

Wel:

~~~text
chronological OOF prediction blocks
        ↓ bootstrap
same decision/execution rules
        ↓
backtester v1
~~~

Per bootstrap-run bewaren:

- cumulative net bps;
- mean net bps/trade;
- max realized-exit drawdown bps;
- profit factor;
- hit rate;
- trade count;
- signal coverage;
- exposure fraction;
- worst horizon;
- worst regime;
- base/stress difference.

Samenvatting over alle runs:

- median;
- 5e / 25e / 75e / 95e percentiel;
- probability of negative aggregate result;
- probability drawdown boven vooraf gedefinieerde limiet;
- expected shortfall van strategy outcome;
- sensitivity aan block length;
- sensitivity aan random seed;
- sensitivity aan base versus stress costs.

**Opmerking:** de huidige backtester rapporteert arithmetic fixed-notional bps en realized-exit drawdown, geen compounded account equity. Noem dit dus niet "account risk of ruin" zolang er geen formeel account/position-sizingmodel bestaat.

---

## 6. Samplingstrategie

### 6.1 Geen default Gaussian-aanname

XAU/USD heeft perioden met heavy tails, volatility clustering en regimeveranderingen. Een vaste Normal(mu, sigma) mag daarom niet zonder OOF evidence de standaard worden.

### 6.2 Voorkeursvolgorde

1. Directe gekalibreerde modeldistributie, als Phase 9 die betrouwbaar levert.
2. Empirical OOF residual bootstrap.
3. Robuuste parametric residual family wanneer zij aantoonbaar beter gekalibreerd is.
4. Complexere mixtures alleen als een simpele distributie aantoonbaar tekortschiet.

### 6.3 Regimeconditioning

Conditioneer niet te fijn.

Een bruikbare eerste versie kan per:

- horizon;
- brede volatility bucket;
- eventueel event/non-event;

residuals trekken.

Wanneer een bucket onvoldoende observaties heeft:

~~~text
exact bucket
→ bredere regimepool
→ horizonpool
→ expliciete fallback / no-MC
~~~

Deze fallback moet gelogd worden.

### 6.4 Aantal draws

Niet op intuïtie één gigantisch getal vastzetten.

Voer tijdens development een convergence test uit, bijvoorbeeld met:

- 1.024;
- 4.096;
- 16.384 draws.

Kies daarna het kleinste budget waarbij de policy-relevante statistieken voldoende stabiel zijn.

Offline research mag duurder zijn dan live inference.

---

## 7. Monte Carlo Dropout: voorlopig niet aanbevolen als hoofdroute

MC Dropout is een geldige techniek om epistemische onzekerheid uit dropoutnetwerken te benaderen, maar het huidige Phase-8 GRU-model bevat geen dropout.

Dropout toevoegen enkel om MC Dropout mogelijk te maken zou:

- de frozen Phase-8 architectuur veranderen;
- retraining vereisen;
- opnieuw predictive admission moeten passeren;
- onzekerheid en predictive performance tegelijk wijzigen.

Daarom:

- **niet retrofitten in Phase 8**;
- Phase-12 ensemble-disagreement eerst gebruiken als praktische epistemische uncertainty proxy;
- MC Dropout eventueel als afzonderlijke Phase-13 challenger testen;
- alleen behouden indien het bovenop de eenvoudigere ensemble/OOF uncertainty aantoonbaar waarde toevoegt.

---

## 8. Integratie met de bestaande decision policy

De huidige backtester gebruikt point expected return en confidence.

De nieuwe Monte Carlo summary moet eerst als **additionele kolommen** aan backtestrecords worden toegevoegd, zodat baseline en challenger exact dezelfde execution engine gebruiken.

Bijvoorbeeld:

~~~text
p_net_positive_base
p_net_positive_stress
mc_expected_net_bps_base
mc_expected_net_bps_stress
mc_q05_net_bps
mc_expected_shortfall_05_bps
mc_interval_width_bps
mc_distribution_version
mc_seed
~~~

Vervolgens kunnen twee policies eerlijk worden vergeleken.

Baseline:

~~~text
current DecisionPolicy(
    confidence_threshold,
    min_expected_net_bps
)
~~~

Challenger:

~~~text
MonteCarloDecisionPolicy(
    calibrated_confidence_threshold,
    min_p_net_positive,
    min_mc_expected_net_bps,
    max_downside_tail,
    ...
)
~~~

De bestaande cash/no-trade fallback blijft verplicht.

---

## 9. Promotiecriteria

Monte Carlo mag niet worden gepromoveerd omdat één backtest mooier oogt.

De concrete thresholds moeten in het Phase-12/13 protocol vooraf worden bevroren, maar de structurele gate hoort minimaal te vereisen:

1. dezelfde OOF/outer samples als de baseline;
2. geen gebruik van de finale 2025+ holdout;
3. voldoende aantal beslissingen en coverage;
4. stabieler of beter mean economic result over folds;
5. geen onaanvaardbare regressie in de slechtste fold;
6. robuustheid onder stresskosten;
7. geen winst die uitsluitend uit één horizon/regime komt;
8. gekalibreerde P(net > 0) die empirisch betrouwbaar genoeg is;
9. reproduceerbaarheid over seeds;
10. acceptabele runtime;
11. volledige artifact- en configtraceerbaarheid.

Wanneer deze gate faalt:

~~~text
keep calibrated baseline policy
~~~

Niet:

~~~text
tune Monte Carlo harder totdat hij wint
~~~

---

## 10. Tests

Voorgestelde testset:

~~~text
tests/test_uncertainty_contracts.py
tests/test_predictive_distribution.py
tests/test_monte_carlo_sampling.py
tests/test_monte_carlo_reproducibility.py
tests/test_monte_carlo_calibration.py
tests/test_monte_carlo_policy.py
tests/test_monte_carlo_strategy_bootstrap.py
tests/test_monte_carlo_no_leakage.py
~~~

Verplichte invarianten:

- zelfde seed + zelfde artifact versions → zelfde simulation summary;
- geen observation na prediction_time_utc mag samplerparameters beïnvloeden;
- OOF residuals mogen nooit van het eigen record/in-sample fit afkomstig zijn;
- probabilities blijven binnen [0, 1];
- quantielen zijn monotonic;
- gesimuleerde OHLC-candles respecteren OHLC-invarianten;
- path timestamps blijven causaal en correct uitgelijnd;
- base/stress costs worden exact één keer toegepast;
- cash/no-trade blijft geldig;
- resampling verandert de inhoud van blocks niet;
- strategy bootstrap behoudt chronologische volgorde binnen elk block;
- finale holdout kan door MC-code niet stilzwijgend worden geopend.

---

## 11. Reproduceerbaarheid en artifacts

Voorgesteld configbestand:

~~~text
configs/monte_carlo.yaml
~~~

Minimaal versioneren:

~~~yaml
schema_version: 1
protocol_version: monte-carlo-v1
sampler: ...
draw_count: ...
seed: ...
tail_probability: ...
distribution_source: ...
block_bootstrap:
  method: ...
  block_length: ...
~~~

Per formele run opslaan:

- volledige config;
- seed;
- RNG/library versions;
- source model hash;
- calibrator hash;
- OOF store hash;
- sampler/distribution version;
- policy version;
- draw count;
- convergence diagnostics;
- per-forecast summaries;
- strategy-bootstrap summaries;
- code commit;
- data version.

De ruwe miljoenen random draws hoeven standaard niet permanent te worden bewaard wanneer ze deterministisch reproduceerbaar zijn. Bewaar de inputs, seed, versie en samenvattingen; bewaar volledige draws alleen voor geselecteerde audit/debugcases.

---

## 12. Voorgestelde repositorystructuur

Na volledige implementatie:

~~~text
src/gold_forecasting/
    uncertainty/
        __init__.py
        contracts.py
        distributions.py
        calibration.py
        monte_carlo.py
        resampling.py
        summaries.py
    decision/
        __init__.py
        policy.py

configs/
    monte_carlo.yaml

docs/
    monte_carlo_integration_proposal.md
    research_protocol_monte_carlo.md

tests/
    test_uncertainty_contracts.py
    test_predictive_distribution.py
    test_monte_carlo_sampling.py
    test_monte_carlo_reproducibility.py
    test_monte_carlo_policy.py
    test_monte_carlo_strategy_bootstrap.py
    test_monte_carlo_no_leakage.py
~~~

De exacte modulegrenzen mogen tijdens Phase 11/12 nog worden aangepast om duplicatie met calibratie- en decisioncode te vermijden.

---

## 13. Aanbevolen implementatieslices

### Slice MC-0 — nu

Alleen dit voorstel bewaren.

Geen production code, geen nieuwe threshold en geen benchmark opnieuw draaien.

### Slice MC-1 — tijdens Phase 9

- distribution contracts;
- future-path sampling contract;
- quantile/distribution diagnostics;
- synthetische sampler-tests;
- nog geen economische policy-integratie.

### Slice MC-2 — Phase 11

- OOF residual/distribution store;
- distribution calibration;
- PIT/coverage/CRPS;
- calibratorversioning;
- no-leakage tests.

### Slice MC-3 — vroege Phase 12

- vectorized per-forecast Monte Carlo sampler;
- deterministic seeds;
- base/stress net-return summaries;
- downside quantiles / expected shortfall;
- convergence tests.

### Slice MC-4 — midden Phase 12

- Monte Carlo summary koppelen aan meta-labeler/OOD/decision policy;
- baseline versus challenger op exact dezelfde OOF samples;
- cash fallback behouden.

### Slice MC-5 — Phase 13

- moving-block/stationary bootstrap;
- volledige backtester per simulated timeline;
- robustness distributions;
- seed/block-length sensitivity;
- formele promotion gate.

### Slice MC-6 — vóór Phase 14

Wanneer en alleen wanneer MC is gepromoveerd:

- freeze sampler;
- freeze distribution/calibrator;
- freeze draw budget;
- freeze thresholds;
- freeze bootstrapmethode;
- hash alle artifacts;
- pas daarna de finale holdout éénmalig openen.

### Slice MC-7 — Phase 15

- profile inference latency;
- vectorize/batch sampling;
- reduce live draw count alleen na convergence-equivalentie;
- log simulation summary per append-only prediction;
- nooit live random settings stilzwijgend wijzigen.

---

## 14. Verwachte impact

### Raw forecasting accuracy

**Verwachte impact: laag.**

Monte Carlo verandert de onderliggende feature-extractie of predictive edge niet automatisch.

### Probability/uncertainty usefulness

**Verwachte impact: hoog**, mits Phase 11 echte calibratie oplevert.

### No-trade/selectiviteit

**Potentieel hoge impact.**

Dit is waarschijnlijk de belangrijkste toepassing voor Aurum: niet vaker handelen, maar beter herkennen wanneer de model-edge te onzeker of te klein is na kosten.

### Tail-risk en drawdowninzicht

**Hoge impact voor researchvalidatie.**

Een enkele historische backtestcurve geeft geen goede verdeling van mogelijke path/orderings. Block-bootstrap Monte Carlo kan zichtbaar maken hoe kwetsbaar resultaten zijn voor loss clustering en regimevolgorde.

### TP/SL en path-aware beslissingen

**Potentieel hoog voor korte horizons**, maar pas nadat Phase 9 de gezamenlijke future-candle-path betrouwbaar genoeg voorspelt.

---

## 15. Eindbesluit

Monte Carlo verdient een plaats in Aurum, maar de correcte positie is **na probabilistische forecasting en calibratie, vóór/naast de beslispolicy en als aparte robustnesslaag rond de backtester**.

De huidige repository is nog net te vroeg om Monte Carlo als echte tradefilter te activeren:

- Phase-8 probabilities zijn nog ongekalibreerd;
- Phase 8 heeft geen economic promotion candidate;
- de huidige neural outputs bevatten nog geen volledige predictive distribution;
- Phase 9 en Phase 11 zijn al ontworpen om precies deze prerequisites te leveren.

Daarom is de optimale keuze:

> **bouw de distributionele prerequisites in Phase 9, valideer en kalibreer ze OOF in Phase 11, implementeer predictive Monte Carlo in Phase 12 en strategy Monte Carlo in Phase 13.**

Zo wordt Monte Carlo een gecontroleerde verbetering van **decision quality, uncertainty awareness en robustness**, in plaats van extra complexiteit zonder bewezen edge.

---

## 16. Methodologische referenties

- Gneiting, Balabdaoui & Raftery (2007), *Probabilistic forecasts, calibration and sharpness*, JRSS B, DOI: 10.1111/j.1467-9868.2007.00587.x.
- Politis & Romano (1994), *The Stationary Bootstrap*, Journal of the American Statistical Association, DOI: 10.1080/01621459.1994.10476870.
- Lahiri (2003), *Resampling Methods for Dependent Data*, Springer.
- Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML/PMLR 48. Relevant als latere MC-Dropout-challenger, niet als aanbevolen huidige hoofdroute.
- Basel Committee market-risk terminology gebruikt Expected Shortfall als gemiddelde tail loss voorbij een gekozen VaR-grens. Aurum gebruikt dit concept alleen als research risk summary; dit maakt het systeem niet tot een gereguleerd bank-riskmodel.
