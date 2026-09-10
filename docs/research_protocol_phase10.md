# Researchprotocol — fase 10 externe context

**Strict protocol:** `phase10-context-v1`  
**Exploratory silver protocol:** `phase10-silver-modeled-v1`  
**Exploratory dollar protocol:** `phase10-dollar-eurusd-modeled-v1`  
**Status:** silver afgerond (`stop`); dollarcontract vooraf vastgelegd, marktbenchmark nog gesloten.

## Doel

Fase 10 test externe context één bron tegelijk bovenop de bevroren price-only
researchchampion. Een databron wordt niet toegelaten omdat zij economisch plausibel klinkt;
zij moet point-in-time traceerbaar zijn en op exact dezelfde development folds aantoonbaar
meerwaarde leveren.

De bronvolgorde blijft:

1. zilver;
2. dollarproxy;
3. renteproxy;
4. CPI VS;
5. Amerikaanse arbeidsmarktdata;
6. FOMC/rentebesluiten.

Brede nieuws-NLP, social sentiment en historische consensus/surprise blijven buiten scope
tot hun eigen historische point-in-time bewijs is geleverd.

## Frozen price-only reference

De eerste silver ablation gebruikt uitsluitend de **15min** researchhorizon en vergelijkt
met de canonieke Phase-7 champion:

- run: `20260907T014255645674Z-b2afe281`;
- benchmarkcode: `3f0a703568224fe9169b1e9f8d61dad131f0005b`;
- completion:
  `sha256:beac58092d06350bb067fbd2df144bb9cb2507f45053c19487474c87d0ceca0f`;
- variant/family: `mvp/logistic`.

Phase 8 en Phase 9 hebben deze 15min champion niet vervangen. Phase-9 direct blijft alleen
een research/distributionele output en wordt niet als baseline voor de silver feature
ablation gebruikt.

## Development-only tijdscontract

De finale holdout vanaf **2025-01-01 UTC** blijft gesloten.

De eerste silver ablation gebruikt dezelfde frozen outer testjaren als Phase 7:

- 2022;
- 2023;
- 2024.

Dezelfde **181 minuten** train-side gap/purge discipline wordt behouden. Inner folds,
outer folds en sample-ID-semantiek mogen niet worden aangepast na het zien van silver
resultaten.

De silver bron mag de gold sample universe niet verkleinen. Elke gold sample blijft
bestaan. Wanneer silver ontbreekt, stale is of onvoldoende featurehistorie heeft, wordt
exact de price-only fallback gebruikt.

## Point-in-time contract

Iedere scalar contextwaarneming bewaart:

- `source_id`;
- `observation_id`;
- `observed_at_utc`;
- `available_at_utc`;
- `ingested_at_utc`;
- `value`;
- `revision_id`;
- `source_uri`;
- `raw_sha256`.

Een contextwaarde mag uitsluitend worden gekoppeld wanneer haar specifieke versie op de
prediction cutoff al beschikbaar was. De join is backward-only. Latere revisies mogen
eerdere prediction rows niet herschrijven.

`age_seconds` wordt berekend vanaf `observed_at_utc`, niet vanaf de revisie- of
downloadtijd. Een late revisie maakt oude data dus niet kunstmatig vers.

## Twee strikt gescheiden evidence-niveaus

### 1. Strict PIT

Een bron kan alleen kandidaat zijn voor formele Phase-10 admission wanneer
`availability_basis=provider_timestamp` is ondersteund door controleerbare historische
release/vintage-evidence.

Een huidige downloadtijd, een publicatieschema of een gemodelleerde latency is hiervoor
niet voldoende.

### 2. Exploratory modeled latency

HistData XAGUSD levert bruikbare historische bid-OHLC, maar geen historische per-row
release timestamps. Daarom gebruikt het afzonderlijke protocol
`phase10-silver-modeled-v1`:

- HistData M1 timestamp wordt als candle-open geïnterpreteerd;
- bronzone is vaste UTC−05:00 zonder DST;
- `observed_at_utc` is candle close;
- `available_at_utc = candle_close + 60s`;
- deze 60 seconden zijn een onderzoeksaanname;
- de project-ingestiontijd wordt apart opgeslagen en bewijst geen historische beschikbaarheid.

Een positief exploratory resultaat mag **geen champion promoten** en mag niet als strict
point-in-time bewijs worden beschreven. Het kan alleen rechtvaardigen dat een betere
historische bron gezocht of aangeschaft wordt.

## XAGUSD data-ingestie

De frozen XAUUSD HistData-adapter blijft ongewijzigd. XAGUSD krijgt een afzonderlijke
Phase-10 adapter zodat instrumentidentiteit, ZIP-membernaam, bron-URL, hashes en metadata
niet door hernoemen kunnen worden vervalst.

De lokale import:

- accepteert uitsluitend `DAT_ASCII_XAGUSD_M1_<period>.csv` in de ZIP;
- controleert archive SHA-256 en CRC;
- weigert encrypted of excessief grote members;
- controleert source-year in vaste UTC−05:00;
- vereist positieve, geldige OHLC;
- verwijdert alleen exact dubbele identieke candles;
- weigert conflicterende candles op dezelfde timestamp;
- bewaart `bid_close` als scalar contextwaarde;
- partitioneert real-data observations per jaar als Parquet;
- authenticeert iedere partitie met een manifest;
- bundelt de jaarlijkse manifests in één bundle-set.

Geen bestand uit 2025+ wordt toegelaten.

## Silver featurecontract

De eerste candidate gebruikt alleen causaal afleidbare silvercontext:

- one-step log-return in bps;
- vijf-step momentum in bps;
- gold/silver log-priceratio;
- twintig-return rolling Pearson correlation;
- observation age.

Missing/stale flags zijn routing/auditvelden. Zij mogen niet dienen als substituut voor
een feitelijk ontbrekende bron.

Een repeated as-of quote is geen nieuwe candle en produceert dus geen kunstmatige
0-return. Een timestampgap reset momentum/correlationhistorie.

## Model-isolatie

De eerste silver ablation test primair **de databron**, niet een nieuwe modelarchitectuur.

Daarom:

- model family: logistic regression;
- dezelfde MVP price-only featurebasis als de frozen 15m Phase-7 champion;
- silver features worden alleen aan de challenger toegevoegd;
- logistic grid blijft `C ∈ {0.1, 1.0, 10.0}`;
- class weighting blijft exact de frozen Phase-7 instelling: `balanced`;
- selectiecriterium blijft validation macro-F1 met log-loss tie-break;
- preprocessing wordt uitsluitend op train gefit;
- geen outer-teststatistiek beïnvloedt imputation, scaling, selectie of policy.

Wanneer de silver candidate op een sample niet bruikbaar is, worden exact de frozen
price-only probabilities voor die sample gebruikt.

## Primaire predictive gate

Voor strict-PIT admission wordt dezelfde Phase-7 predictive gate behouden. Tegenover de
frozen price-only reference moet de candidate:

1. hogere **mean macro-F1** hebben;
2. **worst-fold macro-F1** niet verslechteren;
3. mean **Brier score** niet verslechteren.

Daarnaast worden log loss, ECE/calibration error, coverage en usable-context fraction
gerapporteerd, maar na de eerste benchmark wordt geen nieuwe numerieke admissiondrempel
verzonnen.

Voor het exploratory modeled-latency protocol worden dezelfde metrics berekend, maar zelfs
een geslaagde gate heeft uitsluitend de status `exploratory_signal_worth_strict_source`.

## Economische gate

Wanneer een strict-PIT candidate later economische toelating krijgt, blijft de Phase-7
economic gate gelden:

- worst-fold trade count minstens 20;
- worst-fold base net bps > 0;
- mean stress net bps > 0;
- worst-fold base net bps minstens zo goed als de baseline.

Policyselectie gebeurt uitsluitend in inner folds onder dezelfde execution assumptions.
Een no-trade fallback blijft geldig en een negatief resultaat is een geaccepteerde
onderzoeksuitkomst.

Een modeled-latency exploratory run kan nooit economic promotion activeren.

## Verplichte rapportage

Per outer fold en run-level:

- exacte common gold sample digest;
- usable / missing / stale context rows;
- fallback row count en fraction;
- maximum en distributie van age;
- candidate vs frozen reference macro-F1;
- worst-fold macro-F1;
- Brier;
- log loss;
- ECE/calibration error;
- return-MAE indien dezelfde comparable return output beschikbaar is;
- base/stress economische metrics wanneer policy-evaluatie wordt uitgevoerd;
- train/validation/test row counts;
- train-only preprocessing evidence;
- source archive hashes en bundle hashes;
- code/config/dependency snapshot;
- expliciete `holdout_opened=false`.

## Stopregels

- Bij onvoldoende historische availability-evidence: geen strict-PIT promotie.
- Bij lage usable coverage: bron niet kunstmatig forward-fillen om coverage te verhogen.
- Bij predictive regressie: price-only champion behouden.
- Bij alleen één goede fold: geen promotie.
- Bij betere accuracy maar slechtere macro-F1: geen promotie.
- Bij economic no-trade: geen tradingclaim.
- Bij modeled latency: resultaten uitsluitend exploratory noemen.
- De 2025+ holdout blijft gesloten tot de latere, volledig bevroren finale fase.

## Eerstvolgende uitvoeringsvolgorde

1. bouw en verifieer lokale XAGUSD 2020–2024 archive bundles;
2. draai source preflight;
3. controleer coverage/staleness en exacte UTC-alignment;
4. bouw frozen Phase-7 15m reference parity;
5. implementeer de logistic silver challenger met inner-only selectie;
6. voer eerst een synthetic/integration benchmark uit;
7. voer daarna de real-data modeled-latency exploratory ablation uit;
8. neem een expliciet `stop / seek_strict_source` besluit;
9. alleen met echte historische release-evidence mag een strict-PIT silver benchmark worden geopend.

Geen stap opent de finale holdout.

## Bron 2 — inverse EURUSD-dollarproxy (2026-09-10)

De [bronvergelijking en acquisitie-instructies](phase10_dollar_source_research.md)
onderbouwen de expliciete keuze voor HistData EURUSD M1. De hypothese is bilaterale
dollarsterkte tegenover de euro. Deze bron is geen ICE DXY en geen brede dollarindex.
Voor echte DXY ontbreken lokaal een concrete geauthenticeerde levering, aantoonbare
gebruiksrechten en historische release/correctie-evidence. HistData UDXUSD heeft
bovendien onvoldoende gedocumenteerde indexidentiteit.

Silver run `20260909T133304370198Z-be3c4015` blijft afgesloten met `stop`; silver wordt
niet met dollar gecombineerd. De volgende contracten gelden ongewijzigd voor dollar:

- dezelfde frozen Phase-7 15m `mvp/logistic` reference en completion hierboven;
- dezelfde gold rows, waarden, sample-ID-semantiek en common sample digests;
- outer testjaren 2022/2023/2024, exact dezelfde inner folds en 181 minuten gap/purge;
- dezelfde MVP price-featurebasis, logistic `C={0.1,1.0,10.0}`, `class_weight="balanced"`;
- train-only preprocessing; model- en policyselectie uitsluitend op inner folds;
- dezelfde predictive/economic gates, kosten en no-trade policy;
- exact persisted Phase-7 probabilities bij missing/stale, onvoldoende historie of
  een onbruikbaar contextmodel; de gold universe wordt niet op context gefilterd;
- `champion_promotion=false`, `trading_activation=false`, `holdout_opened=false`.

De ruwe scalar `dollar_value` blijft EURUSD bid-close (USD per EUR), met
`source_id=dollar`, originele `eurusd-m1-...` observation IDs en EURUSD archive-hashes.
De adapter accepteert uitsluitend de vastgelegde EURUSD/source-binding. Zij maakt geen
inverse executable bid/ask en reconstrueert geen spread, volume of microstructure.
Timestamp-open is een expliciete interpretatie; vaste UTC-05 zonder DST, observed close
= open + 1min, modeled availability = close + 60s, stalegrens 600s. Downloadtijd is
afzonderlijke provenance en bewijst geen historische release.

De vijf modelkolommen zijn vóór de benchmark bevroren, op de bestaande 3min goldcadans:

| Feature | Causale definitie |
|---|---|
| `dollar_return_1_bps` | `-10000 * diff(log(EURUSD))` over één opeenvolgende beschikbare observatie |
| `dollar_momentum_5_bps` | Dezelfde negatieve logverandering over vijf stappen (15min) |
| `gold_eur_return_1_bps` | Gold-logreturn plus dollar-logreturn: indicatieve verandering van gold in EUR uit beschikbare bids |
| `gold_dollar_correlation_20` | Trailing Pearson-correlatie van twintig gepaarde gold/dollarreturns |
| `dollar_age_seconds` | Prediction cutoff minus gebruikte observatietijd |

De gold/EUR-interactie is indicatief: de as-of EURUSD-bar kan ouder zijn dan de goldbar.
Alle modelkolommen moeten eindig zijn. `dollar_is_missing`, `dollar_is_stale` en
`dollar_usable` zijn routing/audit, geen vervangende modelinputs. Een herhaalde quote
geeft geen nieuwe nulreturn. Prediction- of observatiegaps starten de benodigde vensters
opnieuw; correlatie vereist 21 opeenvolgende niveaus. Er wordt niets geïnterpoleerd.

Het protocol kan uitsluitend exploratory draaien. Het wijzigen van een config naar
`provider_timestamp` of een strict protocol faalt gesloten; eventuele strict-PIT
admission vereist later een afzonderlijk geverifieerd contract en historische evidence.
Een geslaagde exploratory predictive gate kan alleen `seek_strict_source` rechtvaardigen.

De gemeenschappelijke preflight authenticeert eerst alle jaarmetadata, vervolgens
archive/bundle-parity en de canonieke Phase-7 reference, goldwaarden en folds. Alleen na
volledig groene preflight op een clean commit mag RunRegistry worden gestart. Ontbrekende
lokale EURUSD-bestanden blokkeren dit; er wordt niets automatisch gedownload. De runner
bewaart bron/config/code/dependencies, features, alle inner/outer voorspellingen,
train-audits, policies en base/stress metrics. Completion wordt pas na semantische replay
verzegeld; de validator herhaalt hashes, features, fallback, selectie en gates.

De bron moet na die nog uit te voeren ablation haar eigen `stop/seek_strict_source`
besluit krijgen. De 2025+ holdout blijft gesloten en PR #4 blijft draft zolang Phase 10
niet volledig is afgerond.
