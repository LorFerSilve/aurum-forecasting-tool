# Lokale datalagen

- `raw/`: originele providerbestanden; immutable en gehasht.
- `curated/`: gevalideerde UTC-candles.
- `features/`: afgeleide, point-in-time correcte features.
- `labels/`: logisch/fysiek van features gescheiden toekomstige targets.
- `oof_predictions/`: uitsluitend out-of-foldvoorspellingen voor latere fasen.

Gegenereerde data wordt niet in Git opgenomen. Iedere directory bevat alleen een placeholder zodat de laagstructuur zichtbaar blijft.

Het geharde profiel (`configs/phase5.yaml`) deelt de immutable raw archieven met de MVP,
maar schrijft eigen manifests en curated bestanden onder
`curated/histdata/XAU_USD/phase5/<timeframe>/utc_year=<jaar>/`.
De MVP behoudt haar bestaande `source_year`-partities. `phase5_build.json` wordt pas na
data en kwaliteitsrapporten gepubliceerd en is vereist voor hergebruik.
Zie [het fase-5-datacontract](../docs/data_contract_phase5.md) voor kalenderkennis,
lege dag-/maanddatasets, afzonderlijke logicaversies en dagelijkse kwaliteitsrapporten.
