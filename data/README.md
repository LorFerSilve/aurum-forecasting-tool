# Lokale datalagen

- `raw/`: originele providerbestanden; immutable en gehasht.
- `curated/`: gevalideerde UTC-candles.
- `features/`: afgeleide, point-in-time correcte features.
- `labels/`: logisch/fysiek van features gescheiden toekomstige targets.
- `oof_predictions/`: uitsluitend out-of-foldvoorspellingen voor latere fasen.

Gegenereerde data wordt niet in Git opgenomen. Iedere directory bevat alleen een placeholder zodat de laagstructuur zichtbaar blijft.

