# Insurer view vs FEMA National Risk Index (19 Oct 2026)

Engine: `insurance.run` on one pin per county, USD 10 m sum insured, cement sector, 3,000 simulated years (`tests/golden/insurer_snapshot_2026-10-19.json`).
FEMA: National Risk Index county building expected annual loss divided by county building value (public ArcGIS service; `python tools/benchmark_fema_nri.py` reproduces the table).
NRI is a county average and a HAZUS-based model, not claims. The engine value is one pin and includes deductibles. Read orders of magnitude only.

| County (pin) | Peril | Engine, per mille | NRI, per mille | Engine / NRI |
|---|---|---|---|---|
| Miami-Dade (Miami) | wind | 2.48 | 0.81 | 3.1 |
| | inland flood | 0 | 0.75 | 0 |
| | coastal flood | 0 | 0.05 | 0 |
| Orleans (New Orleans) | wind | 3.94 | 3.23 | 1.2 |
| | inland flood | 0.20 | 0.69 | 0.3 |
| | coastal flood | 0 | 0.02 | 0 |
| Harris (Houston) | wind | 1.82 | 0.51 | 3.6 |
| | inland flood | 0.80 | 1.30 | 0.6 |
| | coastal flood | 0 | 0.003 | 0 |

Reading: wind is the right order of magnitude and sits at or above NRI, as expected for a coastal pin against a county average. Inland flood is under NRI everywhere and zero at Miami: NRI's inland flooding includes rainfall-driven surface-water flooding, which the engine does not model (known gap). Coastal flood is tiny in NRI at all three, so the engine's zero costs little here. Three counties, one pin each, and NRI is itself a model: this shows no gross wind error and confirms the flood gap; it does not validate the engine.
