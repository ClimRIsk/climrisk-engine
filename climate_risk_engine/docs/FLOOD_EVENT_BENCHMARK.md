# Flood event benchmark

Question: does the model's terrain flood layer rank the places that actually flooded above the places that did not?

**Truth.** (1) Copernicus EMS Rapid Mapping observed flood extents, 72 events 2023–2026, 125 windows of 8 km (19 events outside Europe). (2) England Environment Agency Recorded Flood Outlines (Open Government Licence), 216 fluvial windows, 24 per decade, 1940s–2020s.

**Model under test.** Terrain only (HAND at 0.1 and 1 km² channel thresholds, elevation relative to the 8 km surroundings). No imagery from the event is used, so there is no leakage.

**Metric.** Per-window ROC AUC (0.5 = chance), averaged, with a 95 % bootstrap over events.

| Test | Old rule (built-up land < 3 m / 6 m) | New event-trained model |
|---|---|---|
| Held-out CEMS events (24 events, never fitted on) | 0.55 (0.52–0.59) | 0.76 (0.72–0.80) |
| Fitted on Europe, scored outside Europe (19 events) | 0.55 | 0.71 |
| England 1940s–2020s outlines (model fitted only on CEMS) | 0.57 (0.55–0.58) | 0.81 (0.79–0.82) |

Skill does not fall with event age (Spearman of AUC vs year −0.01, p = 0.87).

Flooded share of cells by model rank band (CEMS): lowest fifth 0.6 %, top two fifths about 20 % against a 9.7 % average.

**Limits.** AUC ≈ 0.8 is ranking skill, not a classifier. Windows are in flood-affected landscapes, so this does not measure how often a random place floods. SAR under-maps urban and vegetated flooding. Modern terrain is used for old floods. A plain "share of surroundings lying below the site" scores as well as the fitted model (0.81 vs 0.81 England), so the extra weights add little. Depth/magnitude is not tested: no machine-readable depth truth. England is the only historic country so far.

Reproduce: `scripts/flood_event_benchmark.py` stages `catalog, extents, features, evaluate, uk_fetch, uk_events, uk_features, history_eval`. Raw cache (~2.6 GB) is under `data/backtest/raw/events/` and is not committed.
