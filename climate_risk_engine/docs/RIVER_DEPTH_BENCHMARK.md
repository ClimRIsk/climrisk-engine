# Small/mid river depth (HAND + rating curve) — benchmark and decision

**Model:** `finance/river_depth.py` → `site_river_depth`. LiDAR HAND × GloFAS return-level flow (Open-Meteo flood API, 1991–2023, Gumbel) → Manning stage in a regime-width channel → depth at the pin = stage − HAND.

**Test:** `scripts/river_depth_benchmark.py`, 11 flooded English towns (12th, Upton upon Severn, produced no drainage network), 150 points inside Environment Agency Recorded Flood Outlines vs 150 outside, same box, 1 m LiDAR. Question: is the wet/dry **edge** in the right place? (Rank is the HAND rank, already 0.96 AUC.) Critical success index CSI = hits / (hits + misses + false alarms).

| Rule | Pooled CSI | Pooled hit rate | Median town CSI |
|---|---|---|---|
| Model, 1-in-100 depth > 0 | 0.37 | 0.39 | 0.26 |
| HAND < 1 m (existing HIGH band) | 0.32 | 0.33 | 0.30 |
| **HAND < 5 m** (existing MODERATE band) | **0.76** | **0.86** | **0.76** |

Per town the model ranges from CSI 0.80 (Shrewsbury) and 0.62 (Tadcaster), where GloFAS reads a large river, to 0.09–0.24 at Appleby, Hebden Bridge, Cockermouth, Carlisle and Kendal, where the 5 km GloFAS cell snaps to a tributary (Carlisle 1-in-100 flow 43 m³/s on the Eden, which carries well over ten times that) and the stage comes out at 0.4–1.5 m.

## Decision

- **Not wired into the report.** The physics-flavoured model is worse than the empirical HAND band, and its failure is silent: a small modelled flow looks like a confident "dry". The product keeps the benchmarked HAND floor (`terrain/lidar_flood.py`).
- Kept as a lab function with this result recorded. To make it competitive it needs a flow that belongs to the river at the pin (gauged flows, catchment-area regionalisation, or a flow estimated from upstream area on a larger DEM window), not GloFAS point values, and a hydraulic model of the reach rather than normal depth. That is the next step if this is pursued; nothing here was tuned on these towns.
- Practical consequence worth saying plainly: for a site on LiDAR, "HAND < 5 m" is the best-supported statement we can make about whether the river can reach it (hit rate 0.86, false alarms ~15 %), and it is a susceptibility, not a return-period depth.
