# Does open LiDAR rank real flooding better than the global terrain? (19 Oct 2026)

Test: `scripts/lidar_flood_benchmark.py` (reproduces `data/backtest/lidar_flood_benchmark.json`).
Truth: Environment Agency Recorded Flood Outlines, fluvial events (Open Government Licence). 12 English towns that have flooded: for each, 120 random points inside
recorded outlines and 120 random points at least 100 m outside them, within the same 5 km box. The same points are scored from two terrains of the same place:
the engine's previous global tiles (AWS Terrarium, about 10 m) and the Environment Agency 1 m bare-earth LiDAR composite (resampled to about 6 m).
Scores are used unfitted, lower = more flood-prone: HAND (height above nearest drainage, 1 km2 channels) and height above the lowest ground within 1 km.
Metric: ROC area under the curve (0.5 = no skill, 1 = perfect separation).

| Town | HAND, global | HAND, LiDAR | Height above local low, global | Height above local low, LiDAR |
|---|---|---|---|---|
| Carlisle | 0.47 | 0.95 | 0.51 | 0.95 |
| Cockermouth | 0.88 | 0.99 | 0.91 | 1.00 |
| York | 0.61 | 0.81 | 0.62 | 0.77 |
| Tadcaster | 0.68 | 0.92 | 0.72 | 0.91 |
| Hebden Bridge | 0.85 | 0.99 | 0.89 | 1.00 |
| Shrewsbury | 0.60 | 0.97 | 0.66 | 0.99 |
| Appleby | 0.77 | 0.81 | 0.78 | 0.90 |
| Lewes | 0.75 | 0.98 | 0.79 | 1.00 |
| Kendal | 0.85 | 0.95 | 0.87 | 1.00 |
| Morpeth | 0.82 | 0.98 | 0.84 | 0.97 |
| Upton upon Severn | 0.85 | 0.89 | 0.78 | 0.80 |
| Tewkesbury | 0.69 | 0.98 | 0.68 | 0.98 |
| **Median** | **0.76** | **0.96** | **0.78** | **0.97** |

LiDAR scored higher in all 12 towns on both scores. The largest gain is where the global terrain has no skill at all (Carlisle 0.47 to 0.95; Shrewsbury 0.60 to 0.97).

Share of sampled points inside a recorded outline, by LiDAR HAND (the sample is balanced 50/50 inside and outside by construction, so read these as relative likelihoods, not probabilities of flooding):
HAND below 1 m: 86-95 %; 1-5 m: 79-88 %; 5-10 m: 42 %; 10 m or more: 3 %. The global-terrain HAND separates far less (69 % below 0.5 m, 23 % above 10 m).

What this does and does not show
- It shows that ground-level detail from LiDAR separates flood-affected from unaffected ground within a flood-prone landscape far better than 10-30 m terrain. It does not give a flood depth, a return period, or the chance that a random place floods.
- Not independent in three ways: the thresholds in `terrain/lidar_flood.py` and the England golden sites come from the same 12 towns; recorded outlines are where floods were mapped (often near rivers and towns); the Environment Agency's own flood mapping may share survey and terrain inputs with its LiDAR. Treat 0.96 as an upper bound for a new town.
- Modern terrain for past floods: defences built after an event are in the LiDAR, which would, if anything, lower the score.
- England only. The same method runs on Dutch AHN LiDAR for elevation, but HAND is not used in pumped polders. In the US, the global tiles already contain 10 m national LiDAR-derived terrain; 1 m 3DEP is not wired in.

In the engine: `connectors/lidar_dem.py` supplies the grid where open LiDAR exists (England 1 m, Netherlands 0.5 m) and falls back to the global tiles elsewhere or on any failure; `terrain/lidar_flood.py` turns it into a floodplain-position screen that holds the river flood rating at LOW (HAND under 10 m), MODERATE (under 5 m) or HIGH (under 1 m) and says so in the report. On the golden sites it brings Carlisle, Cockermouth and Tewkesbury to MODERATE or HIGH.
