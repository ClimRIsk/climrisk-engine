# Surface-water (rain-on-grid) model — benchmark

**Model:** `finance/pluvial.py` → `site_pluvial`. Local-inertial shallow-water solver (`modellab/flood.simulate`, mass-conserving, checked in `modellab.flood.run_benchmarks`) on the best open terrain (1 m EA LiDAR in England, resampled to ~13 m; global tiles elsewhere), driven by a design storm: ERA5 annual-maximum daily rain → Gumbel return level → share of the daily total falling in the storm (1 h: 0.35, 3 h: 0.50, 6 h: 0.65, an assumption) → pervious ground sheds SCS-CN excess (CN 75), built-up ground (ESA WorldCover share) sheds everything above a 12 mm/h sewer allowance. Wired into the report's surface-water zone as a **second opinion that can only raise the rating**; switch off with `CRI_PLUVIAL=0`.

**Test:** `scripts/pluvial_benchmark.py`, scored on Environment Agency Recorded Flood Outlines with cause "local drainage/surface water" (small polygons only). Positives are points inside an outline; negatives are points 100 m to 1 km outside any recorded outline, so both classes sit in the same neighbourhood. ROC AUC, nothing fitted, 1-in-50 design storm, 150 + 150 points per town.

| Town | Model on LiDAR | Same model, global tiles | Pit (blue-spot) depth | Height above local low (500 m) |
|---|---|---|---|---|
| Oxford | 0.66 | 0.46 | 0.65 | **0.97** |
| Reading | 0.64 | 0.54 | 0.72 | **0.97** |
| Bristol | 0.71 | 0.63 | 0.78 | 0.71 |

Five further towns had no usable small surface-water outline in the box and were skipped.

## What this says (read it straight)

- **LiDAR helps the model in every town** (+0.08 to +0.20 AUC over the global tiles). On the global tiles at Oxford the model has no skill (0.46).
- **The model has modest skill (0.64–0.71) and is beaten by a plain terrain score.** Height above the lowest nearby ground scored 0.97 at Oxford and Reading, and the blue-spot depth beat the model at Reading and Bristol. At Oxford and Reading the recorded "surface-water" outlines sit on low ground next to the river, so that score is partly picking up fluvial floodplain that the outline cause field does not separate; do not read 0.97 as surface-water skill. Three towns is a small sample.
- **Depths are low.** ERA5 daily rain is smooth and the 12 mm/h sewer allowance removes most light storms, so a 1-in-100 storm often gives a few cm. A shallow or dry result is not a clean bill; the report says so. Short convective bursts, sewers, culverts and buildings are not resolved.
- Nothing was tuned on these three towns (storm share, CN and allowance are the values written before the run).

## Use in the product

Screening only. It raises, never lowers, the surface-water rating; every result lists terrain source, design rain, storm share and sewer allowance. It is not a substitute for a drainage study or the national surface-water maps where those exist.

## Update 11 Nov 2026

The 600 m configuration above is still `site_pluvial`. The product now uses `site_surface_water` (about 12 km, 24 h storm, gauge-anchored design rain, drains-working and drains-overwhelmed bounds); see `docs/SURFACE_WATER_CHENNAI.md`. The AUC range above must not be quoted for it; it is re-scored below.

## Re-score of the product model, 17 Nov 2026

`scripts/surface_water_benchmark.py` runs `site_surface_water` (drains overwhelmed, the rating basis; 1-in-50; 24 h storm; EA 1 m LiDAR on a 60 m grid; gauge-anchored design rain) on **the same 150 + 150 points per town** as the table above (same seeds and zones), and scores the plain terrain indices on the same 60 m grid. AUC with a 95 % bootstrap interval; POD and FAR are the share of mapped-outline points and of nearby non-outline points where the model gives at least 5 cm. Nothing was fitted. Raw points are in `data/backtest/surface_water_benchmark.json`.

| Town | Old 600 m model | **Product model** (95 % interval) | POD / FAR at 5 cm | Pit depth | Height above local low |
|---|---|---|---|---|---|
| Oxford | 0.66 | **0.84** (0.80–0.89) | 0.67 / 0.05 | 0.76 | 0.96 |
| Reading | 0.64 | **0.85** (0.80–0.89) | 0.72 / 0.15 | 0.79 | 0.96 |
| Bristol | 0.71 | **0.74** (0.69–0.80) | 0.37 / 0.22 | 0.80 | 0.74 |
| Mean of three | 0.67 | **0.81** | | 0.78 | 0.89 |
| Newcastle (added later) | | 0.88 (0.84–0.92) | 0.44 / 0.09 | 0.68 | 0.97 |

Newcastle has only two mapped outlines (0.006 km2), so its points are tightly clustered; it is shown but left out of the mean. Nine further towns (five from the first set, Gloucester, Exeter, Nottingham, Leicester, Northampton) had no usable small surface-water outline and were skipped. The Oxford and Reading terrain scores are both 0.961 by coincidence; the point arrays differ.

**Read it straight.**
- The product model ranks surface-water outline points clearly better than the old 600 m setup (mean 0.81 against 0.67), in all three towns.
- It is still **not better than a plain terrain index**. Height above the nearest low ground scores 0.96 at Oxford and Reading and equals it at Bristol; blue-spot depth beats it at Bristol. The same caveat as before applies: at Oxford and Reading the mapped "surface-water" outlines sit on low ground next to the river, so the terrain score partly picks up floodplain. What the physical model adds is a depth and a storm, and **depth is not validated here** (no observed depths were used).
- It misses a lot: at 5 cm it wets 37–72 % of the mapped points, and 5–22 % of nearby unmapped points. Unmapped does not mean dry.
- The intervals are too narrow. Points inside one outline are spatially correlated, three towns is a very small sample, and towns are not resampled. Do not quote the intervals as the uncertainty of the model.
- Terrain is modern and floods are historic; sewers are not in the terrain; 60 m cells cannot resolve a street.

## A data bug this benchmark found (fixed 17 Nov 2026)

The first Reading run gave a 1-in-50 daily design rain of **194.8 mm**, against 47.7 mm (ERA5) and 40.2 mm (NASA POWER) for the same place. The cause was the Heathrow gauge series from NOAA GHCN-Daily: four "days" of 255–258 mm (14 Jul 1975, 24 Oct 1977, 19 Aug 1993, 10 Aug 1979), each carrying NOAA's quality flag O (climatological outlier). The engine took the largest of the three sources, so the gauge set the design rain, and an unflagged read would also have been used by every other consumer of the station service.

- Fix: `modellab/stationcheck.py` now asks NCEI for the attribute flags and drops any value whose quality flag is non-blank (`clean_rows`); `rain_design.daily_gauge` reports how many values were dropped and the largest three (`qc_dropped`). A dropped day is left as a gap, which the annual-maximum code never counts as zero. `annual_max` (used for the station-versus-grid checks, including temperature) uses the same filter.
- Effect at Reading: design rain 194.8 mm becomes 58.8 mm. In the unsaved first run the model had AUC 0.85 but POD 0.81 / FAR 0.37 and wet 23 % of cells; with the fix POD 0.72 / FAR 0.15 and 9 % of cells. Rank skill barely moved; depth and false alarms did.
- Check: the Chennai 2 Dec 2015 value (344.9 mm) carries a blank flag and stays; Chennai design rain is unchanged at about 338 mm (1-in-100). One Chennai day (14 Jun 1996, 268.5 mm, flag O) is dropped.
- Limit: a genuine extreme can be flagged by NOAA's outlier check, so the filter can remove a real record. The dropped values are listed in the result so a reader can see them.
- **Not re-run:** any earlier station-versus-grid result that read these stations without the filter (`modellab/studies_evt.py`, `analysis/offshore_wind.py` consumers) should be treated as unchecked until re-run.
